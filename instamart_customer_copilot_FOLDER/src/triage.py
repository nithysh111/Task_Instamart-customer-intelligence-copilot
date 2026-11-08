"""Part C: privacy-first ticket triage, retries, validation and routing."""
from pathlib import Path
import json
import random
import re
import time

import pandas as pd

import mock_llm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
CLASSIFIED_STATUSES = {"classified_json", "classified_keyword"}
HUMAN_REVIEW_STATUSES = {
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted",
}

PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO = (
    'CLASSIFY the customer support ticket. Allowed categories: billing, delivery, '
    'technical, fraud. Reply with JSON only in the form '
    '{"category": "...", "reason": "..."}.'
)
FEW = (
    ZERO
    + "\nEXAMPLES:\n"
    + 'Text: Unauthorized transaction on my account. '
    '-> {"category": "fraud", "reason": "unauthorized transaction"}\n'
    + 'Text: My order arrived late. '
    '-> {"category": "delivery", "reason": "late delivery"}\n'
    + 'Text: The app crashes when I log in. '
    '-> {"category": "technical", "reason": "app crash"}'
)


def redact(text):
    """Redact PII using the specified patterns in their required order."""
    counts = {}
    for name, pattern, replacement in PATTERNS:
        text, count = pattern.subn(replacement, text)
        counts[name] = count
    return text, counts


def retry_chat(messages, ticket_id):
    """Retry transient mock failures at most three times with jittered backoff."""
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages), retry_number
        except mock_llm.TransientLLMError:
            if retry_number == MAX_RETRIES:
                raise
            wait = BASE_DELAY * (2**retry_number) * random.uniform(0.9, 1.15)
            print(
                f"retry ticket={ticket_id} retry={retry_number} "
                f"wait={wait:.4f}s"
            )
            time.sleep(wait)


def _validate_response(content):
    """Return the accepted category and status without guessing."""
    try:
        result = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(content).lower())
        matches = [word for word in words if word in ALLOWED_CATEGORIES]
        if len(matches) == 1:
            return matches[0], "classified_keyword"
        return "", "flagged_invalid_output"

    if isinstance(result, dict):
        category = result.get("category")
        if isinstance(category, str) and category in ALLOWED_CATEGORIES:
            return category, "classified_json"
    return "", "flagged_invalid_output"


def _is_premium(value):
    return value is True or str(value).strip().lower() in {"true", "1"}


def run_pipeline(prompt=FEW, token_budget=100000, output_name="triage_results.csv"):
    """Run one ordered ticket pass and optionally save its routing output.

    Pass ``token_budget=None`` for an unbudgeted comparison run and
    ``output_name=None`` to avoid writing an intermediate run to disk.
    """
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
    segment_path = OUT / "customer_segments.csv"
    if segment_path.exists():
        segments = pd.read_csv(segment_path, keep_default_na=False)
    else:
        segments = pd.DataFrame(
            columns=["customer_id", "segment_name", "is_premium"]
        )
    segment_map = (
        segments.set_index("customer_id").to_dict("index")
        if not segments.empty
        else {}
    )

    mock_llm.reset()
    random.seed(42)
    total_tokens = 0
    retry_count = 0
    model_call_tickets = []
    redaction_totals = {name: 0 for name, _, _ in PATTERNS}
    remaining_pii = {name: 0 for name, _, _ in PATTERNS}
    rows = []

    for ticket in tickets.itertuples(index=False):
        text = str(ticket.text)
        safe_text, counts = redact(text)
        for name, count in counts.items():
            redaction_totals[name] += count
        for name, pattern, _ in PATTERNS:
            remaining_pii[name] += len(pattern.findall(safe_text))
        assert all(pattern.search(safe_text) is None for _, pattern, _ in PATTERNS)

        category = ""
        status = ""
        tokens_used = 0

        if not text.strip():
            status = "skipped_empty"
        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            status = "budget_exhausted"
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": safe_text},
            ]
            assert messages[1]["content"] == safe_text
            assert all(
                pattern.search(messages[1]["content"]) is None
                for _, pattern, _ in PATTERNS
            )
            assert all(
                pattern.search(message["content"]) is None
                for message in messages
                for _, pattern, _ in PATTERNS
            )
            model_call_tickets.append(ticket.ticket_id)
            response, retries = retry_chat(messages, ticket.ticket_id)
            retry_count += retries
            tokens_used = int(response.usage.total_tokens)
            total_tokens += tokens_used
            content = response.choices[0].message.content or ""
            category, status = _validate_response(content)

        segment = segment_map.get(ticket.customer_id, {})
        segment_name = segment.get("segment_name", "")
        if status in HUMAN_REVIEW_STATUSES:
            route, priority = "human_review", "NORMAL"
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif _is_premium(segment.get("is_premium", False)):
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"

        rows.append(
            {
                "ticket_id": ticket.ticket_id,
                "customer_id": ticket.customer_id,
                "status": status,
                "category": category,
                "segment_name": segment_name,
                "priority": priority,
                "route": route,
                "tokens_used": tokens_used,
            }
        )

    result = pd.DataFrame(
        rows,
        columns=[
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ],
    )
    if output_name is not None:
        OUT.mkdir(exist_ok=True)
        result.to_csv(OUT / output_name, index=False)

    classified = result[result.status.isin(CLASSIFIED_STATUSES)]
    accuracy = (
        float((classified.category == tickets.set_index("ticket_id").loc[
            classified.ticket_id, "true_category"
        ].to_numpy()).mean())
        if not classified.empty
        else None
    )
    confusion = pd.crosstab(
        tickets.set_index("ticket_id").loc[classified.ticket_id, "true_category"],
        classified.set_index("ticket_id").category,
        rownames=["true_category"],
        colnames=["predicted_category"],
    ).reindex(
        index=sorted(ALLOWED_CATEGORIES),
        columns=sorted(ALLOWED_CATEGORIES),
        fill_value=0,
    )
    redaction_totals["total"] = sum(redaction_totals.values())
    remaining_pii["total"] = sum(remaining_pii.values())
    status_counts = (
        result.status.value_counts()
        .reindex(
            [
                "classified_json",
                "classified_keyword",
                "flagged_invalid_output",
                "skipped_empty",
                "flagged_long",
                "budget_exhausted",
            ],
            fill_value=0,
        )
        .to_dict()
    )
    guardrail_tickets = set(
        result.loc[
            result.status.isin({"skipped_empty", "flagged_long"}),
            "ticket_id",
        ]
    )
    assert guardrail_tickets.isdisjoint(model_call_tickets)
    classified_count = len(classified)
    return result, {
        "tokens": total_tokens,
        "total_tokens": total_tokens,
        "tokens_per_classified_ticket": (
            total_tokens / classified_count if classified_count else 0
        ),
        "classified_tickets": classified_count,
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "redactions": redaction_totals,
        "remaining_pii": remaining_pii,
        "status_counts": status_counts,
        "retry_count": retry_count,
        "model_call_tickets": len(model_call_tickets),
        "guardrail_model_calls": len(guardrail_tickets.intersection(model_call_tickets)),
        "token_budget": token_budget,
    }
