"""Part D: database-backed refund tools, bounded ReAct loop, and resumable runs."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import time
import uuid

import mock_llm
from src.policy_rag import PolicyRAG

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
RAG = PolicyRAG()


def get_order(order_id):
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return dict(
        zip(
            ("order_id", "status", "order_amount", "delivery_minutes"),
            row,
        )
    )


def search_policy(query):
    return {"results": RAG.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    refund = (
        float(order_amount)
        if delivery_minutes is not None and float(delivery_minutes) > 45
        else 0.0
    )
    return {"refund_inr": round(refund, 2)}


def _validated_order_refund(order_id, refund_inr):
    order = get_order(order_id)
    if "error" in order:
        return order
    if order["status"] != "completed":
        return {"error": "only completed orders can be refunded"}
    try:
        proposed_refund = float(refund_inr)
    except (TypeError, ValueError):
        return {"error": "refund amount must be a finite number"}
    if not math.isfinite(proposed_refund):
        return {"error": "refund amount must be a finite number"}
    expected_refund = calculate_refund(
        order["order_amount"], order["delivery_minutes"]
    )["refund_inr"]
    if expected_refund <= 0 or abs(proposed_refund - expected_refund) > 0.01:
        return {"error": "refund amount does not match the eligible order amount"}
    return {"order": order, "refund_inr": expected_refund}


def _append_ledger(tool_name, order_id, refund_inr):
    OUTPUT_DIR.mkdir(exist_ok=True)
    ledger_path = OUTPUT_DIR / "refund_ledger.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }
    with ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(record) + "\n")
    return {"refund_inr": record["refund_inr"]}


def issue_refund(order_id, refund_inr):
    validated = _validated_order_refund(order_id, refund_inr)
    if "error" in validated:
        return validated
    if validated["refund_inr"] > 1000:
        return {"error": "refunds above INR 1000 require human approval"}
    return _append_ledger("issue_refund", order_id, validated["refund_inr"])


def escalate_to_human(order_id, refund_inr):
    validated = _validated_order_refund(order_id, refund_inr)
    if "error" in validated:
        return validated
    if validated["refund_inr"] <= 1000:
        return {"error": "refunds of INR 1000 or less are not escalated"}
    return _append_ledger("escalate_to_human", order_id, validated["refund_inr"])


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order's status, amount, and delivery time by order ID.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve the three most relevant policy chunks for a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate a full refund only when delivery exceeded 45 minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": ["number", "null"]},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an eligible refund up to INR 1000 in the local ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record an eligible refund above INR 1000 for human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def _fallback(status, steps, llm_calls, tokens_used):
    return {
        "status": status,
        "final_answer": f"{status}: hand off to a human agent for review.",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
    }


def _checkpoint(run_id, scenario_id, messages, tokens_used, llm_calls, steps, tools):
    STATE_DIR.mkdir(exist_ok=True)
    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": len(steps),
        "tools_executed": tools,
        "steps": steps,
        "llm_calls": llm_calls,
    }
    path = STATE_DIR / f"{run_id}.json"
    path.write_text(json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8")


def _continue_agent(
    run_id,
    scenario_id,
    messages,
    tokens_used,
    llm_calls,
    steps,
    tools_executed,
    max_iterations,
    token_limit,
    time_limit_seconds,
    crash_after=None,
):
    started_at = time.monotonic()
    while llm_calls < max_iterations:
        if time.monotonic() - started_at >= time_limit_seconds:
            return _fallback(
                "FALLBACK_TIMEOUT", steps, llm_calls, tokens_used
            )

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += int(response.usage.total_tokens)
        if tokens_used > token_limit:
            _checkpoint(
                run_id, scenario_id, messages, tokens_used, llm_calls,
                steps, tools_executed
            )
            return _fallback(
                "FALLBACK_TOKEN_LIMIT", steps, llm_calls, tokens_used
            )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []
        if not tool_calls:
            answer = assistant_message.content or ""
            return {
                "status": answer.split(":", 1)[0].strip(),
                "final_answer": answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                    for tool_call in tool_calls
                ],
            }
        )
        for tool_call in tool_calls:
            name = tool_call.name
            arguments = tool_call.arguments
            if name not in TOOL_MAP:
                observation = {"error": f"unknown tool: {name}"}
            else:
                observation = TOOL_MAP[name](**arguments)
            tools_executed.append(name)
            steps.append(
                {
                    "tool": name,
                    "arguments": arguments,
                    "observation": observation,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": json.dumps(observation, ensure_ascii=False),
                }
            )
            _checkpoint(
                run_id, scenario_id, messages, tokens_used, llm_calls,
                steps, tools_executed
            )
            if crash_after is not None and len(tools_executed) == crash_after:
                raise RuntimeError("Simulated crash after checkpoint was saved")

    return _fallback(
        "FALLBACK_MAX_ITERATIONS", steps, llm_calls, tokens_used
    )


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after=None,
):
    run_id = run_id or str(uuid.uuid4())
    messages = [
        {
            "role": "system",
            "content": (
                "AGENT: Verify the order, follow the supplied refund policy, "
                "and use tools before deciding."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    return _continue_agent(
        run_id=run_id,
        scenario_id=scenario_id,
        messages=messages,
        tokens_used=0,
        llm_calls=0,
        steps=[],
        tools_executed=[],
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after=crash_after,
    )


def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
):
    checkpoint_path = STATE_DIR / f"{run_id}.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return _continue_agent(
        run_id=checkpoint["run_id"],
        scenario_id=checkpoint["scenario_id"],
        messages=checkpoint["messages"],
        tokens_used=checkpoint["tokens_used"],
        llm_calls=checkpoint["llm_calls"],
        steps=checkpoint["steps"],
        tools_executed=checkpoint["tools_executed"],
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
    )
