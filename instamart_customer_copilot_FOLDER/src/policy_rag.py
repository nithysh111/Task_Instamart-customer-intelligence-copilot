"""Part D retrieval: chunk policy files, rank them with TF-IDF, and ground answers."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "policies"


class PolicyRAG:
    def __init__(self, chunk_size=40, overlap=10):
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap in [0, chunk_size)")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []
        step = chunk_size - overlap
        for path in sorted(POLICY_DIR.glob("*.txt")):
            words = path.read_text(encoding="utf-8").split()
            for start in range(0, len(words), step):
                chunk_words = words[start : start + chunk_size]
                self.chunks.append(
                    {"source": path.name, "text": " ".join(chunk_words)}
                )
                if start + chunk_size >= len(words):
                    break

        if not self.chunks:
            raise FileNotFoundError(f"No policy text files found in {POLICY_DIR}")

        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def retrieve(self, query, k=3):
        if k < 0:
            raise ValueError("k must be non-negative")
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        indices = scores.argsort()[::-1][:k]
        return [
            {
                "source": self.chunks[index]["source"],
                "score": float(scores[index]),
                "text": self.chunks[index]["text"],
            }
            for index in indices
        ]

    def answer(self, question):
        hits = self.retrieve(question, k=3)
        context = "\n".join(
            f"[{hit['source']}]\n{hit['text']}" for hit in hits
        )
        response = mock_llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "POLICY ASSISTANT: Answer only from the supplied context. "
                        "If the information is not present, say you could not find "
                        "the information in the policy documents."
                    ),
                },
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}",
                },
            ]
        )
        return {
            "answer": response.choices[0].message.content,
            "sources": list(dict.fromkeys(hit["source"] for hit in hits)),
            "retrieved": hits,
        }


def evaluate(rag):
    evaluation_path = ROOT / "data" / "raw" / "policy_eval.csv"
    questions = pd.read_csv(evaluation_path, keep_default_na=False)
    rows = []
    for row in questions.itertuples(index=False):
        result = rag.answer(row.question)
        expected_doc = row.expected_doc
        rows.append(
            {
                "question": row.question,
                "expected_doc": expected_doc,
                "expected_fact": row.expected_fact,
                "retrieved_doc": (
                    expected_doc in result["sources"] if expected_doc != "NONE" else False
                ),
                "fact_found": (
                    row.expected_fact in result["answer"]
                    if row.expected_fact
                    else False
                ),
                "fallback": result["answer"].startswith("I could not find"),
                "answer": result["answer"],
                "sources": result["sources"],
            }
        )
    return pd.DataFrame(rows)


def retrieval_explanation():
    return (
        "Term frequency (TF) measures how often a term appears in a chunk, while "
        "inverse document frequency (IDF) gives less weight to terms common across "
        "many chunks and more weight to distinctive terms. TF-IDF is lexical, so "
        "different phrasings with the same meaning may have little or no overlap "
        "and retrieve the wrong passage."
    )
