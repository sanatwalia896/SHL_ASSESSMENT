from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.catalog import get_catalog_store
from app.query_planner import QueryPlanner
from app.retrieval import RetrievalEngine
from app.ranker import Ranker
from app.schemas import Message


EVAL_CASES = [
    {
        "name": "rust_engineer",
        "messages": [
            Message(
                role="user",
                content="I'm hiring a senior Rust engineer for high-performance networking infrastructure.",
            )
        ],
        "expected": [
            "Smart Interview Live Coding",
            "Linux Programming (General)",
            "Networking and Implementation (New)",
            "SHL Verify Interactive G+",
        ],
    },
    {
        "name": "admin_excel_word",
        "messages": [
            Message(
                role="user",
                content="I need a quick screen for admin assistants using Excel and Word daily.",
            )
        ],
        "expected_keywords": ["excel", "word"],
    },
    {
        "name": "vague_query_clarification",
        "messages": [
            Message(role="user", content="I need an assessment.")
        ],
        "expect_clarification": True,
    },
    {
        "name": "legal_refusal",
        "messages": [
            Message(
                role="user",
                content="Are we legally required to test all staff under HIPAA?",
            )
        ],
        "expect_refusal": True,
    },
]


def recall_at_k(predicted: list[str], expected: list[str], k: int = 10) -> float:
    if not expected:
        return 0.0

    predicted_set = set(predicted[:k])
    expected_set = set(expected)

    return len(predicted_set & expected_set) / len(expected_set)


def main() -> None:
    catalog = get_catalog_store()
    planner = QueryPlanner()
    retriever = RetrievalEngine(catalog=catalog)
    ranker = Ranker()

    print("\nSHL Evaluation Report")
    print("=" * 60)

    recall_scores = []

    for case in EVAL_CASES:
        print(f"\nCase: {case['name']}")

        plan = planner.plan(case["messages"])

        print(f"Intent: {plan.intent}")
        print(f"Direct keywords: {plan.direct_keywords}")
        print(f"Related keywords: {plan.related_keywords}")
        print(f"Semantic query: {plan.semantic_query}")

        if case.get("expect_clarification"):
            passed = plan.needs_clarification or plan.intent == "clarify"
            print(f"Clarification expected: {'PASS' if passed else 'FAIL'}")
            continue

        if case.get("expect_refusal"):
            passed = plan.intent == "refuse"
            print(f"Refusal expected: {'PASS' if passed else 'FAIL'}")
            continue

        candidates = retriever.retrieve(plan)
        ranked = ranker.rank(candidates, plan)

        names = [item.product.name for item in ranked]
        urls_valid = all(item.product.url.startswith("https://www.shl.com/") for item in ranked)

        print("Top recommendations:")
        for i, item in enumerate(ranked, start=1):
            print(f"{i}. {item.product.name} | score={item.score:.2f}")

        print(f"Catalog URL grounding: {'PASS' if urls_valid else 'FAIL'}")

        expected = case.get("expected", [])
        if expected:
            score = recall_at_k(names, expected, k=10)
            recall_scores.append(score)
            print(f"Recall@10: {score:.2f}")

        expected_keywords = case.get("expected_keywords", [])
        if expected_keywords:
            joined = " ".join(names).lower()
            passed = all(keyword in joined for keyword in expected_keywords)
            print(f"Keyword relevance check: {'PASS' if passed else 'FAIL'}")

    if recall_scores:
        mean_recall = sum(recall_scores) / len(recall_scores)
        print("\n" + "=" * 60)
        print(f"Mean Recall@10: {mean_recall:.2f}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()