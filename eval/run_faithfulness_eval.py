import json
import importlib.util
from pathlib import Path
import argparse

# ---------------------------
# Load app_streamlit.py directly
# ---------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "src" / "app_streamlit.py"

spec = importlib.util.spec_from_file_location("app_streamlit", APP_PATH)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

generate_answer = app.generate_answer


def normalize(name: str) -> str:
    """
    Normalize source names so that:
    - 'something.txt' and 'something' match
    - leading/trailing spaces are removed
    """
    name = name.strip()
    return Path(name).stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=str,
        default=str(PROJECT_ROOT / "eval" / "questions.jsonl"),
        help="Path to questions JSONL file",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["English", "Français"],
        default="English",
        help="Target answer language used by the model",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions)

    with questions_path.open("r", encoding="utf-8") as f:
        questions = [json.loads(x.strip()) for x in f if x.strip()]

    print(f"Running faithfulness + recall evaluation on {len(questions)} questions...")
    print(f"Language: {args.lang}")
    print(f"Questions file: {questions_path}\n")

    results = []

    for q in questions:
        print(f"Q{q['id']}: {q['question']}")

        answer, sources = generate_answer(q["question"], args.lang)

        print("Retrieved sources:", sources)
        print("Gold source:", q["gold_source"])

        # Normalize for comparison
        normalized_gold = normalize(q["gold_source"])
        normalized_sources = [normalize(s) for s in sources]

        # Faithfulness
        is_correct = int(normalized_gold in normalized_sources)

        # Recall@k
        ks = [1, 3, 5]
        recall = {k: 0 for k in ks}
        for k in ks:
            recall[k] = int(normalized_gold in normalized_sources[:k])

        print("Match (any position):", "✅" if is_correct else "❌")
        print("-" * 60)

        results.append(
            {
                "id": q["id"],
                "question": q["question"],
                "gold_source": q["gold_source"],
                "retrieved_sources": sources,
                "answer": answer,
                "correct": is_correct,
                "recall@1": recall[1],
                "recall@3": recall[3],
                "recall@5": recall[5],
            }
        )

    results_file = PROJECT_ROOT / "eval" / (
        "results_fr.jsonl" if args.lang == "Français" else "results_en.jsonl"
    )
    with results_file.open("w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r) + "\n")

    total = len(results)
    faithfulness = sum(r["correct"] for r in results) / total
    recall_1 = sum(r["recall@1"] for r in results) / total
    recall_3 = sum(r["recall@3"] for r in results) / total
    recall_5 = sum(r["recall@5"] for r in results) / total

    print("\n====== Retrieval Metrics ======")
    print(f"Faithfulness Accuracy: {faithfulness:.2%}")
    print(f"Recall@1: {recall_1:.2%}")
    print(f"Recall@3: {recall_3:.2%}")
    print(f"Recall@5: {recall_5:.2%}")
    print("===============================\n")
    print(f"Results written to: {results_file}")


if __name__ == "__main__":
    main()
