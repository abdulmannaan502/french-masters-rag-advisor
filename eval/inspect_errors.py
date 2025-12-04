import json
from pathlib import Path

RESULTS_FILE = Path("eval/results.jsonl")

def main():
    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]

    print(f"Total questions: {len(rows)}")
    print("Showing only incorrect cases:\n")

    for r in rows:
        if not r["correct"]:
            print(f"ID: {r['id']}")
            print(f"Q: {r['question']}")
            print(f"Gold: {r['gold_source']}")
            print(f"Retrieved: {r['retrieved_sources']}")
            print(f"R@1={r['recall@1']} R@3={r['recall@3']} R@5={r['recall@5']}")
            print("ANSWER:")
            print(r["answer"][:600], "...\n")
            print("-" * 80)

if __name__ == "__main__":
    main()
