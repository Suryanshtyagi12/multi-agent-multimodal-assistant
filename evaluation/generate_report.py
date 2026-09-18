import json
import os

RESULTS_PATH = "evaluation/results.json"

def generate_report():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    
    metrics = data["metrics"]
    results = data["results"]
    
    print("=" * 60)
    print("SCHOLARRAG EVALUATION REPORT")
    print(f"Date: {data['evaluation_date']}")
    print(f"Model: {data['model']}")
    print("=" * 60)
    
    print("\nOVERALL METRICS:")
    print(f"  Answer Relevance:   {metrics['answer_relevance']:.3f} / 1.0")
    print(f"  Faithfulness:       {metrics['faithfulness']:.3f} / 1.0")
    print(f"  Context Precision:  {metrics['context_precision']:.3f} / 1.0")
    print(f"  Context Recall:     {metrics['context_recall']:.3f} / 1.0")
    if metrics.get("guardrail_accuracy"):
        print(f"  Guardrail Accuracy: {metrics['guardrail_accuracy']:.3f} / 1.0")
    
    print("\nBREAKDOWN BY DIFFICULTY:")
    for difficulty in ["easy", "medium", "hard"]:
        subset = [r for r in results
                  if r["difficulty"] == difficulty
                  and not r.get("skipped_metrics")]
        if subset:
            avg_rel = sum(r["answer_relevance"] for r in subset) / len(subset)
            avg_faith = sum(r["faithfulness"] for r in subset) / len(subset)
            print(f"  {difficulty.upper()} ({len(subset)} questions): "
                  f"relevance={avg_rel:.2f} faithfulness={avg_faith:.2f}")
    
    print("\nBREAKDOWN BY CONTEXT TYPE:")
    types = set(r["context_type"] for r in results if not r.get("skipped_metrics"))
    for ct in sorted(types):
        subset = [r for r in results
                  if r["context_type"] == ct
                  and not r.get("skipped_metrics")]
        if subset:
            avg_rel = sum(r["answer_relevance"] for r in subset) / len(subset)
            print(f"  {ct} ({len(subset)} questions): relevance={avg_rel:.2f}")
    
    print("\nFAILED QUESTIONS (relevance < 0.5):")
    failed = [r for r in results
              if r.get("answer_relevance") is not None
              and r["answer_relevance"] < 0.5]
    if failed:
        for r in failed:
            print(f"  [{r['id']}] {r['question'][:60]}")
            print(f"    Relevance: {r['answer_relevance']} | "
                  f"Route: {r['route']}")
    else:
        print("  None — all questions scored above 0.5")
    
    print("\nGUARDRAIL TEST RESULTS:")
    guardrail_results = [r for r in results
                         if r["context_type"] == "guardrail"
                         or r["test_type"] == "out_of_scope"]
    for r in guardrail_results:
        status = "PASS" if r.get("guardrail_correct") else "FAIL"
        print(f"  [{status}] {r['id']} — {r['test_type']}: {r.get('guardrail_note', '')}")
    
    print("=" * 60)

if __name__ == "__main__":
    generate_report()
