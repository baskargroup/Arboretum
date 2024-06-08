from pathlib import Path
import json

results_filename = "/scratch/bf996/vlhub/logs/imagenet-captions-fang-ttd-ep1-128/eval_results.jsonl"

results = {}
with open(results_filename, "r") as f:
    lines = [json.loads(s) for s in f.readlines()]
    for line in lines:
        results[line["dataset"]] = line

print("=== Final results ===")
for line in results.values():
    print(f"{line['dataset']}: {line['metrics']['main_metric']}")
print("=== Average results ===")
avg_results = {}
for k, v in results.items():
    if v["metrics"]["main_metric"] is not None:
        avg_results[k] = v
avg_score = sum(
    [v["metrics"]["main_metric"] for v in avg_results.values()]
) / len(avg_results)
print(f"Average score: {avg_score:.4f}")