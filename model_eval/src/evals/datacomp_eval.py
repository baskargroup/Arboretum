import json
import os
import time
import pickle
import yaml
import warnings
import torch

from datetime import datetime

from pathlib import Path

from .eval_utils.main import evaluate_model

def datacomp_eval(args):
    print("Running datacomp eval")
    warnings.filterwarnings("ignore", message="Length of IterableDataset")
    torch.multiprocessing.set_sharing_strategy('file_system')
    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    args.train_output_dir = args.output_dir = Path(os.path.join(args.logs, args.name))
    results_filename = args.output_dir / "eval_results.jsonl"

    # TODO: This resume will never actually trigger
    results = {}
    if args.output_dir.exists():
        # Read existing results
        if results_filename.exists():
            with open(results_filename, "r") as f:
                lines = [json.loads(s) for s in f.readlines()]
                for line in lines:
                    if line["key"] not in tasks:
                        continue
                    results[line["dataset"]] = line
            print(f"Found {len(results)} eval result(s) in {results_filename}.")

    if args.resume:
        train_checkpoint = Path(args.resume)
    elif args.pretrained:
        train_checkpoint = str(args.pretrained)
    starttime = int(time.time())

    # Get list of datasets
    with open(os.path.join(os.path.dirname(__file__), "tasklist.yml")) as f:
        tasks = yaml.safe_load(f)

    for task_key in tasks:
        task_name = tasks[task_key].get("name", task_key)
        if task_name in results:
            print(
                f"Skipping {task_name} since results are already in {results_filename}"
            )
        else:
            print(f"Evaluating on {task_name}")
            metrics = evaluate_model(
                task_key,
                args.model,
                train_checkpoint,
                args.dc_eval_data_dir,
                tasks[task_key].get("size"),
                batch_size=args.batch_size,
            )
            metrics["main_metric"] = metrics.get(
                tasks[task_key].get("main_metric", "acc1")
            )
            results[task_name] = {
                "key": task_key,
                "dataset": task_name,
                "metrics": metrics,
            }
            with open(results_filename, "a+") as f:
                f.write(json.dumps(results[task_name]) + "\n")

        if results[task_name]["metrics"]["main_metric"] is not None:
            print(f"Score: {results[task_name]['metrics']['main_metric']:.4f}")
            rc = tasks[task_key].get('random_score', 0.0)
            print(f"Random chance: {rc:.4f}")
        else:
            print(f"Score: No summary metric")

    elapsed = int(time.time()) - starttime
    print(
        f"Evaluation time: {elapsed // 3600} hour(s) {elapsed % 3600 // 60} minute(s) {elapsed % 60} second(s)"
    )
    print()
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