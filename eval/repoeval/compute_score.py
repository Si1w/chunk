"""Compute evaluation scores for RepoEval benchmark.

Calculates Exact Match (EM) and Edit Similarity (ES) metrics over
code completion results, and produces summary CSV reports.
"""

import argparse
import csv
import os

import editdistance
import yaml

from .utils import FilePathBuilder, Tools, CONSTANTS


def compute_EM(target, predictions, passk):
    """Compute Exact Match score over predictions.

    Args:
        target: Ground truth code string.
        predictions: List of prediction strings.
        passk: Number of predictions to consider (pass@k).

    Returns:
        1 if any prediction exactly matches the target, 0 otherwise.
    """
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    for prediction in predictions[:passk]:
        pred_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
        if len(target_lines) == len(pred_lines) and target_lines == pred_lines:
            return 1
    return 0


def compute_ES(target, predictions, passk):
    """Compute Edit Similarity score over predictions.

    Args:
        target: Ground truth code string.
        predictions: List of prediction strings.
        passk: Number of predictions to consider (pass@k).

    Returns:
        Maximum edit similarity across the top-k predictions.
    """
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_str = "\n".join(target_lines)
    es_scores = []
    for prediction in predictions[:passk]:
        pred_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
        pred_str = "\n".join(pred_lines)
        es_scores.append(
            1 - (editdistance.eval(target_str, pred_str) / max(len(target_str), len(pred_str), 1))
        )
    return max(es_scores) if es_scores else 0


def compute_score_by_repo_with_metadata(lines, stype, passk=1):
    """Compute average EM or ES score across all samples.

    Args:
        lines: List of completion result dicts with 'completion' and 'ground_truth'.
        stype: Score type ('EM' or 'ES').
        passk: Number of predictions to consider.

    Returns:
        Average score rounded to 4 decimal places.
    """
    scores = []
    for line in lines:
        predictions = line["completion"]
        if isinstance(predictions, str):
            predictions = [predictions]

        if stype == "EM":
            score = compute_EM(line["ground_truth"], predictions, passk)
        elif stype == "ES":
            score = compute_ES(line["ground_truth"], predictions, passk)
        else:
            raise ValueError(f"Unknown score type: {stype}")
        scores.append(score)
    return round(sum(scores) / len(scores), 4) if scores else 0


def compute_token_cost(lines):
    """Compute average token cost across all samples.

    Args:
        lines: List of completion result dicts with 'token_cost'.

    Returns:
        Average token cost rounded to 4 decimal places.
    """
    costs = [line["token_cost"] for line in lines]
    return round(sum(costs) / len(costs), 4) if costs else 0


def scan_and_compute_scores(split, method, max_chunk_sizes, top_k, passk,
                            max_crossfile_context_list=None):
    """Scan completion directory and compute scores for all retriever-llm combinations.

    Args:
        split: Dataset split ('api' or 'line').
        method: Chunking method or 'all'.
        max_chunk_sizes: List of chunk sizes to evaluate.
        top_k: Number of retrieved windows.
        passk: pass@k for scoring.
        max_crossfile_context_list: List of cross-file context token budgets.

    Returns:
        List of result dictionaries.
    """
    if max_crossfile_context_list is None:
        max_crossfile_context_list = [2048]
    if not isinstance(max_chunk_sizes, list):
        max_chunk_sizes = [max_chunk_sizes]
    if not isinstance(max_crossfile_context_list, list):
        max_crossfile_context_list = [max_crossfile_context_list]

    combinations = Tools.scan_completion_directory()
    if not combinations:
        print("No combinations found in completion directory")
        return []

    print(f"Found {len(combinations)} retriever-llm combinations:")
    for retriever, llm in combinations:
        print(f"  - {retriever} + {llm}")

    all_results = []
    methods = CONSTANTS.ALL_METHODS if method == "all" else [method]

    for retriever, llm in combinations:
        if retriever == "none":
            path = FilePathBuilder.code_completion_result_path(
                "baseline", 0, "none", llm, split, 0, 0,
            )
            if not os.path.exists(path):
                print(f"Warning: File not found - {path}")
                continue
            try:
                lines = Tools.load_jsonl(path)
                total_time = None
                if lines and "total_inference_time" in lines[-1]:
                    total_time = lines[-1]["total_inference_time"]
                    lines = lines[:-1]

                all_results.append({
                    "retriever": retriever,
                    "llm": llm,
                    "method": "baseline",
                    "max_chunk_size": 0,
                    "max_crossfile_context": 0,
                    "top_k": 0,
                    "split": split,
                    "passk": passk,
                    "EM": compute_score_by_repo_with_metadata(lines, "EM", passk),
                    "ES": compute_score_by_repo_with_metadata(lines, "ES", passk),
                    "avg_token_cost": compute_token_cost(lines),
                    "total_inference_time": total_time,
                })
            except Exception as e:
                print(f"Error processing {path}: {e}")
            continue

        for m in methods:
            for chunk_size in max_chunk_sizes:
                for ctx_tokens in max_crossfile_context_list:
                    path = FilePathBuilder.code_completion_result_path(
                        m, chunk_size, retriever, llm, split, top_k, ctx_tokens,
                    )
                    if not os.path.exists(path):
                        print(f"Warning: File not found - {path}")
                        continue

                    try:
                        lines = Tools.load_jsonl(path)
                        total_time = None
                        if lines and "total_inference_time" in lines[-1]:
                            total_time = lines[-1]["total_inference_time"]
                            lines = lines[:-1]

                        all_results.append({
                            "retriever": retriever,
                            "llm": llm,
                            "method": m,
                            "max_chunk_size": chunk_size,
                            "max_crossfile_context": ctx_tokens,
                            "top_k": top_k,
                            "split": split,
                            "passk": passk,
                            "EM": compute_score_by_repo_with_metadata(lines, "EM", passk),
                            "ES": compute_score_by_repo_with_metadata(lines, "ES", passk),
                            "avg_token_cost": compute_token_cost(lines),
                            "total_inference_time": total_time,
                        })
                    except Exception as e:
                        print(f"Error processing {path}: {e}")

    return all_results


def main():
    """Entry point: load YAML config and compute evaluation scores."""
    parser = argparse.ArgumentParser(description="Compute scores for RepoEval benchmark.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    chunking = cfg["chunking"]
    retrieval_cfg = cfg["retrieval"]
    inference_cfg = cfg["inference"]
    eval_cfg = cfg["evaluation"]

    splits = ["api", "line"] if eval_cfg["split"] == "both" else [eval_cfg["split"]]

    for split in splits:
        print(f'\n{"#" * 80}')
        print(f"# Evaluating code completion results for {split.upper()} set")
        print(f'{"#" * 80}\n')

        all_results = scan_and_compute_scores(
            split=split,
            method=chunking["method"],
            max_chunk_sizes=chunking["max_chunk_sizes"],
            top_k=retrieval_cfg["top_k"],
            passk=eval_cfg["passk"],
            max_crossfile_context_list=inference_cfg["max_crossfile_contexts"],
        )

        if not all_results:
            print(f"\nNo results generated for {split} set")
            continue

        csv_path = FilePathBuilder.output_summary_result_path(split)
        fieldnames = [
            "retriever", "llm", "method", "max_chunk_size", "max_crossfile_context",
            "top_k", "split", "passk", "EM", "ES", "avg_token_cost", "total_inference_time",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f'\n{"=" * 80}')
        print(f"Summary results for {split} saved to: {csv_path}")
        print(f"Total combinations evaluated: {len(all_results)}")
        print(f'{"=" * 80}')


if __name__ == "__main__":
    main()
