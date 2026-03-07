"""Compute evaluation scores for CrossCodeEval benchmark.

Calculates 6 metrics aligned with official cceval:
- EM (Exact Match)
- ES (Edit Similarity via fuzz.ratio, 0-100)
- ID EM, ID Precision, ID Recall, ID F1 (identifier-level metrics)
"""

import argparse
import csv
import os

import yaml

from .utils import FilePathBuilder, Tools, CONSTANTS
from .eval_utils import (
    cal_edit_sim,
    postprocess_code_lines,
    extract_identifiers,
    compute_id_match,
)


def compute_EM(target, prediction):
    """Exact match: strip lines, remove empty, compare lists."""
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    pred_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    pred_lines = pred_lines[:len(target_lines)]
    if len(target_lines) == len(pred_lines) and target_lines == pred_lines:
        return 1
    return 0


def compute_ES(target, prediction):
    """Edit similarity using fuzz.ratio (0-100 scale)."""
    return cal_edit_sim(prediction, target)


def compute_id_metrics(target, prediction, lang="python"):
    """Compute identifier-level metrics: EM, Precision, Recall, F1."""
    gt_ids = extract_identifiers(target, lang)
    pred_ids = extract_identifiers(prediction, lang)

    # ID Exact Match
    id_em = 1 if gt_ids == pred_ids else 0

    # Precision, Recall, F1
    tp, fp, fn = compute_id_match(pred_ids, gt_ids)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return id_em, precision, recall, f1


def scan_and_compute_scores(method, max_chunk_sizes, top_k, passk,
                            max_crossfile_context_list=None, lang="python"):
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

    def _score_file(path):
        """Load a completion file and compute all metrics."""
        lines = Tools.load_jsonl(path)
        total_time = None
        if lines and "total_inference_time" in lines[-1]:
            total_time = lines[-1]["total_inference_time"]
            lines = lines[:-1]

        em_scores = []
        es_scores = []
        id_em_scores = []
        id_prec_scores = []
        id_rec_scores = []
        id_f1_scores = []
        token_costs = []

        for line in lines:
            gt = line["ground_truth"]
            completion = line["completion"]
            if isinstance(completion, list):
                completion = completion[0]
            prompt = line.get("prompt", "")

            processed = postprocess_code_lines(prompt, completion, lang)

            em_scores.append(compute_EM(gt, processed))
            es_scores.append(compute_ES(gt, processed))

            id_em, id_prec, id_rec, id_f1 = compute_id_metrics(gt, processed, lang)
            id_em_scores.append(id_em)
            id_prec_scores.append(id_prec)
            id_rec_scores.append(id_rec)
            id_f1_scores.append(id_f1)

            if "token_cost" in line:
                token_costs.append(line["token_cost"])

        n = len(lines)
        avg = lambda scores: round(sum(scores) / n, 4) if n else 0

        return {
            "EM": avg(em_scores),
            "ES": avg(es_scores),
            "ID_EM": avg(id_em_scores),
            "ID_Precision": avg(id_prec_scores),
            "ID_Recall": avg(id_rec_scores),
            "ID_F1": avg(id_f1_scores),
            "avg_token_cost": round(sum(token_costs) / len(token_costs), 4) if token_costs else 0,
            "total_inference_time": total_time,
        }

    for retriever, llm in combinations:
        if retriever == "none":
            path = FilePathBuilder.code_completion_result_path(
                "baseline", 0, "none", llm, 0, 0,
            )
            if not os.path.exists(path):
                print(f"Warning: File not found - {path}")
                continue
            try:
                scores = _score_file(path)
                all_results.append({
                    "retriever": retriever,
                    "llm": llm,
                    "method": "baseline",
                    "max_chunk_size": 0,
                    "max_crossfile_context": 0,
                    "top_k": 0,
                    "passk": passk,
                    **scores,
                })
            except Exception as e:
                print(f"Error processing {path}: {e}")
            continue

        for m in methods:
            for chunk_size in max_chunk_sizes:
                for ctx_tokens in max_crossfile_context_list:
                    path = FilePathBuilder.code_completion_result_path(
                        m, chunk_size, retriever, llm, top_k, ctx_tokens,
                    )
                    if not os.path.exists(path):
                        print(f"Warning: File not found - {path}")
                        continue

                    try:
                        scores = _score_file(path)
                        all_results.append({
                            "retriever": retriever,
                            "llm": llm,
                            "method": m,
                            "max_chunk_size": chunk_size,
                            "max_crossfile_context": ctx_tokens,
                            "top_k": top_k,
                            "passk": passk,
                            **scores,
                        })
                    except Exception as e:
                        print(f"Error processing {path}: {e}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compute scores for CrossCodeEval benchmark.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    chunking = cfg["chunking"]
    retrieval_cfg = cfg["retrieval"]
    inference_cfg = cfg["inference"]
    eval_cfg = cfg["evaluation"]

    print(f'\n{"#" * 80}')
    print("# Evaluating CrossCodeEval code completion results")
    print(f'{"#" * 80}\n')

    all_results = scan_and_compute_scores(
        method=chunking["method"],
        max_chunk_sizes=chunking["max_chunk_sizes"],
        top_k=retrieval_cfg["top_k"],
        passk=eval_cfg["passk"],
        max_crossfile_context_list=inference_cfg["max_crossfile_contexts"],
        lang=chunking["language"],
    )

    if not all_results:
        print("\nNo results generated")
        return

    csv_path = FilePathBuilder.output_summary_result_path()
    fieldnames = [
        "retriever", "llm", "method", "max_chunk_size", "max_crossfile_context",
        "top_k", "passk", "EM", "ES", "ID_EM", "ID_Precision", "ID_Recall", "ID_F1",
        "avg_token_cost", "total_inference_time",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f'\n{"=" * 80}')
    print(f"Summary results saved to: {csv_path}")
    print(f"Total combinations evaluated: {len(all_results)}")
    print(f'{"=" * 80}')


if __name__ == "__main__":
    main()
