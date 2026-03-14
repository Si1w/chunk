"""Ablation study: sliding window overlap on code completion performance.

Varies overlap_lines for SlidingChunkBuilder and runs the full evaluation
pipeline (chunk -> retrieve -> infer -> score). Uses method names like
'sliding_o{N}' to distinguish overlap variants in file paths.

Usage:
    uv run python -m eval.repoeval.ablation_overlap --config configs/ablation_overlap.yaml --steps chunk
    uv run python -m eval.repoeval.ablation_overlap --config configs/ablation_overlap.yaml --steps retrieve --embed_model bm25
    uv run python -m eval.repoeval.ablation_overlap --config configs/ablation_overlap.yaml --steps infer --llm Qwen/Qwen2.5-Coder-7B
    uv run python -m eval.repoeval.ablation_overlap --config configs/ablation_overlap.yaml --steps score
"""

import argparse
import csv
import os
from collections import defaultdict

import yaml

from chunk import SlidingChunkBuilder
from .make_window import make_query_window
from .utils import CONSTANTS, FilePathBuilder, Tools


def overlap_method(n):
    """Encode overlap_lines value into a method name for file paths."""
    return f"sliding_o{n}"


def step_chunk(cfg):
    """Build sliding-window repo chunks for each overlap value."""
    chunking = cfg["chunking"]
    query = cfg["query"]
    repos = CONSTANTS.REPOs
    overlap_values = cfg["ablation"]["overlap_lines"]

    for overlap in overlap_values:
        method = overlap_method(overlap)
        for max_chunk_size in chunking["max_chunk_sizes"]:
            print(f"\n--- chunk: overlap={overlap}, size={max_chunk_size} ---")
            builder = SlidingChunkBuilder(
                max_chunk_size=max_chunk_size,
                overlap_lines=overlap,
                metadata_template=chunking.get("metadata_template", "repoeval"),
            )
            for repo in repos:
                files = Tools.iterate_repository(repo, chunking["language"])
                all_chunks = []
                for fpath_tuple, code in files.items():
                    all_chunks.extend(builder.chunkify(
                        code,
                        repo_level_metadata={
                            "repo": repo,
                            "fpath_tuple": "/".join(fpath_tuple),
                        },
                    ))
                merged = defaultdict(list)
                for c in all_chunks:
                    merged[c["content"]].append(c["metadata"])
                windows = [{"context": ctx, "metadata": metas} for ctx, metas in merged.items()]
                print(f"  {repo}: {len(windows)} windows")
                Tools.dump_jsonl(windows, FilePathBuilder.repo_windows_path(repo, method, max_chunk_size))

    # Ensure query windows exist (shared across all overlap values)
    query_path = FilePathBuilder.query_windows_path(
        "api", query["context_length"], query["prompt_type"], query["window_size"],
    )
    if not os.path.exists(query_path):
        print("\n--- Building query windows ---")
        make_query_window(
            context_length=query["context_length"],
            prompt_type=query["prompt_type"],
            repos=repos,
            window_size=query["window_size"],
            language=chunking["language"],
        )


def step_retrieve(cfg, embed_model_filter=None):
    """Run retrieval for each overlap variant."""
    from .retrieval import Retriever

    chunking = cfg["chunking"]
    query = cfg["query"]
    retrieval_cfg = cfg["retrieval"]
    overlap_values = cfg["ablation"]["overlap_lines"]
    methods = [overlap_method(n) for n in overlap_values]

    eval_split = cfg.get("evaluation", {}).get("split", "both")
    splits = ["api", "line"] if eval_split == "both" else [eval_split]
    embed_models = [embed_model_filter] if embed_model_filter else retrieval_cfg["embed_models"]

    for embed_model in embed_models:
        if embed_model == "none":
            continue
        retriever = Retriever(
            CONSTANTS.REPOs,
            embed_model=embed_model,
            batch_size=retrieval_cfg.get("batch_size", 32),
        )
        for split in splits:
            for max_chunk_size in chunking["max_chunk_sizes"]:
                for method in methods:
                    retriever.retrieval(
                        method=method,
                        max_chunk_size=max_chunk_size,
                        split=split,
                        context_length=query["context_length"],
                        prompt_type=query["prompt_type"],
                        top_k=retrieval_cfg["top_k"],
                    )


def step_infer(cfg, llm_filter=None):
    """Run code completion for each overlap variant."""
    from .code_completion import CodeCompletionInference

    chunking = cfg["chunking"]
    retrieval_cfg = cfg["retrieval"]
    inference_cfg = cfg["inference"]
    overlap_values = cfg["ablation"]["overlap_lines"]
    methods = [overlap_method(n) for n in overlap_values]

    eval_split = cfg.get("evaluation", {}).get("split", "both")
    splits = ["api", "line"] if eval_split == "both" else [eval_split]
    llms = [llm_filter] if llm_filter else inference_cfg["llms"]

    for llm in llms:
        engine = CodeCompletionInference(
            llm=llm,
            max_generate_tokens=inference_cfg["max_generate_tokens"],
            max_seq_length=inference_cfg["max_seq_length"],
        )
        for embed_model in retrieval_cfg["embed_models"]:
            if embed_model == "none":
                continue
            for split in splits:
                for max_crossfile_context in inference_cfg["max_crossfile_contexts"]:
                    for max_chunk_size in chunking["max_chunk_sizes"]:
                        for method in methods:
                            out_path = FilePathBuilder.code_completion_result_path(
                                method, max_chunk_size, embed_model, llm, split,
                                retrieval_cfg["top_k"], max_crossfile_context,
                            )
                            if os.path.exists(out_path):
                                print(f"[Skip] {method} | {split} | {max_chunk_size} (exists)")
                                continue
                            engine.run_inference(
                                method, max_chunk_size, embed_model, split,
                                retrieval_cfg["top_k"], max_crossfile_context,
                            )


def step_score(cfg):
    """Compute scores and write ablation summary CSV."""
    from .compute_score import compute_score_by_repo_with_metadata, compute_token_cost

    chunking = cfg["chunking"]
    retrieval_cfg = cfg["retrieval"]
    inference_cfg = cfg["inference"]
    eval_cfg = cfg["evaluation"]
    overlap_values = cfg["ablation"]["overlap_lines"]
    methods = [overlap_method(n) for n in overlap_values]
    passk = eval_cfg.get("passk", 1)

    eval_split = eval_cfg.get("split", "both")
    splits = ["api", "line"] if eval_split == "both" else [eval_split]

    combinations = Tools.scan_completion_directory()
    if not combinations:
        print("No completion results found")
        return

    all_results = []
    for retriever_name, llm in combinations:
        if retriever_name == "none":
            continue
        for split in splits:
            for method in methods:
                overlap = int(method.rsplit("_o", 1)[1])
                for chunk_size in chunking["max_chunk_sizes"]:
                    for ctx_tokens in inference_cfg["max_crossfile_contexts"]:
                        path = FilePathBuilder.code_completion_result_path(
                            method, chunk_size, retriever_name, llm, split,
                            retrieval_cfg["top_k"], ctx_tokens,
                        )
                        if not os.path.exists(path):
                            continue
                        lines = Tools.load_jsonl(path)
                        total_time = None
                        if lines and "total_inference_time" in lines[-1]:
                            total_time = lines[-1]["total_inference_time"]
                            lines = lines[:-1]

                        all_results.append({
                            "overlap_lines": overlap,
                            "retriever": retriever_name,
                            "llm": llm,
                            "max_chunk_size": chunk_size,
                            "max_crossfile_context": ctx_tokens,
                            "split": split,
                            "passk": passk,
                            "EM": compute_score_by_repo_with_metadata(lines, "EM", passk),
                            "ES": compute_score_by_repo_with_metadata(lines, "ES", passk),
                            "avg_token_cost": compute_token_cost(lines),
                            "total_inference_time": total_time,
                        })

    if not all_results:
        print("No results to report")
        return

    csv_path = os.path.join(
        os.path.dirname(FilePathBuilder.output_summary_result_path("api")),
        "ablation_overlap.csv",
    )
    FilePathBuilder._ensure_dir(csv_path)

    fieldnames = [
        "overlap_lines", "retriever", "llm", "max_chunk_size",
        "max_crossfile_context", "split", "passk", "EM", "ES",
        "avg_token_cost", "total_inference_time",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nAblation results saved to: {csv_path}")
    print(f"Total entries: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser(description="Ablation: sliding window overlap.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--steps", nargs="+", required=True,
                        choices=["chunk", "retrieve", "infer", "score"])
    parser.add_argument("--embed_model", type=str, default=None,
                        help="Run retrieve for a specific embed model only.")
    parser.add_argument("--llm", type=str, default=None,
                        help="Run infer for a specific LLM only.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    for step in args.steps:
        print(f"\n{'=' * 60}")
        print(f"Step: {step}")
        print(f"{'=' * 60}")
        if step == "chunk":
            step_chunk(cfg)
        elif step == "retrieve":
            step_retrieve(cfg, embed_model_filter=args.embed_model)
        elif step == "infer":
            step_infer(cfg, llm_filter=args.llm)
        elif step == "score":
            step_score(cfg)


if __name__ == "__main__":
    main()
