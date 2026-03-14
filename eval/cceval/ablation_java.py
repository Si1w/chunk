"""Cross-language validation: run cceval pipeline on Java dataset.

Tests whether chunking strategies generalize across programming languages.
Handles Java-specific benchmark and query window paths that the existing
cceval pipeline hardcodes to Python.

Usage:
    uv run python -m eval.cceval.ablation_java --config configs/cceval_ablation_java.yaml --steps fetch
    uv run python -m eval.cceval.ablation_java --config configs/cceval_ablation_java.yaml --steps chunk
    uv run python -m eval.cceval.ablation_java --config configs/cceval_ablation_java.yaml --steps retrieve --embed_model bm25
    uv run python -m eval.cceval.ablation_java --config configs/cceval_ablation_java.yaml --steps infer --llm deepseek-ai/deepseek-coder-6.7b-base
    uv run python -m eval.cceval.ablation_java --config configs/cceval_ablation_java.yaml --steps score
"""

import argparse
import csv
import os

import yaml
from tqdm import tqdm

from .make_window import make_repo_window
from .utils import CONSTANTS, FilePathBuilder, Tools

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _benchmark_path(lang):
    """Language-aware benchmark path (cceval stores data per language)."""
    return os.path.join(_BASE_DIR, "datasets", lang, "line_completion_curated.jsonl")


def _query_windows_path(lang, window_size):
    """Language-aware query windows path to avoid overwriting Python data."""
    out = os.path.join(_BASE_DIR, "query", "ablation_java", f"line_completion_{lang}_{window_size}.jsonl")
    FilePathBuilder._ensure_dir(out)
    return out


def step_fetch(cfg):
    """Download and curate the Java dataset, clone required repositories."""
    from .fetch_dataset import download_data, clone_and_curate

    lang = cfg["chunking"]["language"]
    print(f"\n--- Fetching {lang} dataset ---")
    download_data(directory=_BASE_DIR, lang=lang)
    clone_and_curate(directory=_BASE_DIR, lang=lang)


def step_chunk(cfg):
    """Build repo windows (all methods) and Java query windows."""
    chunking = cfg["chunking"]
    query_cfg = cfg["query"]
    lang = chunking["language"]
    repos = CONSTANTS.repos()
    methods = CONSTANTS.ALL_METHODS if chunking["method"] == "all" else [chunking["method"]]

    # Repo windows: reuse existing make_repo_window (already supports language)
    for max_chunk_size in chunking["max_chunk_sizes"]:
        configs = {
            "max_chunk_size": max_chunk_size,
            "language": lang,
            "metadata_template": chunking["metadata_template"],
            "overlap_lines": chunking["overlap_lines"],
            "chunk_expansion": chunking["chunk_expansion"],
            "private_function": chunking["private_function"],
            "function_overlap": chunking["function_overlap"],
        }
        for method in methods:
            print(f"\n--- chunk: method={method}, size={max_chunk_size} ---")
            make_repo_window(repos, method, **configs)

    # Query windows: custom handling for Java benchmark path
    window_size = query_cfg["window_size"]
    query_path = _query_windows_path(lang, window_size)
    if os.path.exists(query_path):
        print(f"\nQuery windows already exist: {query_path}")
        return

    print(f"\n--- Building {lang} query windows ---")
    benchmark = Tools.load_jsonl(_benchmark_path(lang))
    code_windows = []
    cur_repo = None
    source_code_files = None

    for item in tqdm(benchmark, desc=f"Building {lang} query windows"):
        meta = item["metadata"]
        repo = meta["repository"].replace("/", "_")
        if repos and repo not in repos:
            continue
        if repo != cur_repo:
            cur_repo = repo
            source_code_files = Tools.iterate_repository(repo, lang)

        fpath = meta["file"]
        fpath_tuple = None
        for ft in source_code_files:
            if "/".join(ft[1:]) == fpath or "/".join(ft) == fpath:
                fpath_tuple = ft
                break
        if fpath_tuple is None:
            continue

        code_lines = source_code_files[fpath_tuple].splitlines()
        gt_start = meta["groundtruth_start_lineno"]
        non_empty = [l for l in code_lines[:gt_start] if l.strip()]

        code_windows.append({
            "query": "\n".join(non_empty[-window_size:]),
            "metadata": {
                "fpath_tuple": "/".join(fpath_tuple),
                "task_id": meta["task_id"],
                "repo": cur_repo,
                "ground_truth": item["groundtruth"],
            },
            "prompt": item["prompt"],
        })

    print(f"build {len(code_windows)} query windows")
    Tools.dump_jsonl(code_windows, query_path)


def step_retrieve(cfg, embed_model_filter=None):
    """Run retrieval with next-chunk CFC using Java query windows."""
    from .retrieval import Retriever

    chunking = cfg["chunking"]
    lang = chunking["language"]
    retrieval_cfg = cfg["retrieval"]
    query_cfg = cfg["query"]
    methods = CONSTANTS.ALL_METHODS if chunking["method"] == "all" else [chunking["method"]]
    top_k = retrieval_cfg["top_k"]

    query_path = _query_windows_path(lang, query_cfg["window_size"])
    query_lines = Tools.load_jsonl(query_path)

    embed_models = [embed_model_filter] if embed_model_filter else retrieval_cfg["embed_models"]

    for embed_model in embed_models:
        if embed_model == "none":
            continue
        retriever = Retriever(
            CONSTANTS.repos(),
            embed_model=embed_model,
            batch_size=retrieval_cfg.get("batch_size", 32),
        )
        for max_chunk_size in chunking["max_chunk_sizes"]:
            for method in methods:
                out_path = FilePathBuilder.inference_corpus_path(
                    method, max_chunk_size, embed_model, top_k,
                )
                if os.path.exists(out_path):
                    print(f"[Skip] {method} | {max_chunk_size} (exists)")
                    continue

                inference_corpus = []
                cur_repo = None
                index = None
                windows = None
                id2idx = None

                label = "BM25" if retriever.is_bm25 else embed_model
                for ql in tqdm(query_lines, desc=f"{label} {method} {max_chunk_size}"):
                    repo = ql["metadata"]["repo"]
                    if repo != cur_repo:
                        index = retriever._ensure_index(repo, method, max_chunk_size)
                        windows = Tools.load_jsonl(
                            FilePathBuilder.repo_windows_path(repo, method, max_chunk_size)
                        )
                        id2idx = retriever._build_id2idx(windows)
                        cur_repo = repo

                    query_fpath = ql["metadata"]["fpath_tuple"]
                    indices, scores = retriever._search(
                        index, ql["query"], top_k + 50, len(windows),
                    )

                    retrieved = []
                    seen_content = set()
                    for idx, score in zip(indices, scores):
                        if int(idx) < 0 or int(idx) >= len(windows):
                            continue
                        w = windows[int(idx)]
                        w_fpath = w["metadata"]["fpath_tuple"]

                        if w_fpath == query_fpath:
                            continue

                        # Next-chunk CFC strategy
                        next_key = f"{w_fpath}|{w['metadata']['chunk_id'] + 1}"
                        if next_key in id2idx:
                            cfc_window = windows[id2idx[next_key]]
                            content = cfc_window["context"]
                            metadata = cfc_window["metadata"]
                        else:
                            content = w["context"]
                            metadata = w["metadata"]

                        if content in seen_content:
                            continue
                        seen_content.add(content)

                        retrieved.append({
                            "content": content,
                            "metadata": metadata,
                            "score": float(score),
                        })
                        if len(retrieved) >= top_k:
                            break

                    inference_corpus.append({
                        "prompt": ql["prompt"],
                        "retrieved_windows": retrieved[:top_k],
                        "ground_truth": ql["metadata"]["ground_truth"],
                    })

                Tools.dump_jsonl(inference_corpus, out_path)


def step_infer(cfg, llm_filter=None):
    """Run code completion (reuses existing CodeCompletionInference)."""
    from .code_completion import CodeCompletionInference

    chunking = cfg["chunking"]
    retrieval_cfg = cfg["retrieval"]
    inference_cfg = cfg["inference"]
    methods = CONSTANTS.ALL_METHODS if chunking["method"] == "all" else [chunking["method"]]

    llms = [llm_filter] if llm_filter else inference_cfg["llms"]

    for llm in llms:
        engine = CodeCompletionInference(
            llm=llm,
            max_generate_tokens=inference_cfg["max_generate_tokens"],
            max_seq_length=inference_cfg["max_seq_length"],
        )
        for embed_model in retrieval_cfg["embed_models"]:
            if embed_model == "none":
                out_path = FilePathBuilder.code_completion_result_path(
                    "baseline", 0, "none", llm, 0, 0,
                )
                if os.path.exists(out_path):
                    print("[Skip] baseline (exists)")
                    continue
                engine.run_baseline()
                continue
            for max_crossfile_context in inference_cfg["max_crossfile_contexts"]:
                for max_chunk_size in chunking["max_chunk_sizes"]:
                    for method in methods:
                        out_path = FilePathBuilder.code_completion_result_path(
                            method, max_chunk_size, embed_model, llm,
                            retrieval_cfg["top_k"], max_crossfile_context,
                        )
                        if os.path.exists(out_path):
                            print(f"[Skip] {method} | {max_chunk_size} | {max_crossfile_context} (exists)")
                            continue
                        engine.run_inference(
                            method, max_chunk_size, embed_model,
                            retrieval_cfg["top_k"], max_crossfile_context,
                        )


def step_score(cfg):
    """Compute scores with Java-aware identifier metrics."""
    from .compute_score import compute_EM, compute_ES, compute_id_metrics

    chunking = cfg["chunking"]
    retrieval_cfg = cfg["retrieval"]
    inference_cfg = cfg["inference"]
    eval_cfg = cfg["evaluation"]
    lang = chunking["language"]
    methods = CONSTANTS.ALL_METHODS if chunking["method"] == "all" else [chunking["method"]]
    passk = eval_cfg.get("passk", 1)

    combinations = Tools.scan_completion_directory()
    if not combinations:
        print("No completion results found")
        return

    all_results = []
    for retriever_name, llm in combinations:
        if retriever_name == "none":
            path = FilePathBuilder.code_completion_result_path(
                "baseline", 0, "none", llm, 0, 0,
            )
            if not os.path.exists(path):
                continue
            scores = _score_file(path, lang)
            all_results.append({
                "retriever": retriever_name,
                "llm": llm,
                "method": "baseline",
                "max_chunk_size": 0,
                "max_crossfile_context": 0,
                "top_k": 0,
                "passk": passk,
                **scores,
            })
            continue

        for method in methods:
            for chunk_size in chunking["max_chunk_sizes"]:
                for ctx_tokens in inference_cfg["max_crossfile_contexts"]:
                    path = FilePathBuilder.code_completion_result_path(
                        method, chunk_size, retriever_name, llm,
                        retrieval_cfg["top_k"], ctx_tokens,
                    )
                    if not os.path.exists(path):
                        continue
                    scores = _score_file(path, lang)
                    all_results.append({
                        "retriever": retriever_name,
                        "llm": llm,
                        "method": method,
                        "max_chunk_size": chunk_size,
                        "max_crossfile_context": ctx_tokens,
                        "top_k": retrieval_cfg["top_k"],
                        "passk": passk,
                        **scores,
                    })

    if not all_results:
        print("No results to report")
        return

    csv_path = os.path.join(
        os.path.dirname(FilePathBuilder.output_summary_result_path()),
        f"ablation_{lang}.csv",
    )
    FilePathBuilder._ensure_dir(csv_path)

    fieldnames = [
        "retriever", "llm", "method", "max_chunk_size",
        "max_crossfile_context", "top_k", "passk",
        "EM", "ES", "ID_EM", "ID_Precision", "ID_Recall", "ID_F1",
        "avg_token_cost", "total_inference_time",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: {csv_path}")
    print(f"Total entries: {len(all_results)}")


def _score_file(path, lang):
    """Load a completion file and compute all metrics."""
    from .compute_score import compute_EM, compute_ES, compute_id_metrics
    from .eval_utils import postprocess_code_lines

    lines = Tools.load_jsonl(path)
    total_time = None
    if lines and "total_inference_time" in lines[-1]:
        total_time = lines[-1]["total_inference_time"]
        lines = lines[:-1]

    em_scores, es_scores = [], []
    id_em_scores, id_prec_scores, id_rec_scores, id_f1_scores = [], [], [], []
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
    avg = lambda s: round(sum(s) / n, 4) if n else 0

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


def main():
    parser = argparse.ArgumentParser(description="Cross-language validation: cceval on Java.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--steps", nargs="+", required=True,
                        choices=["fetch", "chunk", "retrieve", "infer", "score"])
    parser.add_argument("--embed_model", type=str, default=None,
                        help="Run retrieve for a specific embed model only.")
    parser.add_argument("--llm", type=str, default=None,
                        help="Run infer for a specific LLM only.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Isolate intermediate files under ablation_java/ subdirectories
    FilePathBuilder._subdir = "ablation_java"

    for step in args.steps:
        print(f"\n{'=' * 60}")
        print(f"Step: {step}")
        print(f"{'=' * 60}")
        if step == "fetch":
            step_fetch(cfg)
        elif step == "chunk":
            step_chunk(cfg)
        elif step == "retrieve":
            step_retrieve(cfg, embed_model_filter=args.embed_model)
        elif step == "infer":
            step_infer(cfg, llm_filter=args.llm)
        elif step == "score":
            step_score(cfg)


if __name__ == "__main__":
    main()
