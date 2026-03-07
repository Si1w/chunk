"""Build code windows for CrossCodeEval benchmark evaluation.

This module generates two types of windows:
1. Repo windows: chunk repository source code using different chunking methods.
2. Query windows: extract query context (last N non-empty lines before groundtruth).
"""

import argparse

import yaml
from tqdm import tqdm
from chunk import CASTChunkBuilder, SlidingChunkBuilder, FunctionChunkBuilder, DeclarationChunkBuilder

from .utils import Tools, FilePathBuilder, CONSTANTS

METHODS = {
    "cast": CASTChunkBuilder,
    "function": FunctionChunkBuilder,
    "declaration": DeclarationChunkBuilder,
    "sliding": SlidingChunkBuilder,
}


def make_repo_window(repos, method, **configs):
    for repo in repos:
        worker = RepoWindowMaker(repo, method, **configs)
        worker.build_windows()


def make_query_window(repos, window_size=10, language="python"):
    worker = QueryWindowMaker(repos, window_size=window_size, language=language)
    worker.build_window()


class RepoWindowMaker:
    def __init__(self, repo, method, **configs):
        self.repo = repo
        self.method = method
        self.max_chunk_size = configs.get("max_chunk_size", 1024)
        self.source_code_files = Tools.iterate_repository(repo, configs.get("language", "python"))
        builder_cls = METHODS.get(method)
        if builder_cls is None:
            raise ValueError(f"Unsupported method: {method}")
        self.chunk_builder = builder_cls(**configs)

    def _build_windows_for_a_file(self, fpath_tuple, code):
        repo_name = self.repo
        # fpath_tuple is like (repo_name, "path", "to", "file.py")
        # file path relative to repo root
        fpath = "/".join(fpath_tuple[1:]) if len(fpath_tuple) > 1 else "/".join(fpath_tuple)
        chunks = self.chunk_builder.chunkify(
            code,
            repo_level_metadata={
                "repository": repo_name,
                "file": fpath,
            },
        )
        # Add chunk_id and fpath_tuple to each chunk's metadata (no merging)
        windows = []
        for i, chunk in enumerate(chunks):
            windows.append({
                "context": chunk["content"],
                "metadata": {
                    **chunk["metadata"],
                    "chunk_id": i,
                    "fpath_tuple": "/".join(fpath_tuple),
                },
            })
        return windows

    def build_windows(self):
        all_windows = []
        for fpath_tuple, code in self.source_code_files.items():
            all_windows += self._build_windows_for_a_file(fpath_tuple, code)
        print(f"build {len(all_windows)} windows for {self.repo}")
        output_path = FilePathBuilder.repo_windows_path(self.repo, self.method, self.max_chunk_size)
        Tools.dump_jsonl(all_windows, output_path)


class QueryWindowMaker:
    def __init__(self, repos, window_size=10, language="python"):
        self.repos = repos
        self.window_size = window_size
        self.language = language

    def build_window(self):
        datasets = Tools.load_jsonl(FilePathBuilder.benchmark_path())
        code_windows = []
        cur_repo = None
        source_code_files = None

        for dataset in tqdm(datasets, desc="Building query windows"):
            meta = dataset["metadata"]
            repo = meta["repository"].replace("/", "_")
            if self.repos and repo not in self.repos:
                continue
            if repo != cur_repo:
                cur_repo = repo
                source_code_files = Tools.iterate_repository(repo, self.language)

            fpath = meta["file"]
            # Find matching fpath_tuple
            fpath_tuple = None
            for ft in source_code_files:
                if "/".join(ft[1:]) == fpath or "/".join(ft) == fpath:
                    fpath_tuple = ft
                    break
            if fpath_tuple is None:
                continue

            original_code = source_code_files[fpath_tuple]
            code_lines = original_code.splitlines()
            gt_start = meta["groundtruth_start_lineno"]

            # Query = last window_size non-empty lines before gt_start
            candidate_lines = code_lines[:gt_start]
            non_empty = [l for l in candidate_lines if l.strip()]
            query_lines = non_empty[-self.window_size:]

            code_windows.append({
                "query": "\n".join(query_lines),
                "metadata": {
                    "fpath_tuple": "/".join(fpath_tuple),
                    "task_id": meta["task_id"],
                    "repo": cur_repo,
                    "ground_truth": dataset["groundtruth"],
                },
                "prompt": dataset["prompt"],
            })

        print(f"build {len(code_windows)} query windows")
        output_path = FilePathBuilder.query_windows_path(self.window_size)
        Tools.dump_jsonl(code_windows, output_path)


def main():
    parser = argparse.ArgumentParser(description="Build code windows for CrossCodeEval.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit to first N repos for dry runs.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    chunking = cfg["chunking"]
    query = cfg["query"]
    repos = CONSTANTS.repos()
    if args.num_samples:
        repos = repos[:args.num_samples]

    methods_to_run = list(METHODS) if chunking["method"] == "all" else [chunking["method"]]

    for max_chunk_size in chunking["max_chunk_sizes"]:
        configs = {
            "max_chunk_size": max_chunk_size,
            "language": chunking["language"],
            "metadata_template": chunking["metadata_template"],
            "overlap_lines": chunking["overlap_lines"],
            "chunk_expansion": chunking["chunk_expansion"],
            "private_function": chunking["private_function"],
            "function_overlap": chunking["function_overlap"],
        }
        for method in methods_to_run:
            print(f"\n--- Building windows: method={method}, max_chunk_size={max_chunk_size} ---")
            make_repo_window(repos, method, **configs)

    make_query_window(
        repos=repos,
        window_size=query["window_size"],
        language=chunking["language"],
    )


if __name__ == "__main__":
    main()
