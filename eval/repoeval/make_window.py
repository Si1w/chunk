"""Build code windows for RepoEval benchmark evaluation.

This module generates two types of windows:
1. Repo windows: chunk repository source code using different chunking methods.
2. Query windows: extract code context before each query line in the benchmark.
"""

import argparse
from collections import defaultdict

import yaml
from tqdm import tqdm
from chunk import SlidingChunkBuilder, FunctionChunkBuilder, DeclarationChunkBuilder

from .utils import Tools, FilePathBuilder, CONSTANTS

METHODS = {
    "function": FunctionChunkBuilder,
    "declaration": DeclarationChunkBuilder,
    "sliding": SlidingChunkBuilder,
}


def make_repo_window(repos, method, **configs):
    """Build and save code windows for each repository.

    Args:
        repos: List of repository names to process.
        method: Chunking method name (key in METHODS).
        **configs: Chunking configuration passed to the builder.
    """
    for repo in repos:
        worker = RepoWindowMaker(repo, method, **configs)
        worker.build_windows()


def make_query_window(context_length, prompt_type, repos, window_size=20, language="python"):
    """Build and save query windows for api and line splits.

    Args:
        context_length: Context length variant (e.g. '1k', '2k', '4k').
        prompt_type: Prompt type ('codex' or 'codegen').
        repos: List of repository names.
        window_size: Number of lines before the query line to include.
        language: Programming language of the repositories.
    """
    for split in ["api", "line"]:
        worker = QueryWindowMaker(
            split, context_length, prompt_type, repos,
            window_size=window_size, language=language,
        )
        worker.build_window()


class RepoWindowMaker:
    """Chunk repository source files into code windows."""

    def __init__(self, repo, method, **configs):
        """Initialize the window maker for a single repository.

        Args:
            repo: Repository name.
            method: Chunking method name (key in METHODS).
            **configs: Chunking configuration (max_chunk_size, language, etc.).
        """
        self.repo = repo
        self.method = method
        self.max_chunk_size = configs.get("max_chunk_size", 1024)
        self.source_code_files = Tools.iterate_repository(repo, configs.get("language", "python"))

        builder_cls = METHODS.get(method)
        if builder_cls is None:
            raise ValueError(f"Unsupported method: {method}")
        self.chunk_builder = builder_cls(**configs)

    def _build_windows_for_a_file(self, fpath_tuple, code):
        """Chunk a single source file and return code windows."""
        return self.chunk_builder.chunkify(
            code,
            repo_level_metadata={"repo": self.repo, "fpath_tuple": "/".join(fpath_tuple)},
        )

    def _merge_windows_with_same_context(self, code_windows):
        """Merge windows that have identical content, combining their metadata."""
        merged = defaultdict(list)
        for w in code_windows:
            merged[w["content"]].append(w["metadata"])
        return [{"context": ctx, "metadata": metas} for ctx, metas in merged.items()]

    def build_windows(self):
        """Build windows for all files in the repository and save to disk."""
        all_windows = []
        for fpath_tuple, code in self.source_code_files.items():
            all_windows += self._build_windows_for_a_file(fpath_tuple, code)
        merged = self._merge_windows_with_same_context(all_windows)
        print(f"build {len(merged)} windows for {self.repo}")
        output_path = FilePathBuilder.repo_windows_path(self.repo, self.method, self.max_chunk_size)
        Tools.dump_jsonl(merged, output_path)


class QueryWindowMaker:
    """Extract code context windows for each query in the benchmark dataset."""

    def __init__(self, split, ctx_len, prompt, repos, window_size=20, language="python"):
        """Initialize the query window maker.

        Args:
            split: Dataset split ('api' or 'line').
            ctx_len: Context length variant (e.g. '1k', '2k', '4k').
            prompt: Prompt type ('codex' or 'codegen').
            repos: List of repository names.
            window_size: Number of lines before the query line to include.
            language: Programming language of the repositories.
        """
        self.split = split
        self.context_length = ctx_len
        self.prompt_type = prompt
        self.repos = repos
        self.window_size = window_size
        self.language = language

    def build_window(self):
        """Build query windows from the benchmark dataset and save to disk."""
        datasets = Tools.load_jsonl(
            FilePathBuilder.benchmark_path(self.split, self.context_length, self.prompt_type)
        )
        code_windows = []
        cur_repo = None
        source_code_files = None

        for dataset in tqdm(datasets, desc=f"Building query windows for {self.split} set"):
            repo = dataset["metadata"]["task_id"].split("/")[0]
            if repo not in self.repos:
                continue
            if repo != cur_repo:
                cur_repo = repo
                source_code_files = Tools.iterate_repository(repo, self.language)

            fpath_tuple = tuple(dataset["metadata"]["fpath_tuple"])
            line_no = dataset["metadata"]["line_no"]
            original_code = source_code_files[fpath_tuple]
            code_lines = original_code.splitlines()
            context_start_lineno = dataset["metadata"]["context_start_lineno"]
            start_line_no = max(context_start_lineno, line_no - self.window_size)
            window_lines = code_lines[start_line_no:line_no]

            code_windows.append({
                "query": "\n".join(window_lines),
                "metadata": {
                    "fpath_tuple": fpath_tuple,
                    "line_no": line_no,
                    "task_id": dataset["metadata"]["task_id"],
                    "start_line_no": start_line_no,
                    "end_line_no": line_no,
                    "window_size": self.window_size,
                    "context_start_lineno": context_start_lineno,
                    "repo": cur_repo,
                    "ground_truth": dataset["metadata"]["ground_truth"],
                },
                "prompt": dataset["prompt"],
            })

        print(f"build {len(code_windows)} query windows for {self.split} set")
        output_path = FilePathBuilder.query_windows_path(
            self.split, self.context_length, self.prompt_type, self.window_size,
        )
        Tools.dump_jsonl(code_windows, output_path)


def main():
    """Entry point: load YAML config and build repo + query windows."""
    parser = argparse.ArgumentParser(description="Build code windows for RepoEval benchmark.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit to first N repos for dry runs; omit to run all.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    chunking = cfg["chunking"]
    query = cfg["query"]
    repos = CONSTANTS.REPOs[:args.num_samples] if args.num_samples else CONSTANTS.REPOs

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
            print(f"\n --- Building windows: method={method}, max_chunk_size={max_chunk_size} ---")
            make_repo_window(repos, method, **configs)

    make_query_window(
        context_length=query["context_length"],
        prompt_type=query["prompt_type"],
        repos=repos,
        window_size=query["window_size"],
        language=chunking["language"],
    )


if __name__ == "__main__":
    main()
