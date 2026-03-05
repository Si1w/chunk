"""Retrieval for RepoEval benchmark.

Supports dense embedding retrieval (SentenceTransformers + FAISS) and
sparse lexical retrieval (BM25) through a unified interface.
"""

import argparse
import os

import numpy as np
import yaml
from tqdm import tqdm

from .utils import Tools, FilePathBuilder, CONSTANTS, is_context_file


class Retriever:
    """Build indices and perform retrieval over code windows."""

    def __init__(self, repos, embed_model="Qwen/Qwen3-Embedding-0.6B", batch_size=32):
        """Initialize the retriever (loads model once).

        Args:
            repos: List of repository names.
            embed_model: HuggingFace model name for embeddings, or 'bm25' for lexical retrieval.
            batch_size: Batch size for embedding inference.
        """
        self.repos = repos
        self.embed_model = embed_model
        self.batch_size = batch_size
        self.is_bm25 = embed_model == "bm25"

        if not self.is_bm25:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embed_model, trust_remote_code=True)

    def _embed_texts(self, texts):
        """Encode a list of texts into numpy embeddings."""
        embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True, convert_to_numpy=True)
        return np.array(embeddings, dtype="float32")

    def _ensure_index(self, repo, method, max_chunk_size, rebuild=False):
        """Build index for a repo if it doesn't exist, then load it."""
        index_path = FilePathBuilder.index_window_path(
            repo, method, max_chunk_size, self.embed_model,
        )
        if not os.path.exists(index_path) or rebuild:
            window_path = FilePathBuilder.repo_windows_path(repo, method, max_chunk_size)
            windows = Tools.load_jsonl(window_path)
            code_contents = [w["context"] for w in windows]

            if self.is_bm25:
                import bm25s
                retriever = bm25s.BM25(corpus=code_contents)
                retriever.index(bm25s.tokenize(code_contents))
                retriever.save(index_path)
            else:
                import faiss
                embeddings = self._embed_texts(code_contents)
                faiss.normalize_L2(embeddings)
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                faiss.write_index(index, index_path)

        if self.is_bm25:
            import bm25s
            return bm25s.BM25.load(index_path, load_corpus=False)
        else:
            import faiss
            return faiss.read_index(index_path)

    def _search(self, index, query, k):
        """Search the index for the given query, returning (indices, scores)."""
        if self.is_bm25:
            import bm25s
            results, scores = index.retrieve(bm25s.tokenize(query), k=k)
            return results[0], scores[0]
        else:
            import faiss
            query_embedding = self._embed_texts([query])
            faiss.normalize_L2(query_embedding)
            scores, indices = index.search(query_embedding, k)
            return indices[0], scores[0]

    def retrieval(self, method, max_chunk_size, split, context_length, prompt_type, top_k,
                  num_queries=None, rebuild=False):
        """Retrieve top-k windows for each query and save inference corpus.

        Args:
            method: Chunking method name.
            max_chunk_size: Maximum chunk size used during window building.
            split: Dataset split ('api' or 'line').
            context_length: Context length variant (e.g. '2k').
            prompt_type: Prompt type ('codex' or 'codegen').
            top_k: Number of windows to retrieve per query.
            num_queries: Limit number of queries (for pilot runs).
            rebuild: If True, rebuild index even if it exists.
        """
        query_lines = Tools.load_jsonl(
            FilePathBuilder.query_windows_path(split, context_length, prompt_type, window_size=20)
        )
        if num_queries:
            query_lines = query_lines[:num_queries]

        inference_corpus = []
        cur_repo = None
        index = None
        windows = None

        label = "BM25" if self.is_bm25 else self.embed_model
        for query_line in tqdm(query_lines, desc=f"{label} retrieving top-{top_k} for {split}"):
            repo = query_line["metadata"]["repo"]
            if repo != cur_repo:
                index = self._ensure_index(repo, method, max_chunk_size, rebuild=rebuild)
                windows_path = FilePathBuilder.repo_windows_path(repo, method, max_chunk_size)
                windows = Tools.load_jsonl(windows_path)
                cur_repo = repo

            query = query_line["query"]
            fpath_tuple = "/".join(query_line["metadata"]["fpath_tuple"])
            indices, scores = self._search(index, query, top_k + 50)

            retrieved = []
            for idx, score in zip(indices, scores):
                if is_context_file(windows[idx], fpath_tuple, query_line):
                    continue
                retrieved.append({
                    "content": windows[idx]["context"],
                    "metadata": windows[idx]["metadata"],
                    "score": float(score),
                })
            retrieved.sort(key=lambda x: x["score"], reverse=True)

            inference_corpus.append({
                "prompt": query_line["prompt"],
                "retrieved_windows": retrieved[:top_k],
                "ground_truth": query_line["metadata"]["ground_truth"],
            })

        out_path = FilePathBuilder.inference_corpus_path(
            method, max_chunk_size, self.embed_model, split, top_k,
        )
        Tools.dump_jsonl(inference_corpus, out_path)


def main():
    """Entry point: load YAML config, build indices, and retrieve."""
    parser = argparse.ArgumentParser(description="Retrieval for RepoEval.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index even if it exists.")
    parser.add_argument("--embed_model", type=str, default=None,
                        help="Run only for a specific embed model (default: all).")
    parser.add_argument("--split", type=str, default=None,
                        help="Run only for a specific split: api | line (default: from config).")
    parser.add_argument("--num_queries", type=int, default=None,
                        help="Limit number of queries (for pilot runs).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    chunking = cfg["chunking"]
    query = cfg["query"]
    retrieval_cfg = cfg["retrieval"]
    methods = CONSTANTS.ALL_METHODS if chunking["method"] == "all" else [chunking["method"]]
    embed_models = [args.embed_model] if args.embed_model else retrieval_cfg["embed_models"]

    eval_split = cfg.get("evaluation", {}).get("split", "both")
    if args.split:
        splits = [args.split]
    elif eval_split == "both":
        splits = ["api", "line"]
    else:
        splits = [eval_split]

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
                        num_queries=args.num_queries,
                        rebuild=args.rebuild,
                    )


if __name__ == "__main__":
    main()
