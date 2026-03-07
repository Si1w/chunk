"""Retrieval for CrossCodeEval benchmark.

Supports dense embedding retrieval (SentenceTransformers + FAISS) and
sparse lexical retrieval (BM25) through a unified interface.
Implements next-chunk CFC retrieval strategy aligned with official cceval.
"""

import argparse
import os

import numpy as np
import yaml
from tqdm import tqdm

from .utils import Tools, FilePathBuilder, CONSTANTS


class Retriever:
    def __init__(self, repos, embed_model="bm25", batch_size=32):
        self.repos = repos
        self.embed_model = embed_model
        self.batch_size = batch_size
        self.is_bm25 = embed_model == "bm25"

        if not self.is_bm25:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embed_model, trust_remote_code=True)

    def _embed_texts(self, texts):
        embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True, convert_to_numpy=True)
        return np.array(embeddings, dtype="float32")

    def _build_id2idx(self, windows):
        """Build fpath|chunk_id -> index mapping for next-chunk lookup."""
        id2idx = {}
        for i, w in enumerate(windows):
            key = f"{w['metadata']['fpath_tuple']}|{w['metadata']['chunk_id']}"
            id2idx[key] = i
        return id2idx

    def _ensure_index(self, repo, method, max_chunk_size, rebuild=False):
        index_path = FilePathBuilder.index_window_path(repo, method, max_chunk_size, self.embed_model)
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

    def _search(self, index, query, k, corpus_size):
        k = min(k, corpus_size)
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

    def retrieval(self, method, max_chunk_size, top_k, num_queries=None, rebuild=False):
        query_lines = Tools.load_jsonl(FilePathBuilder.query_windows_path(window_size=10))
        if num_queries:
            query_lines = query_lines[:num_queries]

        inference_corpus = []
        cur_repo = None
        index = None
        windows = None
        id2idx = None

        label = "BM25" if self.is_bm25 else self.embed_model
        for query_line in tqdm(query_lines, desc=f"{label} retrieving top-{top_k}"):
            repo = query_line["metadata"]["repo"]
            if repo != cur_repo:
                index = self._ensure_index(repo, method, max_chunk_size, rebuild=rebuild)
                windows_path = FilePathBuilder.repo_windows_path(repo, method, max_chunk_size)
                windows = Tools.load_jsonl(windows_path)
                id2idx = self._build_id2idx(windows)
                cur_repo = repo

            query = query_line["query"]
            query_fpath = query_line["metadata"]["fpath_tuple"]
            indices, scores = self._search(index, query, top_k + 50, len(windows))

            retrieved = []
            seen_content = set()
            for idx, score in zip(indices, scores):
                if int(idx) < 0 or int(idx) >= len(windows):
                    continue
                w = windows[int(idx)]
                w_fpath = w["metadata"]["fpath_tuple"]

                # Same-file exclusion (by fpath, aligned with official)
                if w_fpath == query_fpath:
                    continue

                # Next-chunk CFC: use chunk N+1 instead of chunk N
                next_chunk_id = w["metadata"]["chunk_id"] + 1
                next_key = f"{w_fpath}|{next_chunk_id}"
                if next_key in id2idx:
                    cfc_window = windows[id2idx[next_key]]
                    content = cfc_window["context"]
                    metadata = cfc_window["metadata"]
                else:
                    # No next chunk available, use original
                    content = w["context"]
                    metadata = w["metadata"]

                # Deduplicate by content
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
                "prompt": query_line["prompt"],
                "retrieved_windows": retrieved[:top_k],
                "ground_truth": query_line["metadata"]["ground_truth"],
            })

        out_path = FilePathBuilder.inference_corpus_path(
            method, max_chunk_size, self.embed_model, top_k,
        )
        Tools.dump_jsonl(inference_corpus, out_path)


def main():
    parser = argparse.ArgumentParser(description="Retrieval for CrossCodeEval.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--embed_model", type=str, default=None)
    parser.add_argument("--num_queries", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    chunking = cfg["chunking"]
    retrieval_cfg = cfg["retrieval"]
    methods = CONSTANTS.ALL_METHODS if chunking["method"] == "all" else [chunking["method"]]
    embed_models = [args.embed_model] if args.embed_model else retrieval_cfg["embed_models"]

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
                retriever.retrieval(
                    method=method,
                    max_chunk_size=max_chunk_size,
                    top_k=retrieval_cfg["top_k"],
                    num_queries=args.num_queries,
                    rebuild=args.rebuild,
                )


if __name__ == "__main__":
    main()
