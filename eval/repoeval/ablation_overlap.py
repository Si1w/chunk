"""Ablation study for sliding window overlap parameters on RepoEval.

Evaluates the impact of different overlap values and chunk sizes on
code completion quality using sliding window chunking.
"""

import argparse
import csv
import os
from collections import defaultdict

import bm25s
import yaml
from tqdm import tqdm

from .code_completion import CodeCompletionInference
from .compute_score import compute_score_by_repo_with_metadata, compute_token_cost
from .utils import Tools, FilePathBuilder, CONSTANTS, is_context_file

_BASE_DIR = os.path.dirname(__file__)


class OverlapAblationStudy:
    """Ablation study varying overlap and chunk size for sliding window chunking."""

    def __init__(self, overlap_values, max_chunk_sizes, max_crossfile_contexts,
                 embed_model, llm, batch_size=32):
        self.overlap_values = overlap_values
        self.max_chunk_sizes = max_chunk_sizes
        self.max_crossfile_contexts = max_crossfile_contexts
        self.embed_model = embed_model
        self.llm = llm
        self.batch_size = batch_size
        self.method = "sliding"
        self.is_bm25 = embed_model == "bm25"

    def _ablation_window_path(self, repo, max_chunk_size, overlap):
        out = os.path.join(
            _BASE_DIR, "window", "ablation_overlap",
            f"{repo}_{self.method}_{max_chunk_size}_overlap{overlap}.jsonl",
        )
        FilePathBuilder._ensure_dir(out)
        return out

    def _ablation_index_path(self, repo, max_chunk_size, overlap):
        model_name = "bm25" if self.is_bm25 else Tools.safe_model_name(self.embed_model)
        out = os.path.join(
            _BASE_DIR, "index", "ablation_overlap", model_name,
            f"{repo}_{self.method}_{max_chunk_size}_overlap{overlap}.index",
        )
        FilePathBuilder._ensure_dir(out)
        return out

    def _ablation_inference_corpus_path(self, split, max_chunk_size, overlap, top_k):
        model_name = "bm25" if self.is_bm25 else Tools.safe_model_name(self.embed_model)
        out = os.path.join(
            _BASE_DIR, "inference_corpus", "ablation_overlap", model_name,
            f"{split}_{self.method}_{max_chunk_size}_overlap{overlap}_{top_k}.jsonl",
        )
        FilePathBuilder._ensure_dir(out)
        return out

    def _ablation_completion_path(self, split, max_chunk_size, overlap, top_k, max_crossfile_context):
        model_name = "bm25" if self.is_bm25 else Tools.safe_model_name(self.embed_model)
        safe_llm = Tools.safe_model_name(self.llm)
        out = os.path.join(
            _BASE_DIR, "completion", "ablation_overlap", model_name, safe_llm,
            f"{split}_{self.method}_{max_chunk_size}_overlap{overlap}_{top_k}_ctx{max_crossfile_context}.jsonl",
        )
        FilePathBuilder._ensure_dir(out)
        return out

    @staticmethod
    def ablation_result_path(split):
        out = os.path.join(
            _BASE_DIR, "result", "ablation_overlap",
            f"{split}_overlap_ablation_results.csv",
        )
        FilePathBuilder._ensure_dir(out)
        return out

    def build_windows(self, max_chunk_size, overlap, language="python"):
        """Build sliding windows for a given overlap and chunk size."""
        from chunk import SlidingChunkBuilder

        configs = {
            "max_chunk_size": max_chunk_size,
            "language": language,
            "metadata_template": "repoeval",
            "overlap_lines": overlap,
        }
        chunk_builder = SlidingChunkBuilder(**configs)

        for repo in tqdm(CONSTANTS.REPOs, desc=f"Building windows chunk_size={max_chunk_size}, overlap={overlap}"):
            source_code_files = Tools.iterate_repository(repo, language)
            all_code_windows = []

            for fpath_tuple, code in source_code_files.items():
                code_windows = chunk_builder.chunkify(
                    code,
                    repo_level_metadata={"repo": repo, "fpath_tuple": "/".join(fpath_tuple)},
                )
                all_code_windows.extend(code_windows)

            merged = defaultdict(list)
            for w in all_code_windows:
                merged[w["content"]].append(w["metadata"])

            json_lines = [
                {"context": context, "metadata": metadata_list}
                for context, metadata_list in merged.items()
            ]
            Tools.dump_jsonl(json_lines, self._ablation_window_path(repo, max_chunk_size, overlap))

    def _ensure_index(self, repo, max_chunk_size, overlap):
        """Build index for a repo if it doesn't exist, then load it."""
        index_path = self._ablation_index_path(repo, max_chunk_size, overlap)
        if not os.path.exists(index_path):
            window_path = self._ablation_window_path(repo, max_chunk_size, overlap)
            windows = Tools.load_jsonl(window_path)
            code_contents = [w["context"] for w in windows]

            if self.is_bm25:
                retriever = bm25s.BM25(corpus=code_contents)
                retriever.index(bm25s.tokenize(code_contents))
                retriever.save(index_path)
            else:
                import faiss
                embeddings = self._embed_model_instance.encode(
                    code_contents, batch_size=self.batch_size,
                    convert_to_numpy=True, show_progress_bar=False,
                )
                faiss.normalize_L2(embeddings)
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                faiss.write_index(index, index_path)

        if self.is_bm25:
            return bm25s.BM25.load(index_path, load_corpus=False)
        else:
            import faiss
            return faiss.read_index(index_path)

    def retrieval(self, max_chunk_size, overlap, split, context_length, prompt_type, top_k):
        """Retrieve top-k windows for the ablation configuration."""
        query_lines = Tools.load_jsonl(
            FilePathBuilder.query_windows_path(split, context_length, prompt_type, window_size=20)
        )
        inference_corpus = []
        cur_repo = None
        index = None
        windows = None

        for query_line in tqdm(query_lines, desc=f"Retrieving overlap={overlap}"):
            repo = query_line["metadata"]["repo"]
            if repo != cur_repo:
                index = self._ensure_index(repo, max_chunk_size, overlap)
                windows_path = self._ablation_window_path(repo, max_chunk_size, overlap)
                windows = Tools.load_jsonl(windows_path)
                cur_repo = repo

            query = query_line["query"]
            fpath_tuple = "/".join(query_line["metadata"]["fpath_tuple"])

            if self.is_bm25:
                results, scores = index.retrieve(bm25s.tokenize(query), k=top_k + 50)
                idx_list, score_list = results[0], scores[0]
            else:
                import faiss
                query_embedding = self._embed_model_instance.encode(
                    [query], convert_to_numpy=True,
                )
                faiss.normalize_L2(query_embedding)
                scores, indices = index.search(query_embedding, top_k + 50)
                idx_list, score_list = indices[0], scores[0]

            retrieved = []
            for idx, score in zip(idx_list, score_list):
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

        Tools.dump_jsonl(
            inference_corpus,
            self._ablation_inference_corpus_path(split, max_chunk_size, overlap, top_k),
        )

    def run_completion(self, inference, max_chunk_size, overlap, split, top_k,
                       max_crossfile_context):
        """Run code completion for the ablation configuration using vLLM.

        Args:
            inference: Reusable CodeCompletionInference instance.
        """
        corpus_path = self._ablation_inference_corpus_path(split, max_chunk_size, overlap, top_k)
        inference_corpus = Tools.load_jsonl(corpus_path)

        prompts = [
            inference._build_prompt(c["prompt"], c["retrieved_windows"], max_crossfile_context)
            for c in inference_corpus
        ]
        outputs = inference.model.generate(prompts, inference.sampling_params)

        code_completions = []
        for corpus, output in zip(inference_corpus, outputs):
            code_completions.append({
                "prompt": corpus["prompt"],
                "completion": output.outputs[0].text,
                "ground_truth": corpus["ground_truth"],
                "token_cost": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
            })

        Tools.dump_jsonl(
            code_completions,
            self._ablation_completion_path(split, max_chunk_size, overlap, top_k, max_crossfile_context),
        )

    def compute_scores(self, max_chunk_size, overlap, split, top_k, max_crossfile_context, passk=1):
        """Compute EM/ES scores for the ablation configuration."""
        completion_path = self._ablation_completion_path(split, max_chunk_size, overlap, top_k, max_crossfile_context)
        completion_lines = Tools.load_jsonl(completion_path)

        return {
            "retriever": "bm25" if self.is_bm25 else self.embed_model,
            "llm": self.llm,
            "max_chunk_size": max_chunk_size,
            "overlap": overlap,
            "max_crossfile_context": max_crossfile_context,
            "split": split,
            "top_k": top_k,
            "passk": passk,
            "EM": compute_score_by_repo_with_metadata(completion_lines, "EM", passk, self.llm),
            "ES": compute_score_by_repo_with_metadata(completion_lines, "ES", passk, self.llm),
            "avg_token_cost": compute_token_cost(completion_lines),
        }

    def run_ablation(self, split, context_length, prompt_type, top_k, passk=1,
                     max_seq_length=8192, max_generate_tokens=50,
                     skip_window=False, skip_retrieval=False, skip_completion=False):
        """Run the full ablation study pipeline."""
        if not self.is_bm25 and not hasattr(self, "_embed_model_instance"):
            from sentence_transformers import SentenceTransformer
            self._embed_model_instance = SentenceTransformer(self.embed_model, trust_remote_code=True)

        inference = None
        if not skip_completion:
            inference = CodeCompletionInference(
                llm=self.llm,
                max_generate_tokens=max_generate_tokens,
                max_seq_length=max_seq_length,
            )

        all_results = []

        for max_chunk_size in self.max_chunk_sizes:
            for overlap in self.overlap_values:
                print(f"\n--- Ablation: chunk_size={max_chunk_size}, overlap={overlap}, "
                      f"retriever={'bm25' if self.is_bm25 else self.embed_model} ---")

                if not skip_window:
                    self.build_windows(max_chunk_size, overlap)
                if not skip_retrieval:
                    self.retrieval(max_chunk_size, overlap, split, context_length, prompt_type, top_k)

                for max_crossfile_context in self.max_crossfile_contexts:
                    if not skip_completion:
                        self.run_completion(
                            inference, max_chunk_size, overlap, split, top_k,
                            max_crossfile_context,
                        )

                    result = self.compute_scores(
                        max_chunk_size, overlap, split, top_k, max_crossfile_context, passk,
                    )
                    all_results.append(result)

        return all_results


def main():
    """Entry point: load YAML config and run overlap ablation study."""
    parser = argparse.ArgumentParser(description="Overlap ablation study for RepoEval.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--skip_window", action="store_true", help="Skip window building step.")
    parser.add_argument("--skip_retrieval", action="store_true", help="Skip retrieval step.")
    parser.add_argument("--skip_completion", action="store_true", help="Skip code completion step.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ablation_cfg = cfg["ablation"]
    query = cfg["query"]
    retrieval_cfg = cfg["retrieval"]
    inference_cfg = cfg["inference"]
    eval_cfg = cfg["evaluation"]

    splits = ["api", "line"] if eval_cfg.get("split") == "both" else [eval_cfg.get("split", "api")]

    for split in splits:
        all_results = []

        for embed_model in retrieval_cfg["embed_models"]:
            for llm in inference_cfg["llms"]:
                ablation = OverlapAblationStudy(
                    overlap_values=ablation_cfg["overlap_values"],
                    max_chunk_sizes=ablation_cfg["max_chunk_sizes"],
                    max_crossfile_contexts=ablation_cfg["max_crossfile_contexts"],
                    embed_model=embed_model,
                    llm=llm,
                    batch_size=retrieval_cfg.get("batch_size", 32),
                )

                results = ablation.run_ablation(
                    split=split,
                    context_length=query["context_length"],
                    prompt_type=query["prompt_type"],
                    top_k=retrieval_cfg["top_k"],
                    passk=eval_cfg.get("passk", 1),
                    max_seq_length=inference_cfg["max_seq_length"],
                    max_generate_tokens=inference_cfg["max_generate_tokens"],
                    skip_window=args.skip_window,
                    skip_retrieval=args.skip_retrieval,
                    skip_completion=args.skip_completion,
                )
                all_results.extend(results)

        result_path = OverlapAblationStudy.ablation_result_path(split)
        with open(result_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["retriever", "llm", "max_chunk_size", "overlap",
                          "max_crossfile_context", "split", "top_k", "passk",
                          "EM", "ES", "avg_token_cost"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\nResults saved to: {result_path} ({len(all_results)} combinations)")


if __name__ == "__main__":
    main()
