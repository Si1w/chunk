"""Code completion inference for RepoEval benchmark.

Uses vLLM to generate code completions given retrieved cross-file context
and the original prompt from the benchmark dataset.
"""

import argparse
import time

import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams

from .utils import Tools, FilePathBuilder


class CodeCompletionInference:
    """Run code completion inference using vLLM with retrieved context."""

    def __init__(self, llm, method, max_chunk_size, embed_model,
                 max_generate_tokens=50, max_seq_length=8192, max_crossfile_context=2048):
        """Initialize the code completion inference engine.

        Args:
            llm: HuggingFace model name for code generation.
            method: Chunking method name.
            max_chunk_size: Maximum chunk size used during window building.
            embed_model: Embedding model name (used for path building).
            max_generate_tokens: Maximum number of tokens to generate.
            max_seq_length: Maximum total sequence length (prompt + generation).
            max_crossfile_context: Maximum token budget for cross-file context.
        """
        self.method = method
        self.max_chunk_size = max_chunk_size
        self.embed_model = embed_model
        self.llm_name = llm
        self.max_generate_tokens = max_generate_tokens
        self.max_seq_length = max_seq_length
        self.max_crossfile_context = max_crossfile_context

        self.model = LLM(model=llm, max_model_len=max_seq_length)
        self.tokenizer = self.model.get_tokenizer()
        self.sampling_params = SamplingParams(
            max_tokens=max_generate_tokens,
            temperature=0,
        )

    def _make_a_block(self, retrieved_window):
        """Format a retrieved code window as a commented block.

        Args:
            retrieved_window: Dict with 'content' and 'metadata' keys.

        Returns:
            Formatted string with file paths and commented code.
        """
        content = retrieved_window["content"]
        metadata_list = retrieved_window["metadata"]
        fpath_list = [meta["fpath_tuple"] for meta in metadata_list]
        fpath_str = "\n".join(fpath_list)
        header = "# the below code fragment can be found in:"
        content_lines = [f"# {line}" for line in content.splitlines()]
        return "\n".join([header, fpath_str] + content_lines) + "\n"

    def _build_prompt(self, prompt, retrieved_windows):
        """Build the final prompt by prepending retrieved cross-file context.

        Args:
            prompt: Original code prompt from the benchmark.
            retrieved_windows: List of retrieved window dicts.

        Returns:
            Full prompt string with cross-file context prepended.
        """
        prepend_context = "# Here are some relevant code fragments from other files of the repo:\n"
        prepend_blocks = []
        prompt_tokens = len(self.tokenizer.encode(prompt))
        cur_tokens = len(self.tokenizer.encode(prepend_context))
        context_budget = self.max_seq_length - self.max_generate_tokens - prompt_tokens

        for window in retrieved_windows:
            block = self._make_a_block(window)
            block_tokens = len(self.tokenizer.encode(block))
            if cur_tokens + block_tokens < min(context_budget, self.max_crossfile_context):
                prepend_blocks.append(block)
                cur_tokens += block_tokens

        return prepend_context + "".join(prepend_blocks) + prompt + "\n"

    def run_inference(self, method, max_chunk_size, embed_model, split, top_k):
        """Run code completion over the inference corpus and save results.

        Args:
            method: Chunking method name.
            max_chunk_size: Maximum chunk size used during window building.
            embed_model: Embedding model name (for path building).
            split: Dataset split ('api' or 'line').
            top_k: Number of retrieved windows used.
        """
        corpus_path = FilePathBuilder.inference_corpus_path(method, max_chunk_size, embed_model, split, top_k)
        inference_corpus = Tools.load_jsonl(corpus_path)

        prompts = []
        for corpus in inference_corpus:
            inference_prompt = self._build_prompt(corpus["prompt"], corpus["retrieved_windows"])
            prompts.append(inference_prompt)

        start_time = time.time()
        outputs = self.model.generate(prompts, self.sampling_params)
        total_time = round(time.time() - start_time, 2)

        code_completions = []
        for corpus, output in zip(inference_corpus, outputs):
            generated_text = output.outputs[0].text
            token_cost = len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            code_completions.append({
                "prompt": corpus["prompt"],
                "completion": generated_text,
                "ground_truth": corpus["ground_truth"],
                "token_cost": token_cost,
            })

        code_completions.append({"total_inference_time": total_time})
        print(f"[Time] {method} | {split} | total: {total_time}s | samples: {len(code_completions) - 1}")

        out_path = FilePathBuilder.code_completion_result_path(
            method, max_chunk_size, self.embed_model, self.llm_name, split, top_k, self.max_crossfile_context,
        )
        Tools.dump_jsonl(code_completions, out_path)

    def run_baseline(self, split, context_length, prompt_type):
        """Run code completion without retrieval (no cross-file context).

        Args:
            split: Dataset split ('api' or 'line').
            context_length: Context length variant (e.g. '2k').
            prompt_type: Prompt type ('codex' or 'codegen').
        """
        benchmark = Tools.load_jsonl(
            FilePathBuilder.benchmark_path(split, context_length, prompt_type)
        )

        prompts = [sample["prompt"] + "\n" for sample in benchmark]

        start_time = time.time()
        outputs = self.model.generate(prompts, self.sampling_params)
        total_time = round(time.time() - start_time, 2)

        code_completions = []
        for sample, output in zip(benchmark, outputs):
            code_completions.append({
                "prompt": sample["prompt"],
                "completion": output.outputs[0].text,
                "ground_truth": sample["metadata"]["ground_truth"],
                "token_cost": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
            })

        code_completions.append({"total_inference_time": total_time})
        print(f"[Time] baseline | {split} | total: {total_time}s | samples: {len(code_completions) - 1}")

        out_path = FilePathBuilder.code_completion_result_path(
            "baseline", 0, "none", self.llm_name, split, 0, 0,
        )
        Tools.dump_jsonl(code_completions, out_path)


def main():
    """Entry point: load YAML config and run code completion inference."""
    parser = argparse.ArgumentParser(description="Code completion inference for RepoEval.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--embed_model", type=str, default=None,
                        help="Run only for a specific embed model (default: all).")
    parser.add_argument("--llm", type=str, default=None,
                        help="Run only for a specific LLM (default: all).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    chunking = cfg["chunking"]
    query = cfg["query"]
    retrieval_cfg = cfg["retrieval"]
    inference_cfg = cfg["inference"]
    methods = list({"function", "declaration", "sliding"}) if chunking["method"] == "all" else [chunking["method"]]
    embed_models = [args.embed_model] if args.embed_model else retrieval_cfg["embed_models"]
    llms = [args.llm] if args.llm else inference_cfg["llms"]

    for llm in llms:
        for max_crossfile_context in inference_cfg["max_crossfile_contexts"]:
            inference = CodeCompletionInference(
                llm=llm,
                method=methods[0],
                max_chunk_size=chunking["max_chunk_sizes"][0],
                embed_model=embed_models[0],
                max_generate_tokens=inference_cfg["max_generate_tokens"],
                max_seq_length=inference_cfg["max_seq_length"],
                max_crossfile_context=max_crossfile_context,
            )
            for embed_model in embed_models:
                if embed_model == "none":
                    split = query.get("split", "api")
                    inference.run_baseline(split, query["context_length"], query["prompt_type"])
                    continue
                for max_chunk_size in chunking["max_chunk_sizes"]:
                    for method in methods:
                        split = query.get("split", "api")
                        inference.run_inference(
                            method, max_chunk_size, embed_model, split, retrieval_cfg["top_k"],
                        )


if __name__ == "__main__":
    main()
