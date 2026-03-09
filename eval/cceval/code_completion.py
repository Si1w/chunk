"""Code completion inference for CrossCodeEval benchmark.

Uses vLLM to generate code completions. Prompt construction aligns with
official cceval: right-truncate CFC comments, left-truncate prompt.
"""

import argparse
import os
import time

import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams

from .utils import Tools, FilePathBuilder, CONSTANTS


class CodeCompletionInference:
    def __init__(self, llm, max_generate_tokens=50, max_seq_length=8192):
        self.llm_name = llm
        self.max_generate_tokens = max_generate_tokens
        self.max_seq_length = max_seq_length

        self.model = LLM(model=llm, max_model_len=max_seq_length)
        self.tokenizer = self.model.get_tokenizer()
        self.sampling_params = SamplingParams(
            max_tokens=max_generate_tokens,
            temperature=0,
        )

    def _make_a_block(self, retrieved_window):
        """Format a retrieved window as Python comments (aligned with official cceval)."""
        content = retrieved_window["content"]
        fpath = retrieved_window["metadata"].get("fpath_tuple", "")
        # Strip repo name prefix if present (repo_name/path/to/file.py -> path/to/file.py)
        parts = fpath.split("/")
        if len(parts) > 1:
            filepath = "/".join(parts[1:])
        else:
            filepath = fpath

        header = "# the below code fragment can be found in:"
        fpath_line = f"# {filepath}"
        content_lines = [f"# {line}" for line in content.splitlines()]
        return "\n".join([header, fpath_line] + content_lines) + "\n"

    def _build_prompt(self, prompt, retrieved_windows, max_crossfile_context):
        """Build prompt: right-truncate CFC to budget, left-truncate prompt.

        Aligned with official cceval prompt construction.
        """
        # Build CFC blocks and right-truncate to crossfile_max_tokens budget
        cfc_blocks = []
        cfc_tokens = 0
        for window in retrieved_windows:
            block = self._make_a_block(window)
            block_tokens = len(self.tokenizer.encode(block))
            if cfc_tokens + block_tokens > max_crossfile_context:
                break
            cfc_blocks.append(block)
            cfc_tokens += block_tokens

        cfc_text = "\n".join(cfc_blocks)
        if cfc_text:
            cfc_tokens = len(self.tokenizer.encode(cfc_text))
        else:
            cfc_tokens = 0

        # Left-truncate prompt to fit remaining budget
        prompt_budget = self.max_seq_length - self.max_generate_tokens - cfc_tokens - 100
        prompt_token_ids = self.tokenizer.encode(prompt)
        if len(prompt_token_ids) > prompt_budget:
            prompt_token_ids = prompt_token_ids[-prompt_budget:]
            prompt = self.tokenizer.decode(prompt_token_ids)

        if cfc_text:
            return cfc_text + "\n" + prompt
        return prompt

    def run_inference(self, method, max_chunk_size, embed_model, top_k, max_crossfile_context):
        corpus_path = FilePathBuilder.inference_corpus_path(method, max_chunk_size, embed_model, top_k)
        inference_corpus = Tools.load_jsonl(corpus_path)

        prompts = []
        for corpus in inference_corpus:
            inference_prompt = self._build_prompt(
                corpus["prompt"], corpus["retrieved_windows"], max_crossfile_context,
            )
            token_ids = self.tokenizer.encode(inference_prompt)
            max_tokens = self.max_seq_length - self.max_generate_tokens
            if len(token_ids) > max_tokens:
                token_ids = token_ids[-max_tokens:]
            prompts.append({"prompt_token_ids": token_ids})

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
        print(f"[Time] {method} | total: {total_time}s | samples: {len(code_completions) - 1}")

        out_path = FilePathBuilder.code_completion_result_path(
            method, max_chunk_size, embed_model, self.llm_name, top_k, max_crossfile_context,
        )
        Tools.dump_jsonl(code_completions, out_path)

    def run_baseline(self):
        """Run baseline inference without retrieval."""
        benchmark = Tools.load_jsonl(FilePathBuilder.benchmark_path())

        prompt_budget = self.max_seq_length - self.max_generate_tokens
        prompts = []
        for sample in benchmark:
            token_ids = self.tokenizer.encode(sample["prompt"])
            if len(token_ids) > prompt_budget:
                token_ids = token_ids[-prompt_budget:]
            prompts.append({"prompt_token_ids": token_ids})

        start_time = time.time()
        outputs = self.model.generate(prompts, self.sampling_params)
        total_time = round(time.time() - start_time, 2)

        code_completions = []
        for sample, output in zip(benchmark, outputs):
            code_completions.append({
                "prompt": sample["prompt"],
                "completion": output.outputs[0].text,
                "ground_truth": sample["groundtruth"],
                "token_cost": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
            })

        code_completions.append({"total_inference_time": total_time})
        print(f"[Time] baseline | total: {total_time}s | samples: {len(code_completions) - 1}")

        out_path = FilePathBuilder.code_completion_result_path(
            "baseline", 0, "none", self.llm_name, 0, 0,
        )
        Tools.dump_jsonl(code_completions, out_path)


def main():
    parser = argparse.ArgumentParser(description="Code completion inference for CrossCodeEval.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--embed_model", type=str, default=None)
    parser.add_argument("--llm", type=str, default=None)
    parser.add_argument("--full", action="store_true",
                        help="Re-run all combinations even if output files already exist.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    chunking = cfg["chunking"]
    retrieval_cfg = cfg["retrieval"]
    inference_cfg = cfg["inference"]
    methods = CONSTANTS.ALL_METHODS if chunking["method"] == "all" else [chunking["method"]]
    embed_models = [args.embed_model] if args.embed_model else retrieval_cfg["embed_models"]
    llms = [args.llm] if args.llm else inference_cfg["llms"]

    for llm in llms:
        inference = CodeCompletionInference(
            llm=llm,
            max_generate_tokens=inference_cfg["max_generate_tokens"],
            max_seq_length=inference_cfg["max_seq_length"],
        )
        for embed_model in embed_models:
            if embed_model == "none":
                out_path = FilePathBuilder.code_completion_result_path(
                    "baseline", 0, "none", llm, 0, 0,
                )
                if not args.full and os.path.exists(out_path):
                    print(f"[Skip] baseline (exists)")
                    continue
                inference.run_baseline()
                continue
            for max_crossfile_context in inference_cfg["max_crossfile_contexts"]:
                for max_chunk_size in chunking["max_chunk_sizes"]:
                    for method in methods:
                        out_path = FilePathBuilder.code_completion_result_path(
                            method, max_chunk_size, embed_model, llm,
                            retrieval_cfg["top_k"], max_crossfile_context,
                        )
                        if not args.full and os.path.exists(out_path):
                            print(f"[Skip] {method} | {max_chunk_size} | {max_crossfile_context} (exists)")
                            continue
                        inference.run_inference(
                            method, max_chunk_size, embed_model,
                            retrieval_cfg["top_k"], max_crossfile_context,
                        )


if __name__ == "__main__":
    main()
