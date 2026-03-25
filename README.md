# Chunk

> Structure-aware code chunking library for retrieval-augmented code completion.

## Overview

Chunk provides three chunking strategies that split source code into semantically meaningful segments:

- **Function chunking** - splits code at function boundaries
- **Declaration chunking** - splits code at declaration boundaries (classes, functions, etc.)
- **Sliding window chunking** - splits code using a sliding window with configurable overlap

Supports Python, Java, C#, and TypeScript via [tree-sitter](https://tree-sitter.github.io/).

## Quick Start

### Installation

```bash
# Core library only
uv pip install .

# With evaluation dependencies (vLLM, sentence-transformers, faiss, etc.)
uv pip install ".[eval]"
```

### Library Usage

```python
from chunk import FunctionChunkBuilder, DeclarationChunkBuilder, SlidingChunkBuilder

builder = FunctionChunkBuilder(
    max_chunk_size=2048,
    language="python",
    metadata_template="default",
    overlap_lines=5,
    private_function=False,
    function_overlap=False,
)

chunks = builder.chunkify(source_code)
for c in chunks:
    print(c["content"], c["metadata"])
```

Or run the example script:

```bash
uv run python examples/chunking.py --method function   # or declaration, sliding
```

## Evaluation

The `eval/` module implements full evaluation pipelines on two benchmarks:
- [RepoEval](https://github.com/microsoft/CodeT) (API-level and line-level completion)
- [CrossCodeEval](https://github.com/amazon-science/cceval) (cross-file code completion)

Both follow the same pipeline: chunking → retrieval → code completion → scoring.

Full result CSVs:
- RepoEval: `eval/repoeval/result/` (`api_all_results.csv`, `line_all_results.csv`, `ablation_overlap.csv`)
- CrossCodeEval: `eval/cceval/result/` (`all_results.csv`, `ablation_java.csv`)

### Environment Setup

```bash
sbatch scripts/setup_venv.sh
```

### Pipeline

Each benchmark has scripts for the four pipeline stages. Replace `repoeval` with `cceval` for CrossCodeEval.

```bash
# 1. Fetch dataset & chunking
sbatch scripts/repoeval_chunking.sh

# 2. Retrieval (BM25 + dense embeddings)
bash scripts/repoeval_retrieval.sh configs/repoeval.yaml

# 3. Code completion (vLLM, requires GPU)
bash scripts/repoeval_inference.sh configs/repoeval.yaml

# 4. Compute scores (EM & ES)
sbatch scripts/repoeval_score.sh configs/repoeval.yaml
```

Pilot run for quick sanity check:

```bash
sbatch scripts/repoeval_pilot.sh configs/repoeval_pilot.yaml
```

## Results

Metrics: **Exact Match (EM)** and **Edit Similarity (ES)**. All values are averaged across retriever × LLM × parameter configurations.

### RQ1: Strategy Effect

**RepoEval**

| Method | EM (API) | ES (API) | EM (Line) | ES (Line) |
|--------|----------|----------|-----------|-----------|
| Baseline | 0.3413 | 0.6280 | 0.4380 | 0.6653 |
| Function | 0.4227 | 0.6925 | 0.5127 | 0.7175 |
| Declaration | 0.4585 | 0.7179 | 0.5484 | 0.7389 |
| Sliding | 0.4623 | 0.7300 | 0.5691 | 0.7521 |
| CAST | 0.4593 | 0.7284 | 0.5654 | 0.7506 |

**CrossCodeEval**

| Method | EM | ES |
|--------|----|----|
| Baseline | 0.0914 | 62.48 |
| Function | 0.2421 | 71.67 |
| Declaration | 0.2771 | 73.35 |
| Sliding | 0.2840 | 73.79 |
| CAST | 0.2819 | 73.81 |

### RQ2: Interaction Effect

#### (a) Method × Retriever

**RepoEval — API Split (EM / ES)**

| Method | BM25 | GemmaEmb-300M | Qwen3Emb-0.6B | Qwen3Emb-4B |
|--------|------|---------------|----------------|--------------|
| Function | 0.4253 / 0.6909 | 0.4237 / 0.6952 | 0.4241 / 0.6938 | 0.4176 / 0.6899 |
| Declaration | 0.4633 / 0.7193 | 0.4599 / 0.7199 | 0.4547 / 0.7157 | 0.4560 / 0.7164 |
| Sliding | 0.4674 / 0.7365 | 0.4620 / 0.7339 | 0.4584 / 0.7245 | 0.4613 / 0.7249 |
| CAST | 0.4661 / 0.7351 | 0.4608 / 0.7336 | 0.4552 / 0.7236 | 0.4554 / 0.7212 |

**RepoEval — Line Split (EM / ES)**

| Method | BM25 | GemmaEmb-300M | Qwen3Emb-0.6B | Qwen3Emb-4B |
|--------|------|---------------|----------------|--------------|
| Function | 0.5111 / 0.7172 | 0.5136 / 0.7166 | 0.5132 / 0.7193 | 0.5129 / 0.7171 |
| Declaration | 0.5472 / 0.7386 | 0.5508 / 0.7390 | 0.5474 / 0.7395 | 0.5481 / 0.7386 |
| Sliding | 0.5762 / 0.7565 | 0.5666 / 0.7503 | 0.5651 / 0.7507 | 0.5684 / 0.7509 |
| CAST | 0.5687 / 0.7514 | 0.5648 / 0.7505 | 0.5648 / 0.7517 | 0.5633 / 0.7488 |

**CrossCodeEval (EM / ES)**

| Method | BM25 | GemmaEmb-300M | Qwen3Emb-0.6B | Qwen3Emb-4B |
|--------|------|---------------|----------------|--------------|
| Function | 0.2340 / 71.28 | 0.2398 / 71.59 | 0.2510 / 72.14 | 0.2435 / 71.68 |
| Declaration | 0.2653 / 72.90 | 0.2744 / 73.41 | 0.2871 / 73.67 | 0.2814 / 73.43 |
| Sliding | 0.2760 / 73.27 | 0.2823 / 73.98 | 0.2915 / 73.98 | 0.2860 / 73.92 |
| CAST | 0.2684 / 73.14 | 0.2842 / 74.03 | 0.2873 / 73.90 | 0.2879 / 74.16 |

#### (b) Method × Generator

**RepoEval — API Split (EM / ES)**

| Method | DeepSeek-6.7B | Qwen2.5-7B | Qwen3.5-9B | Seed-8B |
|--------|---------------|------------|------------|---------|
| Function | 0.3965 / 0.6750 | 0.4331 / 0.7012 | 0.4214 / 0.6826 | 0.4398 / 0.7110 |
| Declaration | 0.4355 / 0.7052 | 0.4715 / 0.7265 | 0.4567 / 0.7088 | 0.4702 / 0.7310 |
| Sliding | 0.4418 / 0.7194 | 0.4750 / 0.7382 | 0.4614 / 0.7274 | 0.4710 / 0.7348 |
| CAST | 0.4349 / 0.7150 | 0.4734 / 0.7369 | 0.4582 / 0.7267 | 0.4709 / 0.7349 |

**RepoEval — Line Split (EM / ES)**

| Method | DeepSeek-6.7B | Qwen2.5-7B | Qwen3.5-9B | Seed-8B |
|--------|---------------|------------|------------|---------|
| Function | 0.4858 / 0.6965 | 0.5230 / 0.7255 | 0.5022 / 0.7104 | 0.5399 / 0.7378 |
| Declaration | 0.5279 / 0.7242 | 0.5520 / 0.7423 | 0.5421 / 0.7341 | 0.5715 / 0.7550 |
| Sliding | 0.5517 / 0.7399 | 0.5712 / 0.7545 | 0.5644 / 0.7487 | 0.5890 / 0.7652 |
| CAST | 0.5464 / 0.7369 | 0.5692 / 0.7535 | 0.5619 / 0.7479 | 0.5840 / 0.7642 |

**CrossCodeEval (EM / ES)**

| Method | DeepSeek-6.7B | StarCoder2-7B |
|--------|---------------|---------------|
| Function | 0.2563 / 72.43 | 0.2278 / 70.91 |
| Declaration | 0.2901 / 74.09 | 0.2640 / 72.62 |
| Sliding | 0.2991 / 74.53 | 0.2688 / 73.04 |
| CAST | 0.2964 / 74.51 | 0.2674 / 73.10 |

## License

[MIT](LICENSE)
