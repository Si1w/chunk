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

## RepoEval Benchmark

The `eval/` module implements a full evaluation pipeline on the [RepoEval](https://github.com/microsoft/CodeT) benchmark:
chunking → retrieval → code completion → scoring.

### Container Setup (Optional)

All experiments can run inside a Singularity container for reproducibility. One-time setup:

```bash
bash scripts/setup_container.sh
```

This pulls the NVIDIA PyTorch image to `/scratch/users/$USER/images/`. The SLURM scripts automatically use the container via `singularity exec --nv`.

### 1. Fetch Dataset

```bash
uv run python -m eval.repoeval.fetch_dataset
```

### 2. Chunking

```bash
uv run python -m eval.repoeval.make_window --config configs/repoeval.yaml
```

### 3. Retrieval

Supports BM25 (sparse) and dense embedding models (SentenceTransformers + FAISS).

```bash
# Run all embed models defined in config
bash scripts/repoeval_retrieval.sh configs/repoeval.yaml

# Or run a single model
uv run python -m eval.repoeval.retrieval --config configs/repoeval.yaml --embed_model bm25
```

### 4. Code Completion (Inference)

Uses vLLM for batched code generation. Requires GPU.

```bash
# Submit all (embed_model, llm) pairs as SLURM jobs
bash scripts/repoeval_inference.sh configs/repoeval.yaml

# Or run a single pair
uv run python -m eval.repoeval.code_completion --config configs/repoeval.yaml \
    --embed_model bm25 --llm Qwen/Qwen2.5-Coder-7B
```

### 5. Compute Scores

Computes Exact Match (EM) and Edit Similarity (ES) metrics, outputs a CSV summary.

```bash
uv run python -m eval.repoeval.compute_score --config configs/repoeval.yaml
```

### Pilot Experiment

Run the full pipeline end-to-end with a small number of queries to verify correctness:

```bash
sbatch scripts/repoeval_pilot.sh configs/repoeval_pilot.yaml
```

## License

[MIT](LICENSE)
