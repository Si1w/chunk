#!/usr/bin/env python3
"""
Chunking script for example source code.
Uses the different ChunkBuilder classes with max_chunk_size = 2048.
"""
import os
import argparse

from chunk import FunctionChunkBuilder, DeclarationChunkBuilder, SlidingChunkBuilder


METHODS = {
    "function": FunctionChunkBuilder,
    "declaration": DeclarationChunkBuilder,
    "sliding": SlidingChunkBuilder,
}


def main():
    parser = argparse.ArgumentParser(description="Chunking Example Script")
    parser.add_argument(
        "--method",
        choices=METHODS.keys(),
        default="function",
        help="Chunking method to use",
    )
    args = parser.parse_args()

    input_file = os.path.join(os.path.dirname(__file__), "source_code.txt")
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.method}_chunking_results.txt")

    with open(input_file, "r", encoding="utf-8") as f:
        code = f.read()

    configs = {
        "max_chunk_size": 2048,
        "language": "python",
        "metadata_template": "default",
        "overlap_lines": 5,
        "private_function": False,
        "function_overlap": False,
    }

    chunk_builder = METHODS[args.method](**configs)
    chunks = chunk_builder.chunkify(code)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            f"{args.method.capitalize()} Chunking Results "
            f"(max {configs['max_chunk_size']} non-whitespace chars per chunk)\n"
        )
        f.write("=" * 80 + "\n\n")

        for i, chunk in enumerate(chunks, 1):
            content = chunk["content"]
            metadata = chunk["metadata"]
            line_count = content.count("\n") + 1
            chunk_size = metadata.get("chunk_size", 0)

            header = f"{'-' * 25} Chunk {i} ({line_count} lines / {chunk_size} chars) {'-' * 25}"
            f.write(header + "\n")
            f.write(content + "\n")
            f.write(str(metadata) + "\n")
            f.write("-" * len(header) + "\n\n")

    print(f"{args.method.capitalize()} chunking completed!")
    print(f"Created {len(chunks)} chunks")
    print(f"Results written to: {output_file}")


if __name__ == "__main__":
    main()
