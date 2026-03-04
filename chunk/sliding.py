from .utils import build_metadata
from astchunk.preprocessing import ByteRange, preprocess_nws_count, get_nws_count


class SlidingChunkBuilder:
    """Line-based sliding window chunker with optional overlap."""

    def __init__(self, **configs):
        self.max_chunk_size: int = configs.get("max_chunk_size", 512)
        self.overlap_lines: int = configs.get("overlap_lines", 0)
        self.metadata_template: str = configs.get("metadata_template", "default")

    def chunkify(self, code: str, repo_level_metadata: dict | None = None) -> list[dict]:
        """Split code into fixed-size overlapping chunks at line boundaries.

        Args:
            code: Source code string.
            repo_level_metadata: Optional metadata dict for the repository.

        Returns:
            List of dicts with 'content' and 'metadata' keys.
        """
        if not code:
            return []

        if repo_level_metadata is None:
            repo_level_metadata = {}

        code_bytes = code.encode("utf8")
        nws_cumsum = preprocess_nws_count(code_bytes)

        line_starts = [0]
        line_starts.extend([i + 1 for i, b in enumerate(code_bytes) if b == 10])

        line_boundaries = []
        for i in range(len(line_starts)):
            start = line_starts[i]
            end = line_starts[i + 1] if i < len(line_starts) - 1 else len(code_bytes)
            if start <= end:
                line_boundaries.append((start, end))

        total_lines = len(line_boundaries)
        chunks = []
        current_line = 0

        while current_line < total_lines:
            start_line = current_line
            start_byte = line_boundaries[start_line][0]
            end_line = start_line

            for line_idx in range(start_line, total_lines):
                curr_end = line_boundaries[line_idx][1]
                chunk_nws = get_nws_count(nws_cumsum, ByteRange(start_byte, curr_end))

                if chunk_nws > self.max_chunk_size and line_idx > start_line:
                    break
                end_line = line_idx

            end_byte = line_boundaries[end_line][1]
            chunk_nws = get_nws_count(nws_cumsum, ByteRange(start_byte, end_byte))

            if chunk_nws == 0 and end_line == total_lines - 1:
                break

            chunk_text = code_bytes[start_byte:end_byte].decode("utf8")
            if not chunk_text:
                current_line += 1
                continue

            chunks.append({
                "content": chunk_text,
                "metadata": build_metadata(
                    metadata_template=self.metadata_template,
                    repo_level_metadata=repo_level_metadata,
                    chunk_size=chunk_nws,
                    length=(end_line - start_line) + 1,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                ),
            })

            window_len = (end_line - start_line) + 1
            step_size = max(1, window_len - self.overlap_lines)
            current_line += step_size

            if end_line >= total_lines - 1:
                break

        return chunks
