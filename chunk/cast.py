from astchunk import ASTChunkBuilder as _ASTChunkBuilder

from .utils import build_metadata


class CASTChunkBuilder:
    """Thin wrapper around astchunk's ASTChunkBuilder (original CAST method).

    Uses astchunk for AST chunking logic but builds metadata with the
    project's own build_metadata to support all template names.
    """

    def __init__(self, **configs):
        self.metadata_template = configs.get("metadata_template", "default")
        # astchunk needs a valid template for internal use; actual metadata
        # is rebuilt below so we pass "none" to skip astchunk's own metadata.
        self._builder = _ASTChunkBuilder(
            max_chunk_size=configs["max_chunk_size"],
            language=configs["language"],
            metadata_template="none",
        )

    def chunkify(self, code: str, **configs) -> list[dict]:
        repo_level_metadata = configs.get("repo_level_metadata", {})
        chunk_expansion = configs.get("chunk_expansion", False)

        ast = self._builder.parser.parse(bytes(code, "utf8"))
        ast_windows = list(self._builder.assign_tree_to_windows(code, ast.root_node))
        ast_windows = self._builder.add_window_overlapping(
            ast_windows, configs.get("chunk_overlap", 0),
        )
        ast_chunks = self._builder.convert_windows_to_chunks(
            ast_windows, repo_level_metadata, chunk_expansion,
        )

        # Rebuild metadata using the project's own templates.
        for chunk in ast_chunks:
            chunk.metadata = build_metadata(
                metadata_template=self.metadata_template,
                repo_level_metadata=repo_level_metadata,
                chunk_size=chunk.chunk_size,
                length=chunk.length,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )

        return self._builder.convert_chunks_to_code_windows(ast_chunks)
