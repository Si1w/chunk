from abc import ABC, abstractmethod
from typing import Generator

import numpy as np
import pyrsistent
import tree_sitter as ts
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_c_sharp as tscsharp
import tree_sitter_typescript as tstypescript

from .utils import CLASS_TYPES, FUNCTION_TYPES
from astchunk.astnode import ASTNode
from astchunk.astchunk import ASTChunk
from astchunk.preprocessing import ByteRange, preprocess_nws_count, get_nws_count

LANGUAGE_CONFIG = {
    "python": {
        "parser_factory": lambda: ts.Parser(ts.Language(tspython.language())),
        "function_types": FUNCTION_TYPES.PYTHON,
        "class_types": CLASS_TYPES.PYTHON,
    },
    "java": {
        "parser_factory": lambda: ts.Parser(ts.Language(tsjava.language())),
        "function_types": FUNCTION_TYPES.JAVA,
        "class_types": CLASS_TYPES.JAVA,
    },
    "csharp": {
        "parser_factory": lambda: ts.Parser(ts.Language(tscsharp.language())),
        "function_types": FUNCTION_TYPES.CSHARP,
        "class_types": CLASS_TYPES.CSHARP,
    },
    "typescript": {
        "parser_factory": lambda: ts.Parser(ts.Language(tstypescript.language_tsx())),
        "function_types": FUNCTION_TYPES.TYPESCRIPT,
        "class_types": CLASS_TYPES.TYPESCRIPT,
    },
}


class ASTChunkBuilder(ABC):
    """Base class for AST-based code chunkers."""

    def __init__(self, **configs):
        self.max_chunk_size: int = configs.get("max_chunk_size", 0)
        self.language: str = configs["language"]
        self.metadata_template: str = configs["metadata_template"]
        self.private_function: bool = configs.get("private_function", False)

        if self.language not in LANGUAGE_CONFIG:
            raise ValueError(f"Unsupported language: {self.language}")

        lang_cfg = LANGUAGE_CONFIG[self.language]
        self.parser = lang_cfg["parser_factory"]()
        self.function_types = lang_cfg["function_types"]
        self.class_types = lang_cfg["class_types"]

    @abstractmethod
    def assign_tree_to_windows(
        self, code: str, root_node: ts.Node
    ) -> Generator[list[ASTNode], None, None]:
        """Assign AST tree nodes into windows (tentative chunks)."""
        ...

    def assign_nodes_to_windows(
        self,
        nodes: list[ts.Node],
        nws_cumsum: np.ndarray,
        ancestors: pyrsistent.PVector,
    ) -> Generator[list[ASTNode], None, None]:
        """Greedily assign AST nodes to windows based on non-whitespace char count.

        Recursively splits nodes that exceed max_chunk_size.
        """
        current_window = []
        current_window_size = 0

        for node in nodes:
            node_range = ByteRange(node.start_byte, node.end_byte)
            node_size = get_nws_count(nws_cumsum, node_range)

            if node_size > self.max_chunk_size:
                childs_ancestors = ancestors.append(node)
                child_windows = list(
                    self.assign_nodes_to_windows(
                        node.children, nws_cumsum, childs_ancestors
                    )
                )

                if current_window:
                    if child_windows:
                        first_window_size = sum(n.size for n in child_windows[0])
                        if (
                            current_window_size + first_window_size
                            <= self.max_chunk_size
                        ):
                            child_windows[0] = current_window + child_windows[0]
                        else:
                            yield current_window
                    else:
                        yield current_window

                    current_window = []
                    current_window_size = 0

                if child_windows:
                    yield from self._merge_adjacent_windows(child_windows)

                continue

            if current_window_size + node_size > self.max_chunk_size:
                if current_window:
                    yield current_window
                current_window = []
                current_window_size = 0

            current_window.append(ASTNode(node, node_size, ancestors))
            current_window_size += node_size

        if current_window:
            yield current_window

    def _merge_adjacent_windows(
        self, ast_windows: list[list[ASTNode]]
    ) -> Generator[list[ASTNode], None, None]:
        """Greedily merge adjacent sibling windows that fit within max_chunk_size."""
        assert ast_windows, "Expect non-empty ast_windows"

        merged = [ast_windows[0][:]]

        for window in ast_windows[1:]:
            merged_size = sum(n.size for n in merged[-1]) + sum(
                n.size for n in window
            )
            if merged_size <= self.max_chunk_size:
                merged[-1].extend(window)
            else:
                merged.append(window[:])

        yield from merged

    def _convert_windows_to_chunks(
        self,
        ast_windows: list[list[ASTNode]],
        repo_level_metadata: dict,
        chunk_expansion: bool,
    ) -> list[ASTChunk]:
        """Convert ASTNode windows into ASTChunk objects with metadata."""
        chunks = list[ASTChunk]()

        for window in ast_windows:
            chunk = ASTChunk(
                ast_window=window,
                max_chunk_size=self.max_chunk_size,
                language=self.language,
                metadata_template=self.metadata_template,
            )
            chunk.build_metadata(repo_level_metadata)
            if chunk_expansion:
                chunk.apply_chunk_expansion()
            chunks.append(chunk)

        return chunks

    def _convert_chunks_to_code_windows(self, ast_chunks: list[ASTChunk]) -> list[dict]:
        """Convert ASTChunk objects into output dicts with 'content' and 'metadata'."""
        return [chunk.to_code_window() for chunk in ast_chunks]

    def chunkify(self, code: str, **configs) -> list[dict]:
        """Parse code into structure-aware chunks using AST.

        Args:
            code: Source code string.
            **configs: Optional keys: repo_level_metadata (dict), chunk_expansion (bool).

        Returns:
            List of dicts with 'content' and 'metadata' keys.
        """
        tree = self.parser.parse(bytes(code, "utf8"))
        ast_windows = list(self.assign_tree_to_windows(code, tree.root_node))

        ast_chunks = self._convert_windows_to_chunks(
            ast_windows,
            repo_level_metadata=configs.get("repo_level_metadata", {}),
            chunk_expansion=configs.get("chunk_expansion", False),
        )

        return self._convert_chunks_to_code_windows(ast_chunks)
