from typing import Generator

import pyrsistent
import tree_sitter as ts

from .base import ASTChunkBuilder
from .utils import is_private_function
from astchunk.astnode import ASTNode
from astchunk.preprocessing import ByteRange, preprocess_nws_count, get_nws_count


class FunctionChunkBuilder(ASTChunkBuilder):
    """AST chunker that extracts function/method nodes."""

    def _traverse_function(
        self, root_node: ts.Node, ancestors: pyrsistent.PVector
    ) -> list[tuple[ts.Node, pyrsistent.PVector]]:
        """Extract function/method nodes with their ancestors."""
        nodes = []

        def traverse(node: ts.Node, current_ancestors: pyrsistent.PVector):
            if node.type in self.function_types:
                if self.private_function or not is_private_function(node, self.language):
                    nodes.append((node, current_ancestors))
                return
            new_ancestors = current_ancestors.append(node)
            for child in node.children:
                traverse(child, new_ancestors)

        for child in root_node.children:
            traverse(child, ancestors)

        return nodes

    def assign_tree_to_windows(
        self, code: str, root_node: ts.Node
    ) -> Generator[list[ASTNode], None, None]:
        """Assign function nodes to windows, splitting oversized nodes recursively."""
        nws_cumsum = preprocess_nws_count(bytes(code, "utf8"))
        ancestors = pyrsistent.v(root_node)
        func_nodes = self._traverse_function(root_node, ancestors)

        for func_node, func_ancestors in func_nodes:
            node_range = ByteRange(func_node.start_byte, func_node.end_byte)
            node_size = get_nws_count(nws_cumsum, node_range)

            if self.max_chunk_size <= 0 or node_size <= self.max_chunk_size:
                yield [ASTNode(func_node, node_size, func_ancestors)]
            else:
                yield from self.assign_nodes_to_windows(
                    nodes=func_node.children,
                    nws_cumsum=nws_cumsum,
                    ancestors=func_ancestors.append(func_node),
                )
