from typing import Generator

import pyrsistent
import tree_sitter as ts

from .base import ASTChunkBuilder
from .utils import is_private_function
from astchunk.astnode import ASTNode
from astchunk.preprocessing import ByteRange, preprocess_nws_count, get_nws_count


class DeclarationChunkBuilder(ASTChunkBuilder):
    """AST chunker that extracts class and function declarations."""

    def __init__(self, **configs):
        super().__init__(**configs)
        self.function_overlap: bool = configs.get("function_overlap", False)

    def _traverse_declaration(
        self, root_node: ts.Node, ancestors: pyrsistent.PVector
    ) -> list[tuple[ts.Node, pyrsistent.PVector]]:
        """Extract declaration nodes (classes and functions) with their ancestors."""
        nodes = []

        def traverse(node: ts.Node, current_ancestors: pyrsistent.PVector):
            if node.type in self.class_types:
                nodes.append((node, current_ancestors))
                if not self.function_overlap:
                    return

            elif node.type in self.function_types:
                if self.private_function or not is_private_function(node, self.language):
                    nodes.append((node, current_ancestors))
                return

            for child in node.children:
                traverse(child, current_ancestors.append(node))

        traverse(root_node, ancestors)
        return nodes

    def assign_tree_to_windows(
        self, code: str, root_node: ts.Node
    ) -> Generator[list[ASTNode], None, None]:
        """Assign declaration nodes to windows, splitting oversized nodes recursively."""
        nws_cumsum = preprocess_nws_count(bytes(code, "utf8"))
        ancestors = pyrsistent.v(root_node)
        decl_nodes = self._traverse_declaration(root_node, ancestors)

        for decl_node, decl_ancestors in decl_nodes:
            node_range = ByteRange(decl_node.start_byte, decl_node.end_byte)
            node_size = get_nws_count(nws_cumsum, node_range)

            if self.max_chunk_size <= 0 or node_size <= self.max_chunk_size:
                yield [ASTNode(decl_node, node_size, decl_ancestors)]
            else:
                yield from self.assign_nodes_to_windows(
                    nodes=decl_node.children,
                    nws_cumsum=nws_cumsum,
                    ancestors=decl_ancestors.append(decl_node),
                )
