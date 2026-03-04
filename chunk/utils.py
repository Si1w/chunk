import tree_sitter as ts


class FUNCTION_TYPES:
    PYTHON = ["function_definition"]
    JAVA = ["method_declaration", "function_declaration"]
    CSHARP = ["method_declaration", "function_declaration"]
    TYPESCRIPT = ["function_declaration", "method_definition", "arrow_function"]


class CLASS_TYPES:
    PYTHON = ["class_definition"]
    JAVA = ["class_declaration", "constructor_declaration"]
    CSHARP = ["class_declaration", "constructor_declaration"]
    TYPESCRIPT = ["class_declaration"]


def is_private_function(node: ts.Node, language: str) -> bool:
    """Check if a function/method node is private based on language-specific rules."""
    if language == "python":
        for child in node.children:
            if child.type == "identifier":
                if child.text.startswith(b"_"):
                    return True
        return False

    if language in ["java", "csharp"]:
        for child in node.children:
            if child.type == "modifiers":
                for modifier in child.children:
                    if modifier.text == b"private":
                        return True
        return False

    if language == "typescript":
        for child in node.children:
            if child.type == "accessibility_modifier":
                if child.text == b"private":
                    return True
            if child.type == "private_identifier":
                return True
        return False

    return False


def build_metadata(
    metadata_template: str,
    repo_level_metadata: dict,
    chunk_size: int | None = None,
    length: int | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    """Build chunk metadata based on the specified template."""
    if metadata_template == "none":
        return {}

    if metadata_template == "default":
        return {
            "filepath": repo_level_metadata.get("filepath", ""),
            "chunk_size": chunk_size,
            "line_count": length,
            "start_line_no": start_line,
            "end_line_no": end_line,
        }

    if metadata_template == "repoeval":
        return {
            "fpath_tuple": repo_level_metadata.get("fpath_tuple", []),
            "repo": repo_level_metadata.get("repo", ""),
            "chunk_size": chunk_size,
            "line_count": length,
            "start_line_no": start_line,
            "end_line_no": end_line,
        }

    if metadata_template == "cceval":
        return {
            "task_id": repo_level_metadata.get("task_id", ""),
            "repository": repo_level_metadata.get("repository", ""),
            "file": repo_level_metadata.get("file", ""),
            "chunk_size": chunk_size,
            "line_count": length,
            "start_line_no": start_line,
            "end_line_no": end_line,
        }

    raise ValueError(f"Unsupported metadata template: {metadata_template}")
