"""Evaluation utilities for CrossCodeEval.

Aligned with official cceval/scripts/eval_utils.py:
- Tree-sitter based first-statement extraction for Python
- Bracket/semicolon based truncation for Java/C#/TypeScript
- Identifier extraction and matching
- Comment removal
"""

import keyword
import re

import tree_sitter_python as tspython
import tree_sitter as ts

_PY_LANGUAGE = ts.Language(tspython.language())
_PY_PARSER = ts.Parser(_PY_LANGUAGE)

PYTHON_KEYWORDS = set(keyword.kwlist)
JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while",
}
CSHARP_KEYWORDS = {
    "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char",
    "checked", "class", "const", "continue", "decimal", "default", "delegate",
    "do", "double", "else", "enum", "event", "explicit", "extern", "false",
    "finally", "fixed", "float", "for", "foreach", "goto", "if", "implicit",
    "in", "int", "interface", "internal", "is", "lock", "long", "namespace",
    "new", "null", "object", "operator", "out", "override", "params", "private",
    "protected", "public", "readonly", "ref", "return", "sbyte", "sealed",
    "short", "sizeof", "stackalloc", "static", "string", "struct", "switch",
    "this", "throw", "true", "try", "typeof", "uint", "ulong", "unchecked",
    "unsafe", "ushort", "using", "virtual", "void", "volatile", "while",
}
TYPESCRIPT_KEYWORDS = {
    "break", "case", "catch", "class", "const", "continue", "debugger",
    "default", "delete", "do", "else", "enum", "export", "extends", "false",
    "finally", "for", "function", "if", "import", "in", "instanceof", "let",
    "new", "null", "return", "super", "switch", "this", "throw", "true", "try",
    "typeof", "var", "void", "while", "with", "yield", "async", "await",
    "of", "implements", "interface", "package", "private", "protected", "public",
    "static", "type", "from", "as", "any", "boolean", "number", "string",
    "symbol", "undefined", "never", "unknown",
}

LANGUAGE_KEYWORDS = {
    "python": PYTHON_KEYWORDS,
    "java": JAVA_KEYWORDS,
    "csharp": CSHARP_KEYWORDS,
    "typescript": TYPESCRIPT_KEYWORDS,
}

# Regex for identifiers
_IDENTIFIER_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
# Regex for string literals (single/double/triple quoted)
_STRING_RE = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')


def cal_edit_sim(pred, gt):
    """Compute edit similarity using fuzz.ratio (0-100 scale)."""
    from thefuzz import fuzz
    return fuzz.ratio(pred.strip(), gt.strip())


def remove_comments(code, lang="python"):
    """Remove line comments from code."""
    if lang == "python":
        # Remove # comments (but not inside strings)
        lines = []
        for line in code.splitlines():
            # Simple approach: strip trailing # comments
            stripped = line.split("#")[0].rstrip() if "#" in line else line
            lines.append(stripped)
        return "\n".join(lines)
    else:
        # Remove // comments for Java/C#/TypeScript
        lines = []
        for line in code.splitlines():
            stripped = line.split("//")[0].rstrip() if "//" in line else line
            lines.append(stripped)
        return "\n".join(lines)


def get_python_one_statement(prompt, completion):
    """Extract the first complete Python statement using tree-sitter.

    Aligned with official cceval get_python_one_statement.
    """
    # Try progressively longer prefixes until we get a valid parse
    lines = completion.splitlines()
    if not lines:
        return ""

    # Build the full code context for parsing
    for i in range(1, len(lines) + 1):
        candidate = "\n".join(lines[:i])
        code = prompt + candidate
        tree = _PY_PARSER.parse(bytes(code, "utf8"))
        root = tree.root_node

        # Find the last statement in the tree
        if root.children:
            last_node = root.children[-1]
            # If the last node has no errors and ends at or before our candidate
            if not last_node.has_error and last_node.end_point[0] < len(code.splitlines()):
                # Check if adding the next line would start a new statement
                if i < len(lines):
                    next_line = lines[i].strip()
                    # If next line is non-empty and not indented more (new statement),
                    # the current candidate is a complete statement
                    if next_line and not next_line.startswith(" ") and not next_line.startswith("\t"):
                        return candidate

    # Fallback: return first line
    if lines:
        # Return lines up to the first blank line or dedent
        result_lines = [lines[0]]
        if len(lines) > 1:
            base_indent = len(lines[0]) - len(lines[0].lstrip())
            for line in lines[1:]:
                if not line.strip():
                    break
                curr_indent = len(line) - len(line.lstrip())
                if curr_indent <= base_indent and not line.strip().startswith((".", ")", "]", "}")):
                    break
                result_lines.append(line)
        return "\n".join(result_lines)

    return completion


def get_bracket_lang_statement(completion):
    """Extract first complete statement for bracket languages (Java/C#/TypeScript).

    Aligned with official cceval: track bracket depth and semicolons.
    """
    lines = completion.splitlines()
    if not lines:
        return ""

    depth = 0
    result_lines = []
    for line in lines:
        result_lines.append(line)
        for ch in line:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

        stripped = line.strip()
        # Complete statement: semicolon at depth 0, or closing brace returning to depth 0
        if depth == 0 and (stripped.endswith(";") or stripped.endswith("}")):
            return "\n".join(result_lines)
        # Negative depth means we've exited the enclosing scope
        if depth < 0:
            return "\n".join(result_lines[:-1]) if len(result_lines) > 1 else ""

    return "\n".join(result_lines)


def postprocess_code_lines(prompt, completion, lang="python"):
    """Post-process generated code to extract the first complete statement."""
    if lang == "python":
        return get_python_one_statement(prompt, completion)
    else:
        return get_bracket_lang_statement(completion)


def extract_identifiers(code, lang="python"):
    """Extract identifiers from code, filtering out keywords and string literals.

    Returns a deduplicated set of identifiers.
    """
    # Remove string literals first
    code_no_strings = _STRING_RE.sub("", code)
    # Remove comments
    code_clean = remove_comments(code_no_strings, lang)
    # Find all identifiers
    all_ids = _IDENTIFIER_RE.findall(code_clean)
    # Filter out keywords
    keywords = LANGUAGE_KEYWORDS.get(lang, PYTHON_KEYWORDS)
    return set(id_ for id_ in all_ids if id_ not in keywords)


def is_identifier(token, lang="python"):
    """Check if a token is a valid identifier (not a keyword)."""
    if not _IDENTIFIER_RE.fullmatch(token):
        return False
    keywords = LANGUAGE_KEYWORDS.get(lang, PYTHON_KEYWORDS)
    return token not in keywords


def compute_id_match(pred_ids, gt_ids):
    """Compute identifier match metrics (TP, FP, FN) over deduplicated sets."""
    tp = len(pred_ids & gt_ids)
    fp = len(pred_ids - gt_ids)
    fn = len(gt_ids - pred_ids)
    return tp, fp, fn
