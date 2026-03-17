"""AST-based callable source extraction for scenario serialization.

Extracts Python source text from callables (lambdas, functions) using
inspect.getsource() and AST parsing. Falls back to requiring string
form if extraction fails.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap

logger = logging.getLogger(__name__)


def extract_callable_source(obj: object) -> str | None:
    """Extract Python source from a callable, string, or None.

    Args:
        obj: A callable (lambda or function), a source string, or None.

    Returns:
        Source string, or None if obj is None.

    Raises:
        ValueError: If obj is a callable whose source cannot be extracted.
            The error message tells the user to pass a string form instead.
    """
    if obj is None:
        return None

    if isinstance(obj, str):
        return obj

    if not callable(obj):
        raise TypeError(f"Expected a callable, string, or None; got {type(obj).__name__}")

    try:
        raw_source = inspect.getsource(obj)
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"Cannot extract source from {obj!r}. Pass the source string instead "
            "(e.g., state_update='lambda acc, p: ...')."
        ) from exc

    # Dedent so AST parsing works on indented source
    dedented = textwrap.dedent(raw_source)

    try:
        tree = ast.parse(dedented)
    except SyntaxError as exc:
        raise ValueError(
            f"Cannot extract source from {obj!r}: syntax error in extracted text. Pass the source string instead."
        ) from exc

    # Walk AST to find lambda or function def
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            lambda_source = _extract_lambda_text(dedented, node)
            if lambda_source:
                return lambda_source

    # For named functions, return the full dedented source
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return dedented.strip()

    # Fallback: return the full dedented source
    return dedented.strip()


def _extract_lambda_text(source: str, node: ast.Lambda) -> str | None:
    """Extract lambda expression text from source using AST node position."""
    lines = source.splitlines()
    if not lines:
        return None

    start_line = node.lineno - 1  # 0-indexed
    start_col = node.col_offset

    if start_line >= len(lines):
        return None

    line = lines[start_line]
    lambda_text = line[start_col:]

    # Strip trailing syntax from enclosing expression (parens, commas)
    depth = 0
    end = len(lambda_text)
    for i, ch in enumerate(lambda_text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                end = i
                break
            depth -= 1
        elif ch == "," and depth == 0:
            end = i
            break

    return lambda_text[:end].rstrip()
