from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable


class ScraperGuardError(ValueError):
    """Raised when generated scraper source fails static validation."""


@dataclass(frozen=True)
class ScraperGuardConfig:
    """MVP static checks before ``exec``."""

    allowed_import_modules: frozenset[str] = frozenset(
        {
            "pandas",
            "requests_html",
            "urllib.parse",
            "re",
        }
    )


# Calls to these bare names are rejected (common escape hatches).
_BANNED_CALL_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "dir",
        "memoryview",
    }
)

# Attribute calls like os.system, subprocess.run
_BANNED_ATTRIBUTE_CALLS: frozenset[tuple[str | None, str]] = frozenset(
    {
        ("os", "system"),
        ("os", "popen"),
        ("subprocess", "call"),
        ("subprocess", "run"),
        ("subprocess", "Popen"),
        ("sys", "exit"),
        ("builtins", "open"),
        ("builtins", "eval"),
        ("builtins", "exec"),
    }
)


def validate_scraper_source(source: str, *, config: ScraperGuardConfig | None = None) -> None:
    """Parse *source* and apply MVP import/call rules.

    This is not a security boundary; it catches obvious mistakes and trivial abuse.
    """

    cfg = config or ScraperGuardConfig()
    try:
        tree = ast.parse(source, filename="<scraper>", mode="exec")
    except SyntaxError as exc:
        raise ScraperGuardError(f"Invalid Python syntax: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            _check_import(node, cfg)
        elif isinstance(node, ast.ImportFrom):
            _check_import_from(node, cfg)
        elif isinstance(node, ast.Call):
            _check_call(node)
        elif isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith)):
            raise ScraperGuardError("Async syntax is not allowed in scraper code.")
        elif isinstance(node, ast.ClassDef):
            raise ScraperGuardError("Class definitions are not allowed in scraper code.")

    _assert_run_function_present(tree)


def _check_import(node: ast.Import, cfg: ScraperGuardConfig) -> None:
    for alias in node.names:
        base = alias.name.split(".")[0]
        if alias.name not in cfg.allowed_import_modules and base not in cfg.allowed_import_modules:
            raise ScraperGuardError(f"Import of module '{alias.name}' is not allowed.")
        if alias.name == "urllib":
            raise ScraperGuardError("Import 'urllib' is not allowed; use 'from urllib.parse import ...'.")


def _check_import_from(node: ast.ImportFrom, cfg: ScraperGuardConfig) -> None:
    if node.level and node.level > 0:
        raise ScraperGuardError("Relative imports are not allowed.")
    module = node.module or ""
    if module not in cfg.allowed_import_modules:
        raise ScraperGuardError(f"Import from '{module}' is not allowed.")
    if module in ("pandas", "re"):
        raise ScraperGuardError(f"Use 'import {module}', not from-import from '{module}'.")
    for alias in node.names:
        if alias.name == "*":
            raise ScraperGuardError("Star-imports are not allowed.")
        if module == "urllib.parse" and alias.name not in ("urljoin", "urlparse"):
            raise ScraperGuardError(
                f"Only urljoin and urlparse may be imported from urllib.parse, not '{alias.name}'."
            )
        if module == "requests_html" and alias.name != "HTMLSession":
            raise ScraperGuardError(
                f"Only HTMLSession may be imported from requests_html, not '{alias.name}'."
            )


def _resolve_attribute_chain(node: ast.expr) -> tuple[str | None, str] | None:
    """Return (root_name, final_attr) for a.b.c style attribute access."""

    if isinstance(node, ast.Attribute):
        inner = _resolve_attribute_chain(node.value)
        if inner is None:
            return None
        root, _ = inner
        return (root, node.attr)
    if isinstance(node, ast.Name):
        return (node.id, node.id)
    return None


def _check_call(node: ast.Call) -> None:
    func = node.func
    if isinstance(func, ast.Name):
        if func.id in _BANNED_CALL_NAMES:
            raise ScraperGuardError(f"Call to built-in '{func.id}' is not allowed.")
        return
    if isinstance(func, ast.Attribute):
        chain = _resolve_attribute_chain(func)
        if chain is None:
            return
        root, final = chain
        if (root, final) in _BANNED_ATTRIBUTE_CALLS:
            raise ScraperGuardError(f"Call '{root}.{final}(...)' is not allowed.")
        # Heuristic: block .system / .popen on any object
        if final in {"system", "popen", "spawn"}:
            raise ScraperGuardError(f"Call to '.{final}(...)' is not allowed.")


def _assert_run_function_present(tree: ast.Module) -> None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return
    raise ScraperGuardError("Scraper code must define a top-level function 'run'.")


def iter_top_level_imports(source: str) -> Iterable[str]:
    """Debug helper: yield import lines as strings (best-effort)."""

    tree = ast.parse(source, mode="exec")
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                yield f"import {a.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                yield f"from {mod} import {a.name}"
