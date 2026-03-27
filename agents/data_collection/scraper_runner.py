from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any

from agents.data_collection.scraper_guard import ScraperGuardError, validate_scraper_source
from agents.data_collection.scraper_spec import ScraperSpec

if TYPE_CHECKING:
    import pandas as pd


class ScraperRunError(RuntimeError):
    """Raised when compile-time or run-time execution of generated scraper fails."""


_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "None",
        "True",
        "False",
        "int",
        "float",
        "str",
        "bytes",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "frozenset",
        "len",
        "range",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "enumerate",
        "zip",
        "map",
        "filter",
        "reversed",
        "sorted",
        "isinstance",
        "issubclass",
        "type",
        "object",
        "property",
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "AttributeError",
        "RuntimeError",
        "StopIteration",
        "IndexError",
        "OSError",
        "IOError",
        "ArithmeticError",
        "ZeroDivisionError",
        "all",
        "any",
        "ord",
        "chr",
        "bin",
        "hex",
        "oct",
        "slice",
        "hash",
        "id",
        "repr",
        "format",
        "iter",
        "next",
    }
)


def _safe_builtins_dict() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in _SAFE_BUILTIN_NAMES:
        if hasattr(builtins, name):
            out[name] = getattr(builtins, name)
    # Required for ``import`` statements inside ``exec``; explicit ``__import__(...)``
    # calls are still rejected by :func:`validate_scraper_source`.
    out["__import__"] = builtins.__import__
    return out


def run_generated_scraper(
    code: str,
    *,
    timeout: float = 20.0,
    skip_guard: bool = False,
) -> pd.DataFrame:
    """Compile *code*, ``exec`` into an isolated namespace, call ``run(timeout=...)``.

    Returns the :class:`pandas.DataFrame` from ``run``. This is an MVP helper, not a
    production sandbox (see module docstring in project docs / user-facing summary).
    """

    import pandas as pd

    if not skip_guard:
        try:
            validate_scraper_source(code)
        except ScraperGuardError as exc:
            raise ScraperRunError(str(exc)) from exc

    try:
        compiled = compile(code, "<generated_scraper>", "exec")
    except SyntaxError as exc:
        raise ScraperRunError(f"Compile failed: {exc}") from exc

    namespace: dict[str, Any] = {"__builtins__": _safe_builtins_dict()}
    try:
        exec(compiled, namespace, namespace)
    except Exception as exc:
        raise ScraperRunError(f"Module initialization failed: {exc!r}") from exc

    run_fn = namespace.get("run")
    if not callable(run_fn):
        raise ScraperRunError("Expected a callable top-level 'run' function.")

    try:
        result = run_fn(timeout=timeout)
    except Exception as exc:
        raise ScraperRunError(f"run() failed: {exc!r}") from exc

    if not isinstance(result, pd.DataFrame):
        raise ScraperRunError(
            f"run() must return pandas.DataFrame, got {type(result).__name__!r}"
        )
    return result


def run_scraper_from_spec(
    spec: ScraperSpec,
    *,
    timeout: float = 20.0,
    skip_guard: bool = False,
) -> pd.DataFrame:
    """Codegen + :func:`run_generated_scraper`."""

    from agents.data_collection.scraper_codegen import generate_scraper_code

    code = generate_scraper_code(spec)
    return run_generated_scraper(code, timeout=timeout, skip_guard=skip_guard)
