"""Per-source row budgets for ``max_merged_rows`` before collection (avoid full loads)."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace

from agents.data_collection.schemas import SourceSpec


def _stratum_key(spec: SourceSpec, stratify_column: str) -> str:
    col = (stratify_column or "source").strip() or "source"
    if col == "source_id":
        return spec.id
    if col == "source_type":
        return spec.type.value if hasattr(spec.type, "value") else str(spec.type)
    if col == "source":
        return spec.name or spec.id
    return spec.id


def _per_spec_weight(spec: SourceSpec) -> float:
    if spec.estimated_rows is not None and int(spec.estimated_rows) > 0:
        return float(spec.estimated_rows)
    return 1.0


def _largest_remainder_alloc(keys: list[str], weights: dict[str, float], k: int) -> dict[str, int]:
    if k <= 0 or not keys:
        return {key: 0 for key in keys}
    total_w = sum(max(0.0, weights.get(key, 0.0)) for key in keys)
    if total_w <= 0:
        base = k // len(keys)
        out = {key: base for key in keys}
        rem = k - base * len(keys)
        for i, key in enumerate(keys):
            if i < rem:
                out[key] += 1
        return out
    raw = {key: k * max(0.0, weights[key]) / total_w for key in keys}
    floors = {key: int(math.floor(raw[key])) for key in keys}
    deficit = k - sum(floors.values())
    frac_order = sorted(((raw[key] - floors[key], key) for key in keys), key=lambda t: t[0], reverse=True)
    for _frac, key in frac_order:
        if deficit <= 0:
            break
        floors[key] += 1
        deficit -= 1
    return floors


def allocate_row_budgets(
    sources: list[SourceSpec],
    max_rows: int,
    stratify_column: str,
) -> dict[str, int]:
    """Split *max_rows* across sources (stratified by config column), min 1 row each when possible."""

    n = len(sources)
    if n == 0:
        return {}
    if max_rows <= 0:
        return {s.id: 0 for s in sources}

    if max_rows < n:
        order = sorted(range(n), key=lambda i: _per_spec_weight(sources[i]), reverse=True)
        out = {s.id: 0 for s in sources}
        for i in range(max_rows):
            out[sources[order[i]].id] = 1
        return out

    remaining = max_rows - n
    base: dict[str, int] = {s.id: 1 for s in sources}
    if remaining == 0:
        return base

    groups: dict[str, list[SourceSpec]] = defaultdict(list)
    for s in sources:
        groups[_stratum_key(s, stratify_column)].append(s)

    stratum_keys = list(groups.keys())
    stratum_weights = {k: sum(_per_spec_weight(s) for s in groups[k]) for k in stratum_keys}
    extra_by_stratum = _largest_remainder_alloc(stratum_keys, stratum_weights, remaining)

    for k in stratum_keys:
        specs = groups[k]
        extra = extra_by_stratum[k]
        if not specs:
            continue
        if len(specs) == 1:
            base[specs[0].id] += extra
            continue
        wmap = {s.id: _per_spec_weight(s) for s in specs}
        inner = _largest_remainder_alloc(list(wmap.keys()), wmap, extra)
        for s in specs:
            base[s.id] += inner[s.id]

    return base


def apply_budget_to_sample_size(spec: SourceSpec, budget_rows: int | None) -> SourceSpec:
    """Combine pipeline budget with optional user ``sample_size`` (tighter cap wins)."""

    if budget_rows is None:
        return spec
    if budget_rows <= 0:
        return replace(spec, sample_size=0)
    eff = budget_rows
    if spec.sample_size is not None:
        eff = min(eff, int(spec.sample_size))
    return replace(spec, sample_size=eff)
