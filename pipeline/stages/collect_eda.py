"""Best-effort EDA for the collect stage: markdown summary + matplotlib plots on disk."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

# Light English stopword filter for unigram frequency (no extra dependency).
_STOPWORDS = frozenset(
    "the a an and or to of in is for on with as at by from that this it be are was were not no yes "
    "if we you he she they i my our their one two all can has have had will would could should".split()
)

_CORE_FOR_PROFILE = ("text", "audio", "image", "label", "source", "source_type", "source_url", "source_id", "collected_at", "record_hash")


def _skipped(md_lines: List[str], plots_note: List[str], message: str) -> None:
    md_lines.append(f"- **Skipped / N/A:** {message}")
    plots_note.append(message)


def _nonempty_text_mask(s: pd.Series) -> pd.Series:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str).str.lower() != "nan")
    return s.notna()


def _label_usable(s: pd.Series) -> bool:
    if s.empty or s.isna().all():
        return False
    sn = s.dropna().astype(str).str.strip()
    return bool(((sn != "") & (sn.str.lower() != "nan")).any())


def _top_words_from_series(s: pd.Series, *, top_n: int = 20) -> List[Tuple[str, int]]:
    token_re = re.compile(r"[a-zA-Z][a-zA-Z']{2,}")
    counts: Counter[str] = Counter()
    for val in s.dropna().astype(str).head(50_000):
        for m in token_re.findall(val.lower()):
            if m in _STOPWORDS:
                continue
            counts[m] += 1
    return counts.most_common(top_n)


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    num: List[str] = []
    for c in df.columns:
        if c in _CORE_FOR_PROFILE:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(str(c))
    return num


def _categorical_candidates(df: pd.DataFrame, max_card: int = 50) -> List[str]:
    cats: List[str] = []
    for c in df.columns:
        if c in _CORE_FOR_PROFILE:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        try:
            nunique = df[c].nunique(dropna=True)
        except Exception:
            continue
        if 2 <= nunique <= max_card:
            cats.append(str(c))
    cats.sort(key=lambda x: df[x].nunique(dropna=True))
    return cats[:12]


def _detect_year_columns(df: pd.DataFrame) -> List[str]:
    found: List[str] = []
    for c in df.columns:
        if c in _CORE_FOR_PROFILE:
            continue
        col = df[c].dropna().head(200)
        if col.empty:
            continue
        try:
            parsed = pd.to_datetime(col, errors="coerce", utc=False)
        except Exception:
            continue
        if parsed.notna().mean() >= 0.5:
            found.append(str(c))
    return found[:5]


_TENNIS_HINTS = (
    "surface",
    "tourney_name",
    "tourney_level",
    "winner_name",
    "loser_name",
    "winner_id",
    "loser_id",
    "score",
    "round",
)


def _tennis_subset_summary(df: pd.DataFrame, md_lines: List[str]) -> None:
    lower = {str(c).lower(): c for c in df.columns}
    hits = [h for h in _TENNIS_HINTS if h in lower]
    if not hits:
        return
    md_lines.append("### Domain heuristics (sport / match tables)")
    for h in hits:
        col = lower[h]
        try:
            vc = df[col].value_counts(dropna=True).head(8)
            md_lines.append(f"- `{col}` value counts (top 8): {vc.to_dict()}")
        except Exception:
            md_lines.append(f"- `{col}`: present; summary failed.")


def generate_collect_eda(df: pd.DataFrame, reports_dir: Path) -> Dict[str, Any]:
    """Write ``eda_summary.md`` and plots under ``reports_dir/plots``. Return paths and skip notes."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    md_lines: List[str] = ["# Collect stage — exploratory data analysis\n"]
    plot_paths: List[str] = []
    plots_log: List[str] = []

    n_rows, n_cols = len(df), len(df.columns)
    md_lines.append("## Dataset shape")
    md_lines.append(f"- **Rows:** {n_rows}")
    md_lines.append(f"- **Columns:** {n_cols}")

    if "source_id" in df.columns:
        vc = df["source_id"].value_counts(dropna=True)
        md_lines.append("### Rows per `source_id`")
        md_lines.append("```")
        md_lines.append(vc.to_string())
        md_lines.append("```")
    elif "source" in df.columns:
        vc = df["source"].value_counts(dropna=True)
        md_lines.append("### Rows per `source`")
        md_lines.append("```")
        md_lines.append(vc.to_string())
        md_lines.append("```")

    md_lines.append(
        "- **Full-row duplicates:** not computed in collect — run the **quality** stage; "
        "use ``DataQualityAgent.detect_issues(df)`` (see `duplicates` in the quality report)."
    )

    md_lines.append("\n## Column list")
    md_lines.append(", ".join(f"`{c}`" for c in df.columns))

    md_lines.append("\n## Missingness (share NA per column)")
    if n_rows:
        miss = df.isna().mean().sort_values(ascending=False).head(40)
        md_lines.append("| column | missing_share |")
        md_lines.append("|--------|---------------|")
        for col, v in miss.items():
            md_lines.append(f"| `{col}` | {v:.4f} |")
    else:
        md_lines.append("(empty frame)")

    md_lines.append("\n## Core unified schema — non-null counts")
    for c in _CORE_FOR_PROFILE:
        if c in df.columns:
            nn = int(df[c].notna().sum())
            md_lines.append(f"- `{c}`: {nn} non-null")
        else:
            md_lines.append(f"- `{c}`: **missing column**")
            _skipped(md_lines, plots_log, f"Core column `{c}` absent on merged frame.")

    # Plots: per-source counts (preferred default)
    group_col = "source_id" if "source_id" in df.columns else ("source" if "source" in df.columns else None)
    if group_col and df[group_col].nunique(dropna=False) > 0:
        try:
            vc = df[group_col].value_counts()
            fig, ax = plt.subplots(figsize=(max(6, len(vc) * 0.4), 4))
            vc.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("Row count by source")
            ax.set_xlabel(group_col)
            fig.tight_layout()
            pth = plots_dir / "source_row_counts.png"
            fig.savefig(pth, dpi=120)
            plt.close(fig)
            plot_paths.append(str(pth.resolve()))
            md_lines.append(f"\n![source counts](plots/{pth.name})")
        except Exception as exc:
            _skipped(md_lines, plots_log, f"source row count plot failed: {exc}")
    else:
        _skipped(md_lines, plots_log, "No `source` / `source_id` column for bar plot of row counts.")

    # Label distribution
    if "label" in df.columns and _label_usable(df["label"]):
        try:
            vc = df["label"].astype(str).value_counts().head(30)
            fig, ax = plt.subplots(figsize=(8, 4))
            vc.plot(kind="bar", ax=ax, color="darkseagreen")
            ax.set_title("Label distribution (top 30)")
            fig.tight_layout()
            pth = plots_dir / "label_distribution.png"
            fig.savefig(pth, dpi=120)
            plt.close(fig)
            plot_paths.append(str(pth.resolve()))
            md_lines.append(f"\n![label distribution](plots/{pth.name})")
            md_lines.append("### Label frequencies (top 30)")
            md_lines.append("```")
            md_lines.append(vc.to_string())
            md_lines.append("```")
        except Exception as exc:
            _skipped(md_lines, plots_log, f"label distribution: {exc}")
    else:
        _skipped(md_lines, plots_log, "**Label EDA skipped:** `label` column missing or all empty.")

    # Text length + top words
    if "text" in df.columns and _nonempty_text_mask(df["text"]).any():
        try:
            lens = df.loc[_nonempty_text_mask(df["text"]), "text"].astype(str).str.len()
            md_lines.append("\n## Text column")
            md_lines.append(f"- Non-empty rows: {len(lens)}")
            md_lines.append(f"- Length min / median / max: {lens.min()} / {lens.median():.0f} / {lens.max()}")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(lens.clip(upper=lens.quantile(0.99)), bins=40, color="coral", edgecolor="white")
            ax.set_title("Text length distribution (clipped at 99p)")
            fig.tight_layout()
            pth = plots_dir / "text_length_hist.png"
            fig.savefig(pth, dpi=120)
            plt.close(fig)
            plot_paths.append(str(pth.resolve()))
            md_lines.append(f"\n![text lengths](plots/{pth.name})")
            top_w = _top_words_from_series(df["text"])
            md_lines.append("### Top tokens (unigrams, naive, English stopwords trimmed)")
            md_lines.append("```")
            for w, cnt in top_w:
                md_lines.append(f"  {w}: {cnt}")
            md_lines.append("```")
        except Exception as exc:
            _skipped(md_lines, plots_log, f"text EDA: {exc}")
    else:
        _skipped(md_lines, plots_log, "**Text EDA skipped:** `text` column missing or all empty.")

    # Tabular
    md_lines.append("\n## Tabular overview (non-text primary path)")
    nums = _numeric_columns(df)
    if nums:
        md_lines.append("### Numeric columns (describe)")
        try:
            desc = df[nums[:15]].describe().T
            md_lines.append("```")
            md_lines.append(desc.to_string())
            md_lines.append("```")
        except Exception as exc:
            md_lines.append(f"(describe failed: {exc})")
    cats = _categorical_candidates(df)
    if cats:
        md_lines.append("### Low-cardinality categoricals (top values)")
        for c in cats[:6]:
            try:
                vc = df[c].value_counts(dropna=True).head(10)
                md_lines.append(f"- `{c}`: {vc.to_dict()}")
            except Exception:
                pass

    for ycol in _detect_year_columns(df):
        try:
            years = pd.to_datetime(df[ycol], errors="coerce").dt.year.dropna()
            if years.empty:
                continue
            vc = years.astype(int).value_counts().sort_index()
            md_lines.append(f"\n### Year coverage (from `{ycol}`)")
            md_lines.append("```")
            md_lines.append(vc.to_string())
            md_lines.append("```")
        except Exception:
            continue

    _tennis_subset_summary(df, md_lines)

    md_lines.append("\n## Plot generation notes")
    for line in plots_log:
        md_lines.append(f"- {line}")

    if not plot_paths:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No standard plots generated;\nsee Skipped / N/A above.", ha="center", va="center")
        ax.axis("off")
        pth = plots_dir / "eda_notice.png"
        fig.savefig(pth, dpi=100)
        plt.close(fig)
        plot_paths.append(str(pth.resolve()))
        md_lines.append(f"\n![notice](plots/{pth.name})")

    eda_path = reports_dir / "eda_summary.md"
    eda_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "eda_summary_path": str(eda_path.resolve()),
        "plot_paths": plot_paths,
        "skipped_notes": plots_log,
    }
