from __future__ import annotations

import pandas as pd


def prepare_review_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that require human review."""

    if "needs_review" not in df.columns:
        return df.iloc[0:0].copy()
    review_df = df.loc[df["needs_review"].fillna(False).astype(bool)].copy()
    if "confidence" in review_df.columns and "margin" in review_df.columns:
        review_df = review_df.sort_values(
            by=["confidence", "margin"],
            ascending=[True, True],
        )
    return review_df


def prepare_audit_sample(
    df: pd.DataFrame,
    n_per_class: int = 25,
    label_column: str = "auto_label",
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a stratified audit sample for independent human checks."""

    if label_column not in df.columns or df.empty:
        return df.iloc[0:0].copy()

    samples: list[pd.DataFrame] = []
    for _, group in df.groupby(label_column, dropna=False):
        sample_size = min(n_per_class, len(group))
        if sample_size == 0:
            continue
        samples.append(group.sample(n=sample_size, random_state=random_state))
    if not samples:
        return df.iloc[0:0].copy()
    return pd.concat(samples, ignore_index=False).sort_index().copy()


def merge_human_annotations(
    auto_df: pd.DataFrame,
    human_df: pd.DataFrame,
    id_column: str = "annotation_id",
    human_label_column: str = "human_label",
) -> pd.DataFrame:
    """Merge human labels back into auto-labeled data and build final labels."""

    if human_label_column not in human_df.columns:
        raise ValueError(f"Expected `{human_label_column}` in human annotations.")

    merged = auto_df.copy()
    join_column = _resolve_join_column(auto_df, human_df, id_column)
    if join_column is None:
        merged = merged.reset_index(drop=True)
        human_labels = human_df[human_label_column].reset_index(drop=True)
        merged[human_label_column] = human_labels
    else:
        meta_extra = [
            column
            for column in human_df.columns
            if column not in {join_column, human_label_column}
            and (column.startswith("ls_") or column == "review_status")
        ]
        right_columns = [join_column, human_label_column, *meta_extra]
        right_columns = list(dict.fromkeys(c for c in right_columns if c in human_df.columns))
        merged = merged.merge(
            human_df[right_columns],
            on=join_column,
            how="left",
            suffixes=("", "_human"),
        )

    merged["final_label"] = merged[human_label_column].where(
        merged[human_label_column].notna() & (merged[human_label_column].astype(str).str.len() > 0),
        merged.get("auto_label"),
    )
    merged["final_label_source"] = merged[human_label_column].apply(
        lambda value: "human" if pd.notna(value) and str(value).strip() else "auto"
    )
    return merged


def _resolve_join_column(
    auto_df: pd.DataFrame,
    human_df: pd.DataFrame,
    requested_id_column: str,
) -> str | None:
    """Resolve the best join column between two dataframes."""

    if requested_id_column in auto_df.columns and requested_id_column in human_df.columns:
        return requested_id_column
    if "annotation_id" in auto_df.columns and "annotation_id" in human_df.columns:
        return "annotation_id"
    return None
