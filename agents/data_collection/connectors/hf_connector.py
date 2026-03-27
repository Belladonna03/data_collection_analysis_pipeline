from __future__ import annotations

import os
from typing import Any

import pandas as pd

from agents.data_collection.connectors.base import BaseConnector, SourceSpecValidationError
from agents.data_collection.connectors.file_utils import apply_light_mapping, apply_sample_size
from agents.data_collection.schemas import SourceSpec, SourceType


def _split_string_has_slice(split: str) -> bool:
    s = split.strip()
    return "[" in s and "]" in s


class HuggingFaceConnector(BaseConnector):
    """Connector for Hugging Face datasets."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.getenv("HF_TOKEN")

    @property
    def connector_name(self) -> str:
        """Return the connector name."""

        return "hf_dataset"

    def collect(self, source_spec: SourceSpec) -> pd.DataFrame:
        """Load a Hugging Face dataset and return a DataFrame."""

        self.validate_source_spec(source_spec)
        if self._coerce_source_type(source_spec.type) is not SourceType.HF_DATASET:
            raise SourceSpecValidationError(
                "HFDatasetConnector only supports source type 'hf_dataset'."
            )
        if not source_spec.enabled:
            return pd.DataFrame()

        load_dataset = self._import_load_dataset()
        load_kwargs = self._source_spec_to_load_dataset_kwargs(source_spec)
        sample_cap = source_spec.sample_size
        if (
            sample_cap
            and not source_spec.streaming
            and isinstance(load_kwargs.get("split"), str)
            and not _split_string_has_slice(load_kwargs["split"])
        ):
            load_kwargs["split"] = f"{load_kwargs['split']}[:{int(sample_cap)}]"

        try:
            dataset = load_dataset(**load_kwargs)
        except ValueError as exc:
            if source_spec.split:
                raise ValueError(
                    f"Split '{source_spec.split}' was not found for dataset '{source_spec.name}'."
                ) from exc
            raise

        df = self._dataset_to_dataframe(dataset, source_spec, row_cap=sample_cap)
        if df.empty:
            raise ValueError(f"Dataset '{source_spec.name}' returned no rows.")

        df = apply_sample_size(df, sample_cap)
        if df.empty:
            raise ValueError(
                f"Dataset '{source_spec.name or source_spec.dataset_id}' is empty after applying sample_size."
            )

        return self.normalize_schema(df, source_spec)

    def normalize_schema(self, df: pd.DataFrame, source_spec: SourceSpec) -> pd.DataFrame:
        """Apply lightweight unified field and label mapping."""

        return apply_light_mapping(df, source_spec)

    def _source_spec_to_load_dataset_kwargs(self, source_spec: SourceSpec) -> dict[str, Any]:
        """Convert a source spec into datasets.load_dataset kwargs."""

        kwargs: dict[str, Any] = {
            "path": source_spec.dataset_id or source_spec.name,
            **source_spec.params,
        }
        if source_spec.subset:
            kwargs["name"] = source_spec.subset
        if source_spec.split:
            kwargs["split"] = source_spec.split
        if source_spec.revision:
            kwargs["revision"] = source_spec.revision
        if source_spec.streaming:
            kwargs["streaming"] = True
        if self.token:
            kwargs["token"] = self.token
        return kwargs

    def _dataset_to_dataframe(
        self,
        dataset: Any,
        source_spec: SourceSpec,
        *,
        row_cap: int | None = None,
    ) -> pd.DataFrame:
        """Convert datasets output to a pandas DataFrame."""

        if hasattr(dataset, "to_pandas"):
            limited = self._take_dataset_rows(dataset, row_cap)
            return limited.to_pandas()

        if source_spec.split and hasattr(dataset, "__getitem__"):
            try:
                split_dataset = dataset[source_spec.split]
            except KeyError as exc:
                raise ValueError(
                    f"Split '{source_spec.split}' was not found for dataset '{source_spec.name}'."
                ) from exc
            if hasattr(split_dataset, "to_pandas"):
                limited = self._take_dataset_rows(split_dataset, row_cap)
                return limited.to_pandas()

        if hasattr(dataset, "keys"):
            available_splits = list(dataset.keys())
            if not available_splits:
                raise ValueError(f"Dataset '{source_spec.name}' has no available splits.")

            preferred_split = "train" if "train" in dataset else available_splits[0]
            split_dataset = dataset[preferred_split]
            if hasattr(split_dataset, "to_pandas"):
                limited = self._take_dataset_rows(split_dataset, row_cap)
                return limited.to_pandas()

        raise ValueError(
            f"Dataset '{source_spec.name}' could not be converted to pandas DataFrame."
        )

    @staticmethod
    def _take_dataset_rows(ds: Any, row_cap: int | None) -> Any:
        if row_cap is None or row_cap <= 0:
            return ds
        try:
            n = len(ds)
        except Exception:
            return ds
        if n <= row_cap:
            return ds
        if hasattr(ds, "select"):
            return ds.select(range(int(row_cap)))
        return ds

    @staticmethod
    def _import_load_dataset():
        """Import datasets.load_dataset lazily."""

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "HFDatasetConnector requires the 'datasets' package. "
                "Install it with: pip install datasets"
            ) from exc
        return load_dataset


class HFDatasetConnector(HuggingFaceConnector):
    """Backward-compatible alias for the Hugging Face connector."""


# Example:
# source_spec = SourceSpec(
#     id="imdb-train",
#     type=SourceType.HF_DATASET,
#     name="imdb",
#     split="train",
#     sample_size=100,
#     field_map={"text": "text", "label": "label"},
#     label_map={0: "negative", 1: "positive"},
# )
