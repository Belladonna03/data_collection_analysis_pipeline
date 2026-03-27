from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from agents.data_collection.connectors.base import BaseConnector, SourceSpecValidationError
from agents.data_collection.connectors.file_utils import (
    apply_light_mapping,
    apply_sample_size,
    collect_candidate_files,
    load_data_files,
    select_data_files,
)
from agents.data_collection.schemas import SourceSpec, SourceType


_CREDENTIALS_HELP = (
    "Configure Kaggle API access using one of: "
    "(1) environment variables KAGGLE_USERNAME and KAGGLE_KEY (same values as in kaggle.json), "
    "(2) KAGGLE_API_TOKEN, "
    "(3) file ~/.kaggle/kaggle.json, or "
    "(4) ./.kaggle/kaggle.json with KAGGLE_CONFIG_DIR pointing at the parent directory. "
    "See https://www.kaggle.com/docs/api"
)


class KaggleConnector(BaseConnector):
    """Connector for Kaggle datasets."""

    def __init__(self, download_dir: str | None = None) -> None:
        self.download_dir = Path(download_dir) if download_dir else None

    @property
    def connector_name(self) -> str:
        """Return the connector name."""

        return "kaggle"

    def can_execute(self, source_spec: SourceSpec) -> tuple[bool, str | None]:
        """Return whether Kaggle execution prerequisites are satisfied."""

        valid, reason = super().can_execute(source_spec)
        if not valid:
            return valid, reason
        try:
            self._import_kaggle_api()
        except ImportError as exc:
            return False, str(exc)

        dataset_ref = source_spec.dataset_ref or source_spec.name
        if not dataset_ref:
            return False, "Kaggle source spec requires a dataset reference (dataset_ref or name)."

        if not self._has_kaggle_credentials():
            return False, f"Kaggle credentials are missing. {_CREDENTIALS_HELP}"

        return True, None

    def collect(self, source_spec: SourceSpec) -> pd.DataFrame:
        """Download a Kaggle dataset and parse the primary tabular file(s)."""

        self.validate_source_spec(source_spec)
        if self._coerce_source_type(source_spec.type) is not SourceType.KAGGLE:
            raise SourceSpecValidationError("KaggleConnector only supports source type 'kaggle'.")
        if not source_spec.enabled:
            return pd.DataFrame()

        dataset_ref = source_spec.dataset_ref or source_spec.name
        if not dataset_ref:
            raise SourceSpecValidationError(
                "Kaggle dataset reference is missing: set 'dataset_ref' (owner/slug) or 'name' on SourceSpec."
            )

        if not self._has_kaggle_credentials():
            raise SourceSpecValidationError(
                f"Cannot download Kaggle dataset '{dataset_ref}': credentials are not configured. "
                f"{_CREDENTIALS_HELP}"
            )

        KaggleApi = self._import_kaggle_api()
        with self._temporary_credentials():
            api = KaggleApi()
            try:
                api.authenticate()
            except Exception as exc:
                raise SourceSpecValidationError(
                    f"Kaggle authentication failed for dataset '{dataset_ref}'. "
                    "The API rejected the request — check username/key or token, and that the account "
                    "has accepted Kaggle API rules. "
                    f"Underlying error: {exc}"
                ) from exc

            with self._working_directory(dataset_ref) as working_dir:
                try:
                    api.dataset_download_files(
                        dataset_ref,
                        path=str(working_dir),
                        quiet=True,
                        unzip=True,
                    )
                except Exception as exc:
                    raise SourceSpecValidationError(
                        f"Kaggle download failed for dataset '{dataset_ref}'. "
                        "Verify the dataset slug (owner/name), that you accepted competition/dataset rules on the site, "
                        "and that the dataset is accessible to your account. "
                        f"Underlying error: {exc}"
                    ) from exc

                selected_files = self._select_dataset_files(working_dir, source_spec, dataset_ref=dataset_ref)
                try:
                    dataframe = load_data_files(
                        selected_files,
                        file_format=source_spec.file_format,
                        compression=source_spec.compression,
                        sheet_name=source_spec.sheet_name,
                        max_rows=source_spec.sample_size,
                    )
                except SourceSpecValidationError as exc:
                    raise SourceSpecValidationError(
                        f"Failed to read selected Kaggle files for '{dataset_ref}': {exc}"
                    ) from exc

                dataframe = apply_sample_size(dataframe, source_spec.sample_size)
                return self.normalize_schema(dataframe, source_spec)

    def normalize_schema(self, df: pd.DataFrame, source_spec: SourceSpec) -> pd.DataFrame:
        """Apply lightweight field and label mapping."""

        return apply_light_mapping(df, source_spec)

    @staticmethod
    def _explicit_file_patterns(source_spec: SourceSpec) -> list[str]:
        """Merge non-empty entries from ``files`` and ``file_patterns`` (explicit selection)."""

        merged: list[str] = []
        for item in list(source_spec.files or []) + list(source_spec.file_patterns or []):
            if item and str(item).strip():
                merged.append(str(item).strip())
        return merged

    def _select_dataset_files(
        self,
        working_dir: Path,
        source_spec: SourceSpec,
        *,
        dataset_ref: str,
    ) -> list[Path]:
        """Select files after download: explicit glob patterns or heuristic single best file."""

        candidates = collect_candidate_files(working_dir)
        if not candidates:
            raise SourceSpecValidationError(
                f"No parsable tabular files were found after downloading Kaggle dataset '{dataset_ref}'. "
                "Expected extensions such as .csv, .tsv, .json, .jsonl, .parquet, .xls, or .xlsx under the "
                "extracted bundle (nested zips are expanded when named .zip)."
            )

        explicit = self._explicit_file_patterns(source_spec)
        selected = select_data_files(candidates, preferred_patterns=explicit if explicit else None)
        if not selected:
            raise SourceSpecValidationError(
                f"No downloaded files matched the requested patterns {explicit!r} for Kaggle dataset "
                f"'{dataset_ref}'. List files in the bundle or remove patterns to use automatic file selection."
            )
        return selected

    @staticmethod
    def _import_kaggle_api():
        """Import KaggleApi lazily."""

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:
            raise ImportError(
                "KaggleConnector requires the 'kaggle' package. Install it with: pip install kaggle"
            ) from exc
        return KaggleApi

    @staticmethod
    def _has_kaggle_credentials() -> bool:
        """Return whether environment or standard Kaggle credentials are available."""

        if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
            return True
        if os.getenv("KAGGLE_API_TOKEN"):
            return True
        dotenv_values = KaggleConnector._read_local_dotenv()
        if dotenv_values.get("KAGGLE_API_TOKEN"):
            return True
        username = dotenv_values.get("KAGGLE_USERNAME") or dotenv_values.get("KAGGLE_NAME")
        if username and dotenv_values.get("KAGGLE_KEY"):
            return True
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.is_file():
            return True
        cwd_kaggle = Path.cwd() / ".kaggle" / "kaggle.json"
        return cwd_kaggle.is_file()

    @contextmanager
    def _temporary_credentials(self) -> Iterator[None]:
        """Temporarily expose credentials from env or local files."""

        original_username = os.environ.get("KAGGLE_USERNAME")
        original_key = os.environ.get("KAGGLE_KEY")
        original_config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
        original_token = os.environ.get("KAGGLE_API_TOKEN")

        dotenv_values = self._read_local_dotenv()
        username = (
            os.environ.get("KAGGLE_USERNAME")
            or dotenv_values.get("KAGGLE_USERNAME")
            or dotenv_values.get("KAGGLE_NAME")
        )
        key = os.environ.get("KAGGLE_KEY") or dotenv_values.get("KAGGLE_KEY")
        api_token = os.environ.get("KAGGLE_API_TOKEN") or dotenv_values.get("KAGGLE_API_TOKEN")
        local_kaggle_dir = Path.cwd() / ".kaggle"

        if username and key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
        if api_token:
            os.environ["KAGGLE_API_TOKEN"] = api_token
        if local_kaggle_dir.is_dir() and (local_kaggle_dir / "kaggle.json").is_file():
            os.environ["KAGGLE_CONFIG_DIR"] = str(local_kaggle_dir.resolve())

        try:
            yield
        finally:
            self._restore_env_var("KAGGLE_USERNAME", original_username)
            self._restore_env_var("KAGGLE_KEY", original_key)
            self._restore_env_var("KAGGLE_CONFIG_DIR", original_config_dir)
            self._restore_env_var("KAGGLE_API_TOKEN", original_token)

    @contextmanager
    def _working_directory(self, dataset_ref: str) -> Iterator[Path]:
        """Return a managed download directory."""

        if self.download_dir is not None:
            target_dir = self.download_dir / dataset_ref.replace("/", "__")
            target_dir.mkdir(parents=True, exist_ok=True)
            yield target_dir
            return

        with tempfile.TemporaryDirectory(prefix="kaggle_data_") as temp_dir:
            yield Path(temp_dir)

    @staticmethod
    def _read_local_dotenv() -> dict[str, str]:
        """Read simple KEY=VALUE pairs from the local .env file."""

        env_path = Path.cwd() / ".env"
        if not env_path.exists():
            return {}

        values: dict[str, str] = {}
        try:
            with env_path.open("r", encoding="utf-8") as file:
                for line in file:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key, value = stripped.split("=", 1)
                    values[key.strip()] = value.strip().strip("'\"")
        except OSError:
            return {}
        return values

    @staticmethod
    def _restore_env_var(name: str, value: str | None) -> None:
        """Restore one environment variable."""

        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
