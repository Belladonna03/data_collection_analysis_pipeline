from __future__ import annotations

import fnmatch
import json
import mimetypes
import random
import time
import zipfile
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from agents.data_collection.connectors.base import SourceSpecValidationError
from agents.data_collection.schemas import SourceSpec


TABULAR_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".parquet",
    ".xls",
    ".xlsx",
    ".zip",
}
DATA_DIRECTORY_HINTS = ("data", "dataset", "datasets", "raw", "processed")
DATA_FILE_HINTS = ("data", "dataset", "records", "matches", "samples", "train", "test")
# Path/name markers that usually indicate non-tabular or auxiliary files.
NOISE_MARKERS = (
    "readme",
    "license",
    "copying",
    "schema",
    "dictionary",
    "metadata",
    "changelog",
    "contributing",
    "notice",
    "sample_submission",
    "__macosx",
)
# Below this size (bytes) tabular files are often placeholders or headers-only.
TINY_FILE_BYTE_THRESHOLD = 384
SMALL_FILE_BYTE_THRESHOLD = 2048


@dataclass(frozen=True)
class HttpDownloadResult:
    """Local path plus HTTP metadata from the successful GET response."""

    path: Path
    content_type: str | None = None


def content_type_indicates_html(content_type: str | None) -> bool:
    """Return True when Content-Type clearly denotes an HTML document."""

    if not content_type:
        return False
    primary = content_type.split(";")[0].strip().casefold()
    if primary.startswith("text/html"):
        return True
    return primary in {"application/xhtml+xml", "application/xhtml", "text/xhtml"}


def _bytes_look_like_html(data: bytes) -> bool:
    """Heuristic: leading bytes resemble an HTML document."""

    if not data:
        return False
    head = data.lstrip()[:1200]
    try:
        text = head.decode("utf-8", errors="replace").casefold()
    except Exception:
        return False
    if text.startswith("<!doctype html") or text.startswith("<html"):
        return True
    snippet = text[:500]
    return "<html" in snippet or "<head" in snippet or "<body" in snippet


def file_snippet_looks_like_html(path: Path, max_bytes: int = 8192) -> bool:
    """Read the start of a file and check for HTML markers."""

    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes)
    except OSError:
        return False
    return _bytes_look_like_html(data)


def _http_retry_wait_s(retry_after_header: str | None, attempt_index: int) -> float:
    """Seconds to wait before retrying after 429/503 (Retry-After or exponential)."""

    if retry_after_header:
        try:
            return min(300.0, float(str(retry_after_header).strip()))
        except ValueError:
            pass
    # raw.githubusercontent.com often needs longer gaps than a single quick retry
    base = min(120.0, 3.0 * (2**attempt_index))
    return base * (1.0 + random.random() * 0.15)


def download_http_asset(
    url: str,
    target_dir: Path,
    *,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    session: requests.Session | None = None,
    max_retries: int = 10,
) -> HttpDownloadResult:
    """Stream a URL to disk and return the path plus Content-Type / filename hints."""

    target_dir.mkdir(parents=True, exist_ok=True)
    http = session or requests.Session()
    response: requests.Response | None = None
    for attempt in range(max_retries + 1):
        r = http.get(url, timeout=timeout, headers=headers, stream=True)
        if r.status_code in (429, 503) and attempt < max_retries:
            wait_s = _http_retry_wait_s(r.headers.get("Retry-After"), attempt)
            r.close()
            time.sleep(wait_s)
            continue
        response = r
        break

    assert response is not None
    response.raise_for_status()

    filename = _filename_from_response(response.headers) or _filename_from_url(url) or "downloaded"
    target_path = target_dir / filename
    with target_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)

    content_type = response.headers.get("Content-Type")
    return HttpDownloadResult(path=target_path, content_type=content_type)


def download_file(
    url: str,
    target_dir: Path,
    *,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    session: requests.Session | None = None,
) -> Path:
    """Download a file to the target directory."""

    return download_http_asset(
        url,
        target_dir,
        timeout=timeout,
        headers=headers,
        session=session,
    ).path


def extract_zip_archive(zip_path: Path, target_dir: Path) -> list[Path]:
    """Extract a zip archive and return all files inside."""

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)
    return [path for path in target_dir.rglob("*") if path.is_file()]


def collect_candidate_files(path: Path) -> list[Path]:
    """Return all parsable tabular files from a file or directory."""

    if path.is_file():
        if zipfile.is_zipfile(path):
            extract_dir = path.parent / f"{path.stem}_unzipped"
            return [
                candidate
                for candidate in extract_zip_archive(path, extract_dir)
                if candidate.suffix.casefold() in TABULAR_EXTENSIONS - {".zip"}
            ]
        return [path] if path.suffix.casefold() in TABULAR_EXTENSIONS else []

    return [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.casefold() in TABULAR_EXTENSIONS - {".zip"}
    ]


def select_data_files(
    candidates: list[Path],
    *,
    preferred_patterns: list[str] | None = None,
) -> list[Path]:
    """Select data files using explicit glob patterns or a single heuristic winner.

    If ``preferred_patterns`` is non-empty, every pattern must be non-blank; only
    files matching at least one pattern are returned (sorted). Otherwise the
    highest-scoring candidate under :func:`_file_selection_score` is chosen.
    """

    if not candidates:
        return []

    preferred_patterns = [
        pattern.strip()
        for pattern in (preferred_patterns or [])
        if pattern and str(pattern).strip()
    ]
    if preferred_patterns:
        matched = [
            candidate
            for candidate in candidates
            if any(
                fnmatch.fnmatch(candidate.name, pattern) or fnmatch.fnmatch(candidate.as_posix(), pattern)
                for pattern in preferred_patterns
            )
        ]
        return sorted(matched) if matched else []

    return [max(candidates, key=_file_selection_score)]


def load_data_files(
    file_paths: list[Path],
    *,
    file_format: str | None = None,
    compression: str | None = None,
    sheet_name: str | int | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load one or more local data files into a DataFrame.

    When *max_rows* is set and a single file is loaded, parsers use row limits where supported
    (CSV/TSV/JSONL/Parquet) to avoid reading entire huge files.
    """

    if not file_paths:
        raise SourceSpecValidationError("No parsable data files were selected.")

    nrows = max_rows if max_rows is not None and max_rows > 0 and len(file_paths) == 1 else None

    dataframes: list[pd.DataFrame] = []
    for file_path in file_paths:
        dataframe = load_single_file(
            file_path,
            file_format=file_format,
            compression=compression,
            sheet_name=sheet_name,
            nrows=nrows,
        )
        if len(file_paths) > 1 and "__source_file__" not in dataframe.columns:
            dataframe["__source_file__"] = file_path.name
        dataframes.append(dataframe)

    merged = pd.concat(dataframes, ignore_index=True, sort=False) if len(dataframes) > 1 else dataframes[0]
    if merged.empty:
        raise SourceSpecValidationError("Selected data file(s) produced an empty dataframe.")
    if max_rows is not None and max_rows > 0 and (len(file_paths) > 1 or nrows is None):
        merged = merged.iloc[: int(max_rows)].copy()
    return merged


def load_single_file(
    file_path: Path,
    *,
    file_format: str | None = None,
    compression: str | None = None,
    sheet_name: str | int | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load one supported local file into a DataFrame."""

    detected_format = detect_file_format(file_path, explicit_format=file_format)
    nr = nrows if nrows is not None and nrows > 0 else None
    if detected_format == "csv":
        return pd.read_csv(file_path, compression=compression, nrows=nr)
    if detected_format == "tsv":
        return pd.read_csv(file_path, sep="\t", compression=compression, nrows=nr)
    if detected_format == "parquet":
        if nr is not None:
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(file_path)
                take = min(int(nr), table.num_rows)
                return table.slice(0, take).to_pandas()
            except Exception:
                df = pd.read_parquet(file_path)
                return df.iloc[:nr].copy()
        return pd.read_parquet(file_path)
    if detected_format in {"xls", "xlsx"}:
        df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
        return df.iloc[:nr].copy() if nr is not None else df
    if detected_format in {"json", "jsonl"}:
        return _read_json_file(
            file_path,
            lines=(detected_format == "jsonl"),
            nrows=nr,
        )

    raise SourceSpecValidationError(
        f"Unsupported file format '{detected_format}' for '{file_path.name}'."
    )


def detect_file_format(
    file_path: Path,
    *,
    explicit_format: str | None = None,
    content_type: str | None = None,
    content_disposition: str | None = None,
) -> str:
    """Detect a file format from explicit hints, path, or HTTP metadata."""

    if explicit_format:
        return explicit_format.casefold().lstrip(".")

    suffix = file_path.suffix.casefold().lstrip(".")
    if suffix:
        return "jsonl" if suffix == "ndjson" else suffix

    if content_disposition:
        message = Message()
        message["content-disposition"] = content_disposition
        filename = message.get_filename()
        if filename:
            guessed_suffix = Path(filename).suffix.casefold().lstrip(".")
            if guessed_suffix:
                return guessed_suffix

    if content_type:
        guessed_extension = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed_extension:
            return guessed_extension.lstrip(".")

    raise SourceSpecValidationError(f"Could not detect file format for '{file_path.name}'.")


def apply_sample_size(df: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    """Apply an optional row limit to a DataFrame."""

    if sample_size is None:
        return df
    return df.head(sample_size).copy()


def apply_light_mapping(df: pd.DataFrame, source_spec: SourceSpec) -> pd.DataFrame:
    """Apply field_map and label_map without full normalization."""

    mapped = df.copy()
    for unified_field, source_field in source_spec.field_map.items():
        if source_field in mapped.columns:
            mapped[unified_field] = mapped[source_field]

    if source_spec.label_map and "label" in mapped.columns:
        mapped["label"] = mapped["label"].map(lambda value: source_spec.label_map.get(value, value))

    return mapped


def _file_selection_score(file_path: Path) -> tuple[int, int, int, int]:
    """Return a sortable key: higher first tuple element = better data file.

    Bonuses favor common tabular layouts under ``data/`` / ``train`` and
    ``.csv`` / ``.parquet`` / ``.jsonl``. Penalties down-rank licenses, schemas,
    and very small files that are usually not the main table.
    """

    path_text = file_path.as_posix().casefold()
    name_text = file_path.name.casefold()
    score = 0

    score += sum(22 for hint in DATA_DIRECTORY_HINTS if f"/{hint}/" in path_text)
    # Extra weight for typical ML splits in path or file name.
    for hint in ("train", "test", "sample"):
        if hint in path_text or hint in name_text:
            score += 10
    score += sum(6 for hint in DATA_FILE_HINTS if hint in name_text)

    extension_weights = {
        ".parquet": 16,
        ".csv": 14,
        ".jsonl": 13,
        ".tsv": 10,
        ".xlsx": 9,
        ".xls": 7,
        ".json": 5,
    }
    score += extension_weights.get(file_path.suffix.casefold(), 0)

    for marker in NOISE_MARKERS:
        if marker in path_text or marker in name_text:
            score -= 120

    try:
        file_size = int(file_path.stat().st_size)
    except OSError:
        file_size = 0

    if file_size < TINY_FILE_BYTE_THRESHOLD:
        score -= 90
    elif file_size < SMALL_FILE_BYTE_THRESHOLD:
        score -= 25

    # Prefer larger files up to a cap so the main table wins over tiny samples.
    score += min(file_size // 512, 4_000)

    depth_penalty = len(file_path.parts)
    return (score, file_size, -depth_penalty, -len(name_text))


def _filename_from_response(headers: requests.structures.CaseInsensitiveDict[str]) -> str | None:
    content_disposition = headers.get("content-disposition")
    if not content_disposition:
        return None
    message = Message()
    message["content-disposition"] = content_disposition
    return message.get_filename()


def _filename_from_url(url: str) -> str | None:
    path = urlparse(url).path
    filename = Path(path).name
    return filename or None


def _read_json_file(file_path: Path, *, lines: bool, nrows: int | None = None) -> pd.DataFrame:
    """Read JSON or JSONL into a DataFrame."""

    if lines:
        try:
            return pd.read_json(file_path, lines=True, nrows=nrows)
        except TypeError:
            if nrows is None:
                return pd.read_json(file_path, lines=True)
            rows: list[dict] = []
            with file_path.open("r", encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    if i >= nrows:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            if not rows:
                return pd.DataFrame()
            return pd.json_normalize(rows)

    try:
        return pd.read_json(file_path)
    except ValueError:
        with file_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    return pd.json_normalize(value)
            return pd.json_normalize(payload)
        raise SourceSpecValidationError(f"JSON file '{file_path.name}' is not tabular enough to load.")
