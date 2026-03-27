from __future__ import annotations

import fnmatch
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlparse

import pandas as pd
import requests

from agents.data_collection.connectors.base import BaseConnector, SourceSpecValidationError
from agents.data_collection.connectors.file_utils import (
    apply_light_mapping,
    apply_sample_size,
    download_file,
    load_data_files,
)
from agents.data_collection.schemas import SourceSpec, SourceType


DATASET_FILE_SUFFIXES = frozenset(
    {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet", ".xls", ".xlsx", ".zip"}
)

PATH_BONUS_SEGMENTS = (
    "data",
    "dataset",
    "datasets",
    "raw",
    "processed",
    "train",
    "test",
    "sample",
)

PATH_NOISE_MARKERS = (
    "readme",
    "license",
    "copying",
    "requirements",
    "setup.py",
    ".ipynb",
    "docs/",
    "doc/",
    "examples/",
    "example/",
    "notebooks/",
    "benchmark/",
    "/.github/",
    "__pycache__",
)


@dataclass(frozen=True)
class ParsedGitHubUrl:
    """Components extracted from a github.com repository URL."""

    owner: str
    repo: str
    branch_from_tree: str | None = None
    """Ref segment from a /tree/{ref}/... URL (not used if branch is set on SourceSpec)."""

    path_under_tree: str | None = None
    """Subdirectory under the tree ref (slash-separated, no leading slash)."""


class GitHubDataConnector(BaseConnector):
    """Connector for dataset files stored in public GitHub repositories."""

    API_BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None, timeout: float = 30.0) -> None:
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.timeout = timeout

    @property
    def connector_name(self) -> str:
        """Return the connector name."""

        return "github_dataset"

    def can_execute(self, source_spec: SourceSpec) -> tuple[bool, str | None]:
        """Return whether the repository exposes likely dataset files."""

        valid, reason = super().can_execute(source_spec)
        if not valid:
            return valid, reason

        try:
            repo_info = self._resolve_repo_info(source_spec)
            files = self._list_dataset_candidates(
                owner=repo_info["owner"],
                repo=repo_info["repo"],
                branch=repo_info["branch"],
                source_spec=source_spec,
                root_path=repo_info["root_path"],
            )
        except SourceSpecValidationError as exc:
            return False, str(exc)
        except requests.RequestException as exc:
            return False, f"GitHub API request failed (check network, URL, or token): {exc}"

        if not files:
            return (
                False,
                "No tabular dataset files (.csv, .tsv, .json, .jsonl, .parquet, .xls, .xlsx, .zip) "
                "were found via the GitHub Contents API under the resolved path. "
                "This is not masked as success: the repository may contain only source code, metadata, "
                "or notebooks—not ready-made table files. Try an explicit subpath or file_patterns, "
                "or pick another source (HF/Kaggle/direct file).",
            )
        return True, None

    def _download_repository_file(
        self,
        session: requests.Session,
        *,
        owner: str,
        repo: str,
        branch: str,
        candidate: dict[str, str | int],
        temp_path: Path,
    ) -> Path:
        """Download one repo file; prefer Contents API when token is set (avoids raw.githubusercontent 429 bursts)."""

        rel_path = str(candidate["path"])
        raw_url = str(candidate["download_url"])
        filename = PurePosixPath(rel_path).name

        if self.token:
            segments = [quote(p, safe="") for p in rel_path.strip("/").split("/") if p]
            path_in_url = "/".join(segments) if segments else quote(filename, safe="")
            api_url = f"{self.API_BASE_URL}/repos/{owner}/{repo}/contents/{path_in_url}"
            headers = {
                "Accept": "application/vnd.github.raw",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            try:
                r = session.get(
                    api_url,
                    params={"ref": branch},
                    headers=headers,
                    timeout=self.timeout,
                    stream=True,
                )
                if r.status_code == 200:
                    target = temp_path / filename
                    with target.open("wb") as handle:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                    return target
                r.close()
            except requests.RequestException:
                pass

        return download_file(
            raw_url,
            temp_path,
            timeout=self.timeout,
            headers=self._build_headers(raw=True),
            session=session,
        )

    def collect(self, source_spec: SourceSpec) -> pd.DataFrame:
        """Inspect a repository, download dataset files, and parse them."""

        self.validate_source_spec(source_spec)
        if self._coerce_source_type(source_spec.type) is not SourceType.GITHUB_DATASET:
            raise SourceSpecValidationError(
                "GitHubDataConnector only supports source type 'github_dataset'."
            )
        if not source_spec.enabled:
            return pd.DataFrame()

        try:
            repo_info = self._resolve_repo_info(source_spec)
            candidates = self._list_dataset_candidates(
                owner=repo_info["owner"],
                repo=repo_info["repo"],
                branch=repo_info["branch"],
                source_spec=source_spec,
                root_path=repo_info["root_path"],
            )
        except requests.RequestException as exc:
            raise SourceSpecValidationError(
                f"GitHub API request failed while accessing repository "
                f"(owner/repo from URL): {exc}"
            ) from exc

        if not candidates:
            raise SourceSpecValidationError(
                "No tabular dataset files (.csv, .tsv, .json, .jsonl, .parquet, .xls, .xlsx, .zip) "
                "were found via the GitHub Contents API under the resolved path in this repository. "
                "The repo likely holds code or non-tabular assets rather than a ready-made dataset export."
            )

        preferred_patterns = self._merged_file_patterns(source_spec)
        selected_candidates = self._select_remote_files(candidates, preferred_patterns=preferred_patterns)
        if not selected_candidates:
            raise SourceSpecValidationError(
                "Requested GitHub file patterns did not match any parsable repository files. "
                f"Patterns used: {preferred_patterns!r}."
            )

        with tempfile.TemporaryDirectory(prefix="github_data_") as temp_dir:
            temp_path = Path(temp_dir)
            session = requests.Session()
            downloaded_files: list[Path] = []
            owner, repo, branch = repo_info["owner"], repo_info["repo"], repo_info["branch"]
            for i, candidate in enumerate(selected_candidates):
                if i > 0 and not self.token:
                    url = str(candidate["download_url"])
                    if "raw.githubusercontent.com" in urlparse(url).netloc.casefold():
                        time.sleep(1.75)
                downloaded_files.append(
                    self._download_repository_file(
                        session,
                        owner=owner,
                        repo=repo,
                        branch=branch,
                        candidate=candidate,
                        temp_path=temp_path,
                    )
                )
            dataframe = load_data_files(
                downloaded_files,
                file_format=source_spec.file_format,
                compression=source_spec.compression,
                sheet_name=source_spec.sheet_name,
                max_rows=source_spec.sample_size,
            )
            dataframe = apply_sample_size(dataframe, source_spec.sample_size)
            return self.normalize_schema(dataframe, source_spec)

    def normalize_schema(self, df: pd.DataFrame, source_spec: SourceSpec) -> pd.DataFrame:
        """Apply lightweight field and label mapping."""

        return apply_light_mapping(df, source_spec)

    @staticmethod
    def _merged_file_patterns(source_spec: SourceSpec) -> list[str]:
        """Non-empty glob patterns from ``files`` and ``file_patterns``."""

        merged: list[str] = []
        for item in list(source_spec.files or []) + list(source_spec.file_patterns or []):
            if item and str(item).strip():
                merged.append(str(item).strip())
        return merged

    def _resolve_repo_info(self, source_spec: SourceSpec) -> dict[str, str]:
        """Resolve owner, repo, branch, and listing root path."""

        repo_url = source_spec.repo_url or source_spec.url
        if not repo_url:
            raise SourceSpecValidationError("GitHub source spec requires 'repo_url' or 'url'.")

        parsed = self._parse_github_repo_url(repo_url)
        branch = source_spec.branch or parsed.branch_from_tree
        if not branch:
            branch = self._fetch_default_branch(parsed.owner, parsed.repo)

        url_root = self._normalize_repo_subpath(parsed.path_under_tree)
        spec_root = self._normalize_repo_subpath(source_spec.subpath)
        root_path = spec_root if spec_root else url_root

        return {
            "owner": parsed.owner,
            "repo": parsed.repo,
            "branch": branch,
            "root_path": root_path,
        }

    @staticmethod
    def _parse_github_repo_url(repo_url: str) -> ParsedGitHubUrl:
        """Parse owner, repo, and optional /tree/{ref}/{subdir} from a GitHub HTTP(S) URL."""

        parsed = urlparse(repo_url.strip())
        host = parsed.netloc.casefold()
        if host.startswith("www."):
            host = host[4:]
        if host != "github.com":
            raise SourceSpecValidationError(
                f"Unsupported Git host in URL (expected github.com): '{repo_url}'."
            )

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise SourceSpecValidationError(
                f"GitHub repository URL must include owner and repo name: '{repo_url}'."
            )

        owner = parts[0]
        repo = parts[1]
        if repo.casefold().endswith(".git"):
            repo = repo[:-4]
        branch_from_tree: str | None = None
        path_under_tree: str | None = None

        if len(parts) > 2:
            kind = parts[2].casefold()
            if kind == "tree" and len(parts) > 3:
                branch_from_tree = parts[3]
                path_under_tree = "/".join(parts[4:]) if len(parts) > 4 else None
            elif kind in {"blob", "raw"} and len(parts) > 3:
                # Best-effort: treat like tree ref + file path for subdirectory context
                branch_from_tree = parts[3]
                path_under_tree = "/".join(parts[4:]) if len(parts) > 4 else None

        return ParsedGitHubUrl(
            owner=owner,
            repo=repo,
            branch_from_tree=branch_from_tree,
            path_under_tree=path_under_tree,
        )

    @staticmethod
    def _normalize_repo_subpath(subpath: str | None) -> str:
        """Normalize GitHub contents API path (no leading slash)."""

        if not subpath:
            return ""
        return subpath.strip().strip("/")

    def _list_dataset_candidates(
        self,
        *,
        owner: str,
        repo: str,
        branch: str,
        source_spec: SourceSpec,
        root_path: str = "",
    ) -> list[dict[str, str | int]]:
        """List likely dataset files in the repository."""

        session = requests.Session()
        root_entries = self._get_repo_entries(session, owner, repo, branch, root_path)
        max_depth = source_spec.max_depth if source_spec.max_depth is not None else 4
        files = self._walk_entries(
            session=session,
            owner=owner,
            repo=repo,
            branch=branch,
            entries=root_entries,
            depth=0,
            max_depth=max_depth,
        )
        filtered = [entry for entry in files if self._is_dataset_file(entry["path"])]
        return sorted(filtered, key=self._remote_file_score, reverse=True)

    def _walk_entries(
        self,
        *,
        session: requests.Session,
        owner: str,
        repo: str,
        branch: str,
        entries: list[dict],
        depth: int,
        max_depth: int,
    ) -> list[dict[str, str | int]]:
        """Recursively collect file entries from the repository."""

        collected: list[dict[str, str | int]] = []
        for entry in entries:
            entry_type = entry.get("type")
            if entry_type == "file" and entry.get("download_url"):
                collected.append(
                    {
                        "path": entry["path"],
                        "download_url": entry["download_url"],
                        "size": int(entry.get("size") or 0),
                    }
                )
            elif entry_type == "dir" and depth < max_depth:
                child_entries = self._get_repo_entries(
                    session,
                    owner,
                    repo,
                    branch,
                    entry["path"],
                )
                collected.extend(
                    self._walk_entries(
                        session=session,
                        owner=owner,
                        repo=repo,
                        branch=branch,
                        entries=child_entries,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                )
        return collected

    def _get_repo_entries(
        self,
        session: requests.Session,
        owner: str,
        repo: str,
        branch: str,
        path: str,
    ) -> list[dict]:
        """Fetch one repository directory listing via the GitHub contents API."""

        if path:
            url = f"{self.API_BASE_URL}/repos/{owner}/{repo}/contents/{path}"
        else:
            url = f"{self.API_BASE_URL}/repos/{owner}/{repo}/contents"

        try:
            response = session.get(
                url,
                params={"ref": branch},
                headers=self._build_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            hint = ""
            if status == 404:
                hint = " The path or branch may not exist, or the repository is private without a valid token."
            elif status in (401, 403):
                hint = " Authentication may be required (set GITHUB_TOKEN for private repos)."
            raise SourceSpecValidationError(
                f"Could not read GitHub repository '{owner}/{repo}' at path '{path or '/'}' "
                f"(HTTP {status}).{hint}"
            ) from exc

        if isinstance(payload, dict):
            return [payload]
        if not isinstance(payload, list):
            return []
        return payload

    def _fetch_default_branch(self, owner: str, repo: str) -> str:
        """Fetch the repository default branch."""

        try:
            response = requests.get(
                f"{self.API_BASE_URL}/repos/{owner}/{repo}",
                headers=self._build_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            hint = ""
            if status == 404:
                hint = " Repository not found or not visible (private repo needs GITHUB_TOKEN)."
            raise SourceSpecValidationError(
                f"Could not load GitHub repository metadata for '{owner}/{repo}' (HTTP {status}).{hint}"
            ) from exc

        default_branch = payload.get("default_branch")
        if not default_branch:
            raise SourceSpecValidationError(
                f"GitHub repository '{owner}/{repo}' did not expose a default_branch field."
            )
        return default_branch

    def _build_headers(self, *, raw: bool = False) -> dict[str, str]:
        """Build GitHub request headers."""

        headers = {
            "Accept": "application/vnd.github+json" if not raw else "application/octet-stream",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    @staticmethod
    def _is_dataset_file(path: str) -> bool:
        """Return whether a repository path looks like a dataset file."""

        file_path = PurePosixPath(path)
        if file_path.suffix.casefold() not in {ext.casefold() for ext in DATASET_FILE_SUFFIXES}:
            return False
        lowered = file_path.as_posix().casefold()
        hard_excludes = ("readme", "license", "requirements", "setup.py", ".ipynb")
        return not any(marker in lowered for marker in hard_excludes)

    @staticmethod
    def _remote_file_score(entry: dict[str, str | int]) -> tuple[int, int, int, int]:
        """Rank remote files: higher tuple compares greater."""

        path = str(entry["path"]).casefold()
        path_slash = f"/{path}/"
        name = PurePosixPath(path).name
        score = 0

        for segment in PATH_BONUS_SEGMENTS:
            if f"/{segment}/" in path_slash or path.startswith(f"{segment}/"):
                score += 14

        extension_weights = {
            ".parquet": 18,
            ".csv": 15,
            ".jsonl": 14,
            ".ndjson": 14,
            ".tsv": 12,
            ".xlsx": 10,
            ".xls": 8,
            ".json": 6,
            ".zip": 5,
        }
        ext = PurePosixPath(path).suffix.casefold()
        score += extension_weights.get(ext, 0)

        for marker in PATH_NOISE_MARKERS:
            if marker in path or marker in name:
                score -= 85

        size = int(entry.get("size") or 0)
        if size > 0 and size < 512:
            score -= 40
        elif 0 < size < 4096:
            score -= 12

        score += min(size // 800, 3_500)

        parts_count = len(PurePosixPath(path).parts)
        return (score, size, -parts_count, -len(name))

    @staticmethod
    def _select_remote_files(
        candidates: list[dict[str, str | int]],
        *,
        preferred_patterns: list[str] | None = None,
    ) -> list[dict[str, str | int]]:
        """Select files by glob patterns, or the single best-scoring file."""

        if not candidates:
            return []

        patterns = [
            pattern.strip()
            for pattern in (preferred_patterns or [])
            if pattern and str(pattern).strip()
        ]
        if patterns:
            matched = [
                candidate
                for candidate in candidates
                if any(
                    fnmatch.fnmatch(PurePosixPath(str(candidate["path"])).name, pattern)
                    or fnmatch.fnmatch(str(candidate["path"]), pattern)
                    for pattern in patterns
                )
            ]
            return sorted(matched, key=lambda c: str(c["path"]))

        primary = max(candidates, key=GitHubDataConnector._remote_file_score)
        return [primary]
