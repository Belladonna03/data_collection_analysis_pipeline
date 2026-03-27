from __future__ import annotations

import csv
import os
import shutil
import subprocess
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

from agents.data_collection.schemas import DiscoveryCapability, DiscoveryProvider, RawSearchHit
from agents.data_collection.search_clients.base import BaseSearchClient, SearchClientError

class KaggleSearchClient(BaseSearchClient):
    """Search Kaggle datasets through the Kaggle CLI."""

    @property
    def provider(self) -> DiscoveryProvider:
        """Return the provider identifier."""

        return DiscoveryProvider.KAGGLE

    def check_capability(self) -> DiscoveryCapability:
        """Check whether Kaggle CLI search is available."""
        if not self.config.enabled:
            return DiscoveryCapability(
                provider=self.provider,
                available=False,
                reason="Provider disabled by config.",
            )

        command = self.config.command or "kaggle"
        which_result = shutil.which(command)
        if which_result is None:
            return DiscoveryCapability(
                provider=self.provider,
                available=False,
                reason="Kaggle CLI not found in PATH.",
            )

        dotenv_values = self._read_local_dotenv()
        access_token = (
            os.getenv("KAGGLE_API_TOKEN")
            or dotenv_values.get("KAGGLE_API_TOKEN")
        )
        username = (
            self.config.username
            or os.getenv("KAGGLE_USERNAME")
            or os.getenv("KAGGLE_NAME")
            or dotenv_values.get("KAGGLE_USERNAME")
            or dotenv_values.get("KAGGLE_NAME")
        )
        key = (
            self.config.key
            or os.getenv("KAGGLE_KEY")
            or dotenv_values.get("KAGGLE_KEY")
        )
        if not access_token and (not username or not key):
            return DiscoveryCapability(
                provider=self.provider,
                available=False,
                reason="KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY not configured.",
            )
        return DiscoveryCapability(provider=self.provider, available=True)

    def search(self, query: str) -> list[RawSearchHit]:
        """Search Kaggle datasets using the Kaggle CLI."""

        capability = self.check_capability()
        if not capability.available:
            raise SearchClientError(capability.reason)

        command = self.config.command or "kaggle"
        dotenv_values = self._read_local_dotenv()
        env = dict(os.environ)
        env["KAGGLE_API_TOKEN"] = (
            env.get("KAGGLE_API_TOKEN")
            or dotenv_values.get("KAGGLE_API_TOKEN")
            or ""
        )
        env["KAGGLE_USERNAME"] = (
            self.config.username
            or env.get("KAGGLE_USERNAME")
            or env.get("KAGGLE_NAME")
            or dotenv_values.get("KAGGLE_USERNAME")
            or dotenv_values.get("KAGGLE_NAME")
            or ""
        )
        env["KAGGLE_KEY"] = (
            self.config.key
            or env.get("KAGGLE_KEY")
            or dotenv_values.get("KAGGLE_KEY")
            or ""
        )

        args = [
            command,
            "datasets",
            "list",
            "-s",
            query,
            "--csv",
        ]

        try:
            completed = subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                env=env,
            )
        except Exception as exc:
            raise SearchClientError(f"Kaggle CLI search failed for query '{query}'.") from exc

        reader = csv.DictReader(StringIO(completed.stdout))
        fetched_at = datetime.now(timezone.utc).isoformat()
        hits: list[RawSearchHit] = []
        for row in reader:
            ref = row.get("ref")
            title = row.get("title") or ref
            if not ref or not title:
                continue
            hits.append(
                RawSearchHit(
                    provider=self.provider,
                    query=query,
                    url=f"https://www.kaggle.com/datasets/{ref}",
                    title=title,
                    snippet=row.get("subtitle") or row.get("description"),
                    raw_payload=row,
                    fetched_at=fetched_at,
                )
            )
            if len(hits) >= self.config.max_results_per_query:
                break

        return hits

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
