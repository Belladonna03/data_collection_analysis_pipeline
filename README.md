# Data Collection and Analysis Pipeline

Unified pipeline for **data collection, quality control, annotation, active learning, training, and reporting**.

The main entrypoint is **`run_pipeline.py`**.  
Legacy REPL-based collection is kept only for debugging and is **not** the recommended workflow.

---

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Canonical CLI workflow](#canonical-cli-workflow)
- [Collection stage](#collection-stage)
- [Label Studio integration](#label-studio-integration)
- [Merged dataset schema](#merged-dataset-schema)
- [Discovery and execution capabilities](#discovery-and-execution-capabilities)
- [Executable connectors](#executable-connectors)
- [Credentials](#credentials)
- [DataQualityAgent](#dataqualityagent)
- [Pipeline run, review, and resume](#pipeline-run-review-and-resume)
- [Artifact layout](#artifact-layout)
- [Smoke tests](#smoke-tests)
- [Notes and limitations](#notes-and-limitations)

---

## Overview

This repository provides a **unified CLI** for end-to-end work:

**collect → quality → annotate → active learning → train → report**

Main commands:

```bash
python run_pipeline.py --help
python run_pipeline.py collect --help
```

Top-level commands:

- `collect`
- `quality`
- `annotate`
- `al`
- `train`
- `report`

Each stage supports `status`, `run`, and `review` where applicable.

For backward compatibility, the alias below is still supported:

```bash
python run_pipeline.py stage <name> ...
```

However, all examples in this README use the **modern CLI syntax**.

---

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

The default runnable config in the repository root is:

```text
config.yaml
```

By default, it is configured for **health forum text classification** in English with single-level labels from `annotation.labels`.

You can either:

- edit `config.yaml` directly,
- copy it into a local working config,
- or start from templates in `config/examples/`.

Example:

```bash
cp config.yaml config.local.yaml
```

Available example configs include:

- `config/examples/safety_classification.example.yaml`

### OpenAI-compatible proxy example

```yaml
llm:
  backend: openai_compatible
  model: google/gemini-3-flash-preview
  api_key: "YOUR_KEY"
  base_url: "http://YOUR_PROXY_HOST:PORT/v1"
  temperature: 0.2
  max_tokens: 2048
```

### Google GenAI proxy example

```yaml
llm:
  backend: google_genai
  model: gemini-3-flash-preview
  api_key: "YOUR_KEY"
  base_url: "http://YOUR_PROXY_HOST:PORT/google/"
  temperature: 0.2
  max_tokens: 2048
```

The pipeline loads `.env` from:

- the config directory,
- and the current working directory,

when `python-dotenv` is installed.

### `collection.defaults`

Optional block used by `build_topic_profile` / `infer_topic_profile` to fill missing topic fields.

| Field | Description |
| --- | --- |
| `modality` | Data modality when not inferred from topic |
| `language` | Preferred language when not inferred |
| `task_type` | Task hint such as `classification` or `analysis` |
| `size_target` | Desired number of rows |
| `needs_labels` | Whether labeled data is required |

---

## Canonical CLI workflow

Recommended workflow for coursework, demos, and reproducibility:

```bash
python run_pipeline.py reset
python run_pipeline.py collect discover --config config.yaml
python run_pipeline.py collect recommend
python run_pipeline.py collect select --ids 1,2
python run_pipeline.py collect run --config config.yaml
python run_pipeline.py quality run
python run_pipeline.py annotate run
python run_pipeline.py al run
python run_pipeline.py train run
python run_pipeline.py report run
python run_pipeline.py artifacts
```

Check collection status at any time:

```bash
python run_pipeline.py collect status
```

---

## Collection stage

The collection flow is snapshot-based and reproducible.

### `collect discover`

Builds a topic profile from:

- `--topic`,
- config defaults,
- and inferred metadata.

It then runs discovery against enabled providers and writes:

```text
01_collect/discovery/discovery_snapshot.json
```

This snapshot contains stable **candidate numbers** and **candidate keys**.

### `collect recommend`

Reads the saved snapshot, runs the planner, and writes:

```text
01_collect/recommendations/recommendations.json
```

### `collect select --ids ...`

Validates selected source IDs against the snapshot and stores the chosen plan in:

```text
01_collect/selection/user_selection.json
```

### `collect run`

Executes the selected plan, merges normalized data, and writes final collection artifacts such as:

- merged parquet dataset,
- data card,
- EDA summary,
- plots,
- source summary.

### After `collect run`

Expected outputs under:

```text
artifacts/runs/<run_id>/01_collect/
```

Typical artifacts include:

- `data/merged_dataset.parquet`
- `reports/data_card.md`
- `reports/eda_summary.md`
- `source_summary.json`
- EDA plots directory

`pipeline_state.json` also stores pointers such as:

- `collect_merged_dataset_parquet`
- `collect_data_card_md`
- `collect_eda_summary_md`
- `collect_source_summary_json`
- `collect_eda_plots_dir`

---

## Label Studio integration

For text classification HITL, the pipeline can export the review queue into Label Studio and import manual annotations back.

Enable in `config.yaml`:

```yaml
label_studio:
  enabled: true
```

Default output filenames:

- `labelstudio_import.json`
- `label_config.xml`

Legacy copies are also written:

- `review_queue_labelstudio.json`
- `labelstudio_config.xml`

### Run Label Studio locally

```bash
docker run -it -p 8080:8080 -v $(pwd)/labelstudio-data:/label-studio/data heartexlabs/label-studio:latest
```

Open:

```text
http://localhost:8080
```

### Import tasks

After:

```bash
python run_pipeline.py annotate run
```

go to:

```text
artifacts/runs/<run_id>/03_annotate/review/
```

Then in Label Studio:

1. Import `labelstudio_import.json`
2. Open **Settings → Labeling Interface → Code**
3. Paste the contents of `label_config.xml`
4. Save

The interface must define a single `Choices` control named `label` tied to `Text name="text"`.

### Export labels back into the pipeline

After manual labeling, export tasks as **JSON** and apply them with:

```bash
python run_pipeline.py annotate review --run-id <run_id> --file /path/to/label-studio-export.json
```

The pipeline maps rows by:

- `data.annotation_id`
- `data.source_id`
- or top-level task `id`

If `annotation.labelstudio_strict_labels: true` is enabled, any label absent from `annotation.labels` will raise an error.

---

## Merged dataset schema

After normalization and merge, the final dataframe follows a **unified text contract**.

Implementation reference:

```text
agents/data_collection/text_unified_schema.py
agents/data_collection/canonical_sample.py
```

### Canonical normalization order

For each source, normalization uses structure-based routing:

1. non-empty `text`
2. non-empty `messages`
3. first non-empty instruction-like field among:
   - `text`
   - `instructions`
   - `prompt`
   - `input`
   - `question`
4. fallback title/body composition

Rows with empty final `text` are dropped.

### Typical chat drop reasons

Examples include:

- `malformed_messages`
- `missing_messages`
- `unsupported_message_schema`
- `empty_user_content`
- `empty_instruction`

Raw fields are moved into `metadata` as `raw_*`.

### Required columns

Required columns are ordered first; extra columns are preserved.

| Column | Description |
| --- | --- |
| `id` | Stable row id |
| `text` | Primary document text |
| `target_text` | Optional assistant / gold output |
| `title` / `body` | Optional structured fields |
| `label` | Optional label |
| `source` | Human-readable source name |
| `source_type` | Connector type |
| `source_url` | Canonical URL if available |
| `collected_at` | ISO timestamp |
| `metadata` | UTF-8 JSON object |

### Merge behavior

Merge uses `pandas.concat` with schema alignment.

After concatenation, it:

- drops rows with empty `text`,
- removes exact duplicates by `record_hash`,
- otherwise falls back to `text + source + source_id`.

Validation adds warnings into `merged:validation`.

---

## Discovery and execution capabilities

### Discovery providers

Configured under:

```yaml
discovery:
  providers:
```

Discovery providers only **find** `SourceCandidate` rows.  
They do **not** fetch data themselves.

Supported discovery directions include:

- Hugging Face public dataset search
- GitHub repository search API
- optional Kaggle CLI search
- `web_forum` discovery via DuckDuckGo Lite HTML search
- `devtools_har` hints based on copied browser network metadata

Example:

```yaml
discovery:
  providers:
    web_forum:
      enabled: true
      deny_domains: ["facebook.com"]
    devtools_har:
      enabled: true
      hints:
        - label: my_forum_api
          page_url: "https://example.com/forum"
          json_url: "https://api.example.com/v1/threads"
          method: "GET"
          headers:
            Accept: "application/json"
```

If `discovery.strict_provider_check: true` is enabled, you can also require specific providers.

### Runtime scraping note

For structured scraping, the runtime source of truth is the JSON `scraper_spec` attached to the source.

`ScrapeConnector` uses this spec directly.  
Generated Python scraper files are only **debug/demo artifacts**, not the primary execution path.

---

## Executable connectors

Execution connectors actually run the selected collection plan.

Supported connectors include:

- `hf_dataset`
- `kaggle`
- `github_dataset`
- `http_file`
- `api`
- `scrape`

### Hugging Face

Uses `datasets.load_dataset(...)`.

Example:

```python
SourceSpec(
    id="hf_dataset:imdb",
    type=SourceType.HF_DATASET,
    name="imdb",
    dataset_id="imdb",
    split="train",
    sample_size=200,
    field_map={"text": "text", "label": "label"},
    label_map={0: "negative", 1: "positive"},
)
```

### Kaggle

Downloads the dataset bundle, selects the most likely primary tabular file, then parses it.

Supports:

- `csv`
- `tsv`
- `json`
- `parquet`
- `xls`
- `xlsx`
- zip archives containing those files

Example:

```python
SourceSpec(
    id="kaggle:tennis",
    type=SourceType.KAGGLE,
    name="hakeem/atp-and-wta-tennis-data",
    dataset_ref="hakeem/atp-and-wta-tennis-data",
    files=["atp_matches.csv"],
    field_map={"winner_name": "winner_name", "loser_name": "loser_name"},
)
```

### GitHub

Scans a public repository through the GitHub contents API, detects dataset-like files, downloads, and parses them.

Example:

```python
SourceSpec(
    id="github_dataset:tennis_atp",
    type=SourceType.GITHUB_DATASET,
    name="jeffsackmann/tennis_atp",
    repo_url="https://github.com/jeffsackmann/tennis_atp",
    file_patterns=["atp_matches_2024.csv", "atp_rankings_2024.csv"],
)
```

### HTTP file

Downloads a public direct file URL and parses it by extension.

Example:

```python
SourceSpec(
    id="http_file:tennis_csv",
    type=SourceType.HTTP_FILE,
    name="tennis matches csv",
    url="https://example.com/data/tennis_matches.csv",
)
```

---

## Credentials

### Required

Kaggle execution requires either:

- `KAGGLE_USERNAME` + `KAGGLE_KEY`
- or `~/.kaggle/kaggle.json`

### Optional

- `HF_TOKEN` for private or gated Hugging Face datasets
- `GITHUB_TOKEN` for higher GitHub API rate limits

### Example `.env`

```dotenv
HF_TOKEN=
GITHUB_TOKEN=
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

### Example config snippet

```yaml
connectors:
  hf_dataset:
    token: null
  kaggle:
    download_dir: null
  github_dataset:
    timeout: 30
    token: null
  http_file:
    timeout: 30
```

---

## DataQualityAgent

The second pipeline stage is implemented in:

```text
agents/data_quality_agent.py
```

Additional documentation:

```text
docs/data_quality_pipeline.md
```

### Public API

```python
from agents.data_quality_agent import DataQualityAgent, HumanReviewRequired

agent = DataQualityAgent(config="config.yaml")
report = agent.detect_issues(df)
clean_df = agent.fix(df, {"missing": "median", "duplicates": "drop", "outliers": "clip_iqr"})
comparison = agent.compare(df, clean_df)
```

### Supported checks

Text and health-forum checks include:

- empty or too-short text
- regex-based PII detection
- near-duplicate detection
- optional language mismatch detection via `langdetect`

### `fix(..., strategy)` supports

- missing-value handling
- duplicate dropping
- outlier clipping
- nested `text_quality` strategy options
- convenience aliases such as:
  - `redact_basic_pii`
  - `drop_empty_text`
  - `drop_short_text`
  - `drop_near_duplicates`
  - `language_filter`

### HITL pipeline mode

```python
try:
    cleaned = agent.run_stage(df, task_description="...")
except HumanReviewRequired:
    ...
cleaned = agent.run_stage(df, decision={"approved": True, "selected_preview": "conservative"})
```

### Review bundle workflow

```python
artifacts = agent.prepare_review_bundle(df)
final_df = agent.apply_review_decision(df, "review/quality_review_decision_template.json")
```

### Generated artifacts

- `reports/quality/quality_report.json`
- `reports/quality/quality_report.md`
- `reports/quality/comparison_report.json`
- `reports/quality/comparison_report.md`
- `reports/quality/plots/*.png`
- `data/interim/cleaned_preview_conservative.parquet`
- `data/interim/cleaned_preview_strict.parquet`
- `data/interim/cleaned_final.parquet`
- `review/quality_review_bundle.md`
- `review/quality_review_decision_template.json`
- `review/quality_stage_status.json`

After approved review:

- `reports/quality/final_*`
- `review/final_row_actions.parquet`
- `review/final_removed_rows.parquet`
- `review/final_review_decision_applied.json`

Notebook demo:

```text
notebooks/02_data_quality.ipynb
```

---

## Pipeline run, review, and resume

### Automated run

```bash
python run_pipeline.py run --config config.yaml
```

This advances stages until the pipeline completes or pauses for human review.

### Review flow

If the pipeline stops:

- **QUALITY**: edit  
  `artifacts/runs/<run_id>/02_quality/review/quality_review_decision_template.json`
- **ANNOTATE / AL**: fill review CSVs in the stage `review/` folder

Then continue with:

```bash
python run_pipeline.py quality review --decision artifacts/runs/<run_id>/02_quality/review/quality_review_decision_template.json
python run_pipeline.py resume --run-id <run_id>
```

### Status and artifact inspection

```bash
python run_pipeline.py status
python run_pipeline.py status --run-id <run_id>
python run_pipeline.py resume --run-id <run_id>
python run_pipeline.py artifacts --run-id <run_id>
```

---

## Artifact layout

All run outputs are stored under:

```text
artifacts/runs/<run_id>/
  pipeline_state.json
  01_collect/
  02_quality/
  03_annotate/
  04_al/
  05_train/
    models/
    reports/
  06_report/
    final_pipeline_report.md
```

---

## Internal debug REPL

This command exists only for debugging `DataCollectionAgent`:

```bash
python run_agent_cli.py --config config.yaml
```

It is an internal interactive shell and **not** the recommended main workflow.

For demos, reports, and assignments, use:

```bash
python run_pipeline.py collect ...
```

---

## Notes and limitations

Current MVP limitations:

- GitHub execution works only for public repositories exposed by the contents API
- Kaggle execution downloads the dataset bundle first, then heuristically chooses the main tabular file unless `files` is specified
- HTTP execution expects direct file URLs, not arbitrary HTML landing pages
- connector-level file detection is heuristic and may still require explicit `files` or `file_patterns`

Common expected CLI behavior:

- if no run exists, the CLI explains how to start one
- if collection was discovered but not selected, `collect run` explains the missing steps
- if a review file is missing, the CLI prints the path and exits non-zero
- if nothing is pending, `resume` reports that there is nothing to resume

---

## Recommended demo sequence

```bash
python run_pipeline.py run --config config.yaml
python run_pipeline.py status
python run_pipeline.py artifacts
python run_pipeline.py resume --run-id <run_id>
python run_pipeline.py report status --run-id <run_id>
```
