# DataCollectionAgent

Minimal interactive DataCollectionAgent with terminal CLI, discovery, planning, and execution.

## How To Run DataCollectionAgent CLI

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Fill in config

Copy one of the example configs and replace the placeholder credentials:

```bash
cp config.example.yaml config.yaml
```

For OpenAI-compatible proxy:

```yaml
llm:
  backend: openai_compatible
  model: google/gemini-3-flash-preview
  api_key: "YOUR_KEY"
  base_url: "http://YOUR_PROXY_HOST:PORT/v1"
  temperature: 0.2
  max_tokens: 2048
```

For Google GenAI proxy:

```yaml
llm:
  backend: google_genai
  model: gemini-3-flash-preview
  api_key: "YOUR_KEY"
  base_url: "http://YOUR_PROXY_HOST:PORT/google/"
  temperature: 0.2
  max_tokens: 2048
```

### 3. Launch the CLI

```bash
python3 run_agent_cli.py --config config.yaml
```

### What discovery uses in MVP / Phase 2

Real internet-backed discovery now uses:
- Hugging Face public dataset search
- GitHub repository search API
- optional Kaggle CLI search

Optional query planning augmentation:
- deterministic provider-specific query planner is always on
- optional LLM query planner can augment provider queries when enabled

Optional but recommended:
- `GITHUB_TOKEN` for higher GitHub rate limits
- one of the following for Kaggle discovery, plus installed `kaggle` CLI:
  - `KAGGLE_API_TOKEN`
  - `KAGGLE_USERNAME` / `KAGGLE_KEY`
  - legacy alias `KAGGLE_NAME` is also accepted for username

Demo fallback is disabled by default in `config.example.yaml`.

### 4. Example terminal dialog

```text
> movie reviews
Какая модальность данных нужна: text, image, audio, video, tabular или другая?
> text
Какой язык данных нужен?
> english
Какой тип задачи нужен: classification, NER, QA, summarization, regression или другой?
> classification
Какой желаемый размер датасета в записях?
> 1000
Нужны ли уже готовые метки?
> yes
Topic profile is complete. Agent is ready to discover sources.
Tip: use /discover, /plans, or /run next.
> /discover
> /plans
> /select 1
> /run
> /artifacts
```

### 5. Supported CLI commands

```text
/status
/discover
/plans
/select 1
/run
/artifacts
/reset
/exit
```

## Smoke Test

Set optional credentials:

```bash
export GITHUB_TOKEN="YOUR_GITHUB_TOKEN"
export KAGGLE_API_TOKEN="YOUR_KAGGLE_ACCESS_TOKEN"
export KAGGLE_USERNAME="YOUR_KAGGLE_USERNAME"
export KAGGLE_KEY="YOUR_KAGGLE_KEY"
```

If you want Kaggle discovery, make sure the CLI is installed from project dependencies:

```bash
pip install -r requirements.txt
which kaggle
```

Run:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
python3 run_agent_cli.py --config config.example.yaml
```

Inside the CLI:

```text
Historical ATP and WTA tennis match data from the last 25 years
tabular
english
classification
100000
no
/discover
```

Expected `/discover` behavior:
- print provider capabilities
- print generated provider-aware queries
- print real Hugging Face / GitHub-backed candidates when providers succeed
- print Kaggle-backed candidates when Kaggle CLI and credentials are available
- print demo fallback candidates only if `allow_demo_fallback: true`

Artifacts are saved under `artifacts/data_collection/...`.

## DataQualityAgent

The second pipeline stage is implemented as `agents/data_quality_agent.py`.

Public API:

```python
from agents.data_quality_agent import DataQualityAgent

agent = DataQualityAgent(config="config.yaml")
report = agent.detect_issues(df)
clean_df = agent.fix(df, {"missing": "median", "duplicates": "drop", "outliers": "clip_iqr"})
comparison = agent.compare(df, clean_df)
```

Human-in-the-loop review workflow:

```python
artifacts = agent.prepare_review_bundle(df)
# edit review/quality_review_decision_template.json and set approved=true
final_df = agent.apply_review_decision(df, "review/quality_review_decision_template.json")
```

Generated quality artifacts:

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

Interactive demo notebook: `notebooks/02_data_quality.ipynb`
