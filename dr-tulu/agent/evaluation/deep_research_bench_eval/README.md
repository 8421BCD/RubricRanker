## Introduction

We evaluate on Deep Research Bench (DRB) to acquire results for two metrics:
1. **RACE** (Reference-based and Adaptive Criteria-driven Evaluation framework with Dynamic Weighting) — article quality
2. **FACT** (Factual Abundance and Citation Trustworthiness) — citation verification

---

## Self-Contained Evaluation (Recommended)

The self-contained script `run_eval.py` bundles all evaluation code and data. **No external repo needed.**

### Prerequisites

```bash
pip install google-genai tqdm huggingface_hub
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### Quick Start

```bash
# Full pipeline: format conversion + RACE + FACT
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model

# RACE only:
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model --skip_fact

# FACT only:
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model --skip_race

# Test with limit:
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model --limit 2

# English only / Chinese only:
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model --only_en
```

### Via the unified evaluate.py

```bash
python scripts/evaluate.py deep_research_bench eval_output/auto_search_sft/deep_research_bench.jsonl
```

### Output Structure

```
<output_dir>/
├── raw_data/<task_name>.jsonl          # Formatted articles
├── cleaned_data/<task_name>.jsonl      # Cleaned articles (citations removed)
├── race/<task_name>/
│   ├── raw_results.jsonl               # Per-item RACE scores
│   └── race_result.txt                 # Aggregated RACE metrics
└── fact/<task_name>/
    ├── scraped.jsonl                   # Articles with scraped citation content
    ├── validated.jsonl                 # Validated citations
    └── fact_result.txt                 # Aggregated FACT metrics
```

### Evaluation Data

The following data files are hosted at [`rl-research/dr-tulu-eval-data`](https://huggingface.co/datasets/rl-research/dr-tulu-eval-data) and **auto-downloaded on first run**:
- `query.jsonl` — 100 evaluation queries (50 EN + 50 ZH)
- `criteria.jsonl` — Task-specific evaluation criteria with weights
- `reference.jsonl` — Reference articles for comparison (from the original DRB repo)

---

## Legacy Method (External Repo)

If you prefer using the [original DRB repository](https://github.com/Ayanami0730/deep_research_bench):

### 1. Generate the DRB output from our system
Add `deep_research_bench` to the task in the scripts, e.g., `agent/scripts/auto_search.sh`.

### 2. Format conversion
```bash
python drb_formatter.py \
    --input_file_path /path/to/drb-ablation-s2.jsonl \
    --task_name drb-ablation-s2 \
    --drb_repo_path /path/to/deep_research_bench
```

### 3. Set up the external repo

```bash
git clone https://github.com/Ayanami0730/deep_research_bench
cd deep_research_bench
conda create -n drb python=3.9
conda activate drb
pip install -r requirements.txt
```

🚨 **Crucial**: Change `FACT_Moedel` in `utils/api.py` to `gemini-2.5-flash-preview-09-2025`.

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export JINA_API_KEY="your_jina_api_key_here"
```

### 4. Run evaluation
Copy `run_benchmark_scraped.sh` to the DRB repo root, edit `TARGET_MODELS`, and run:
```bash
bash run_benchmark_scraped.sh
```
Results: RACE in `output_<task>.log`, FACT in `results/fact/<task>/fact_result.txt`.