## Introduction

We evaluate on SQA-CS-V2 (Scientific Question Answering with Structured Citations) using the [astabench](https://pypi.org/project/astabench/) evaluation framework.

SQA-CS-V2 measures both answer quality and citation quality:
- **ingredient_recall**: Coverage of key facts from the rubric
- **answer_precision**: Avoidance of incorrect statements
- **citation_recall / citation_precision**: Quality of supporting citations
- **global_avg**: Overall weighted score

---

## Self-Contained Evaluation (Recommended)

The self-contained script `run_eval.py` handles format conversion and evaluation. **No external repo needed.**

### Prerequisites

```bash
pip install uv  # Recommended: auto-manages Python 3.11 + astabench
export GOOGLE_API_KEY="your_google_api_key_here"
export HF_TOKEN="your_hf_token_here"   # Needs access to allenai/asta-bench (gated dataset)
```

> **Note**: `astabench` requires Python ≥3.11. The script auto-manages this via `uv run` if your current Python is older. Alternatively, install directly: `pip install astabench==0.3.1 inspect_ai datasets` (requires Python ≥3.11).

### Quick Start

```bash
# Full pipeline: convert + evaluate
python evaluation/sqa_eval/run_eval.py run \
    --input_file eval_output/auto_search_sft/sqa_cs_v2.jsonl

# Step-by-step:
# 1. Convert DR Tulu output to ASTA format
python evaluation/sqa_eval/run_eval.py convert \
    --input_file eval_output/auto_search_sft/sqa_cs_v2.jsonl

# 2. Run evaluation
python evaluation/sqa_eval/run_eval.py eval \
    --input_file eval_output/auto_search_sft/sqa_cs_v2_asta_format.jsonl

# With custom scorer model and connections:
python evaluation/sqa_eval/run_eval.py run \
    --input_file eval_output/auto_search_sft/sqa_cs_v2.jsonl \
    --scorer_model "google/gemini-2.5-flash" \
    --max_connections 16
```

### Via the unified evaluate.py

```bash
python scripts/evaluate.py sqa_cs_v2 eval_output/auto_search_sft/sqa_cs_v2.jsonl
```

### SQA Response Format

SQA-CS-V2 requires responses in a specific JSON format with structured sections and citations:

```json
{
  "sections": [
    {
      "text": "text of section 1",
      "citations": [
        {
          "id": "[cite_id]",
          "title": "paper title",
          "snippets": ["evidence 1", "evidence 2"]
        }
      ]
    }
  ]
}
```

### Example Test Data

Example DR-Tulu outputs and their ASTA-format conversions are available at [`rl-research/dr-tulu-eval-data`](https://huggingface.co/datasets/rl-research/dr-tulu-eval-data) under `sqa/`.

---

## Legacy Method (External Repo)

If you prefer using the original [agent-baselines](https://github.com/allenai/agent-baselines) repository:

1. **Convert DR Tulu outputs to SQA format**:
   ```bash
   python evaluation/sqa_eval/convert_to_asta_format.py --folder <folder_name> --file <file_name>
   ```

2. **Clone the evaluation repository**:
   ```bash
   git clone https://github.com/allenai/agent-baselines
   cd agent-baselines
   ```

3. **Run evaluation**:
   ```bash
   uv run --extra sqa inspect eval astabench/sqa --display plain \
     --solver agent_baselines/solvers/sqa/debug/cached_solver.py \
     -S path=<outputfile_from_step1> \
     -T split=test \
     -T with_search_tools=False \
     -T simplified_eval=true \
     -T assess_jointly=true \
     --max-connections 16 \
     -T sentence_wise_cit_eval=false \
     -T all_at_once=true \
     -T scorer_model="google/gemini-2.5-flash"
   ```

**Note**: Export `GOOGLE_API_KEY` and `HF_TOKEN` before running.

