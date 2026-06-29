# Evaluation Scripts

This directory contains evaluation scripts for benchmarking DR agents across various tasks, from short-form QA to long-form deep research.

---

## Available Benchmarks

| Benchmark | Type | Description | Benchmark Name |
|-----------|------|-------------|----------------|
| **SQA-CS-V2** | Long-form | Scientific question answering with structured citations | `sqa_cs_v2` |
| **Deep Research Bench** | Long-form | Deep research reports (RACE & FACT metrics) | `deep_research_bench` |
| **ResearchQA** | Long-form | Research question answering with coverage metrics | `research_qa` |
| **HealthBench** | Long-form | Medical QA with physician-level rubrics | `healthbench` |
| **Genetic Diseases** | Domain-specific | Clinical genetics questions | `genetic_diseases` |
| **SimpleQA** | Short-form | Factuality in short-form answers | `simpleqa` |
| **Short Form QA** | Short-form | Multi-dataset QA framework (14+ datasets) | See supported tasks below |

---

## Running Evaluations

### Example: Evaluate Across All Benchmarks

**Prerequisites**: Before running the evaluation script, launch the required servers **on the same node**:

```bash
# Launch VLLM servers (requires 2 GPUs)
CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --dtype auto --port 30001 --max-model-len 40960
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --dtype auto --port 30002 --max-model-len 40960

# Launch MCP server
python -m dr_agent.mcp_backend.main --port 8000
```

Then run the evaluation script:

```bash
#!/bin/bash
# Example script to run DR Tulu on multiple benchmarks

SAVE_FOLDER=eval_output/
MODEL=auto_search_sft
YAML_CONFIG=workflows/auto_search_sft.yaml
MAX_CONCURRENT=20

mkdir -p $SAVE_FOLDER

# Run evaluations on all benchmarks
for task in healthbench deep_research_bench research_qa genetic_diseases simpleqa 2wiki webwalker; do 
    echo "Running $MODEL on $task"
    python workflows/$MODEL.py \
        generate-dataset $task \
        --num-examples final_run \
        --max-concurrent $MAX_CONCURRENT \
        --batch-size $MAX_CONCURRENT \
        --use-cache \
        --config $YAML_CONFIG \
        --config-overrides "use_browse_agent=true,search_agent_max_tool_calls=10,browse_tool_name=jina" \
        --output $SAVE_FOLDER/$MODEL/$task.jsonl
    
    python scripts/evaluate.py $task $SAVE_FOLDER/$MODEL/$task.jsonl
done
```

For SQA-CS-V2 and Deep Research Bench evaluations, see the dedicated READMEs in their subdirectories.

---

## Deep Research Bench (DRB) Evaluation

See [`deep_research_bench_eval/README.md`](deep_research_bench_eval/README.md) for full instructions.

```bash
# Quick start (self-contained, no external repo needed)
pip install google-genai tqdm huggingface_hub
export GEMINI_API_KEY="your_key"

python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model

# Or via unified script:
python scripts/evaluate.py deep_research_bench eval_output/auto_search_sft/deep_research_bench.jsonl
```

---

## SQA-CS-V2 Evaluation

See [`sqa_eval/README.md`](sqa_eval/README.md) for full instructions.

```bash
# Quick start (self-contained, no external repo needed)
pip install uv
export GOOGLE_API_KEY="your_key"

python evaluation/sqa_eval/run_eval.py run \
    --input_file eval_output/auto_search_sft/sqa_cs_v2.jsonl

# Or via unified script:
python scripts/evaluate.py sqa_cs_v2 eval_output/auto_search_sft/sqa_cs_v2.jsonl
```

