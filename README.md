<h1 align="center"> Training Documents Reranker with Search Rubrics for Deep Research Agent </h1>

<div align="center">
<a href="https://modelscope.cn/models/lwhlwh/rubricranker_sft_rl" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Model-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green.svg"></a>
<a href="https://www.python.org/downloads/release/python-3100/"><img alt="Static Badge" src="https://img.shields.io/badge/Python-3.10+-blue.svg"></a>
</div>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub.</h5>

## 📣 Latest News

- **[Jun 24, 2026]**: 🚀 We released our full codebase, **[model](https://modelscope.cn/models/lwhlwh/rubricranker_sft_rl)**, **[SFT data](https://modelscope.cn/datasets/lwhlwh/rubricranker_sft_data)** and **[RL data](https://modelscope.cn/datasets/lwhlwh/rubricranker_rl_data)** of RubricRanker.

## 📋 Table of Contents

- [1. How to run RubricRanker](#1-how-to-run-rubricranker)
  - [1.1 Environment and Preparation](#11-environment-and-preparation)
  - [1.2 Evaluation](#12-evaluation)
- [2. How to train RubricRanker](#2-how-to-train-rubricranker)
  - [2.1 SFT](#21-sft)
  - [2.2 RL](#22-rl)

## ⚡ 1. How to run RubricRanker

### 🛠️ 1.1 Environment and Preparation

Set up the conda environment and install the agent package:

```bash
cd dr-tulu/agent/
conda create -n agent python=3.10 -y && conda activate agent
uv pip install -e .
```

### 📊 1.2 Evaluation

**a.** Launch the DR-Tulu agent (the deep research agent backbone):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve rl-research/DR-Tulu-8B \
  --dtype auto \
  --port 30001 \
  --max-model-len 40960 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8
```

**b.** Launch the webpage summarization model (following DR-Tulu):

```bash
CUDA_VISIBLE_DEVICES=4,5 \
vllm serve Qwen/Qwen3-8B \
  --dtype auto \
  --port 30002 \
  --max-model-len 40960 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8
```

**c.** Download our RubricRanker from [rubricranker_sft_rl](https://modelscope.cn/models/lwhlwh/rubricranker_sft_rl) and place it under your local path `{RUBRICRANKER_PATH}`. Then launch the document reranker:

```bash
CUDA_VISIBLE_DEVICES=6,7 \
vllm serve {RUBRICRANKER_PATH} \
  --port 30003 \
  --dtype bfloat16 \
  --served-model-name rubricranker \
  --max-model-len 40960 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8
```

**d.** Start the MCP service:

```bash
cd dr-tulu/agent
python -m dr_agent.mcp_backend.main --transport streamable-http --host 0.0.0.0 --port 8000
```

**e.** Configure the search API and run inference:

```bash
cd dr-tulu/agent
# Configure SERPER_API_KEY in the .env file to use serper for web search.
# Modify PYTHON_EXEC in inference.sh to point to the python executable of your conda environment.
bash inference.sh
```

## 🔥 2. How to train RubricRanker

### ❄️ 2.1 SFT

First, install LLaMA-Factory by following the official instructions in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We recommend using a fresh, dedicated conda environment for SFT.

Before training, please download our SFT data `rubricranker_sft-data.json` from [rubricranker_sft_data](https://modelscope.cn/datasets/lwhlwh/rubricranker_sft_data/files) and place it under `LLaMA-Factory/data/` (the file is too large to be hosted on GitHub):

```
LLaMA-Factory/data/rubricranker_sft-data.json
```

Then run the following commands to start supervised fine-tuning:

```bash
cd LLaMA-Factory
bash run_train.sh
```

### 🎯 2.2 RL

First, install verl by following the official documentation at [verl](https://verl.readthedocs.io/en/latest/). We recommend using a fresh, dedicated conda environment for RL training.

Before training, please download our RL data `train.parquet` and `val.parquet` from [rubricranker_rl_data](https://modelscope.cn/datasets/lwhlwh/rubricranker_rl_data/files) and place them under `verl/data/` (the files are too large to be hosted on GitHub):

```
verl/data/train.parquet
verl/data/val.parquet
```

Our rubric-based reward (in `verl/verl/utils/reward_score/set_selection.py`) calls the OpenAI API (default model: `gpt-5.1`) to score set-level rubrics during rollout. Before training, please also make sure you have filled in your OpenAI API key in `dr-tulu/agent/.env`:

```bash
# dr-tulu/agent/.env
OpenAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

Then run the following commands to start GRPO training:

```bash
cd verl/
bash train_grpo.sh
```

Remember to change `YOUR_SFT_MODEL` and `YOUR_RL_MODEL` in `train_grpo.sh` to the paths of your SFT model and the RL model save directory, respectively.

## 📄 License

This project is released under the [MIT License](LICENSE).

## 📞 Contact

For any questions or feedback, please reach out to us at [lwh@ruc.edu.cn](lwh@ruc.edu.cn).
