<h1 align="center"> Training Documents Reranker with Search Rubrics for Deep Research Agent </h1>

## ⚡ How to run RubricRanker

### 🛠️ 1 Environment and Preparation

Set up the conda environment and install the agent package:

```bash
cd dr-tulu/agent/
conda create -n agent python=3.10 -y && conda activate agent
uv pip install -e .
```

### 📊 2 Evaluation

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

**c.** Then launch the document reranker:

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
