#!/bin/bash
# Run DR Tulu with each reranker baseline.
#
# Before running:
#   1. Launch the MCP server (port 8000).
#   2. Launch the search-agent vLLM server (port 30001).
#   3. Launch ONE reranker backend at a time on port 30003 (see commands below);
#      monoT5/RankT5 use HF in-process inside the MCP server (no separate vLLM needed).

export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com,.myqcloud.com,.tencentcos.cn"
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

# Use conda environment Python directly
PYTHON_EXEC="/apdcephfs_zwfy4/share_303945702/wenhanliu/conda_envs/agent/bin/python"

SAVE_FOLDER=eval_output/
mkdir -p $SAVE_FOLDER
MODEL=auto_search_sft
YAML_CONFIG=workflows/auto_search_sft.yaml
MAX_CONCURRENT=10
export MCP_MAX_CONCURRENT_CALLS=30
export MAX_CONCURRENT_REQUESTS=20

LLM_DIR=/apdcephfs_zwfy4/share_303945702/wenhanliu/llm
# BGE uses Cohere SDK -> base_url, no /v1 suffix.
RERANKER_API_URL="http://localhost:30003"
# RankVicuna/RankZephyr/Rank4Gen/SETR hit vLLM's /v1/completions or /v1/chat/completions.
RERANKER_API_URL_V1="http://localhost:30003/v1"

# Common reranker overrides (search-side) shared by all baselines.
COMMON="use_browse_agent=true,search_agent_max_tool_calls=10,browse_tool_name=serper,mcp_host=127.0.0.1,number_documents_to_search=30,return_topn=5,reranker_doc_max_words=200,reranker_skip_browse=false,reranker_append_page_content_to_snippet=true"

# Baseline definitions: name -> reranker-specific overrides.
# To launch the corresponding vLLM service, see the comment block above each.
declare -A BASELINES

# 1) BGE-Reranker-Large (550M, vLLM Cohere rerank API)
#    vllm serve $LLM_DIR/bge-reranker-large --port 30003 --task score
BASELINES[bge_reranker_large]="use_reranker=true,reranker_type=pointwise_bge,reranker_model_name=$LLM_DIR/bge-reranker-large,reranker_api_url=$RERANKER_API_URL"

# 2) monoT5-3B (HF in-process, no separate vLLM server)
BASELINES[monot5_3b]="use_reranker=true,reranker_type=pointwise_monot5,reranker_model_name=$LLM_DIR/monot5-3b-msmarco,reranker_hf_device=cuda:5"

# 3) RankT5-3B (HF in-process, no separate vLLM server)
BASELINES[rankt5_3b]="use_reranker=true,reranker_type=pointwise_rankt5,reranker_model_name=$LLM_DIR/RankT5-3b,reranker_hf_device=cuda:5"

# 4) RankVicuna-7B (vLLM raw completions, sliding window 20/10)
#    vllm serve $LLM_DIR/rankvicuna --port 30003 --dtype bfloat16 --max-model-len 4096
BASELINES[rankvicuna_7b]="use_reranker=true,reranker_type=listwise_rankvicuna,reranker_model_name=$LLM_DIR/rankvicuna,reranker_api_url=$RERANKER_API_URL_V1,reranker_window_size=20,reranker_step=10,reranker_listwise_max_tokens=200"

# 5) RankZephyr-7B (vLLM raw completions, sliding window 20/10)
#    vllm serve $LLM_DIR/rankzephyr --port 30003 --dtype bfloat16 --max-model-len 4096
BASELINES[rankzephyr_7b]="use_reranker=true,reranker_type=listwise_rankzephyr,reranker_model_name=$LLM_DIR/rankzephyr,reranker_api_url=$RERANKER_API_URL_V1,reranker_window_size=20,reranker_step=10,reranker_listwise_max_tokens=200"

# 6) Rank4Gen-8B (setwise; no top_n truncation). reranker_top_n is ignored.
#    vllm serve $LLM_DIR/Rank4Gen --port 30003 --dtype bfloat16 --max-model-len 40960 --served-model-name Rank4Gen
BASELINES[rank4gen_8b]="use_reranker=true,reranker_type=setwise_rank4gen,reranker_model_name=Rank4Gen,reranker_api_url=$RERANKER_API_URL_V1,reranker_listwise_max_tokens=4096"

# 7) SETR-8B (setwise; no top_n truncation). reranker_top_n is ignored.
#    vllm serve $LLM_DIR/SetR --port 30003 --dtype bfloat16 --max-model-len 40960 --served-model-name SetR
BASELINES[setr_8b]="use_reranker=true,reranker_type=setwise_setr,reranker_model_name=SetR,reranker_api_url=$RERANKER_API_URL_V1,reranker_listwise_max_tokens=4096"

# 8) RubricRanker-8B (setwise; rubric-aware student distilled from GPT-5.1 labels;
#    no top_n truncation). Sampling: T=0.7, top_p=0.8, top_k=20, min_p=0.
#    vllm serve /apdcephfs_zwfy4/share_303945702/wenhanliu/trained_models/rubricranker_sft \
#        --port 30003 --dtype bfloat16 --max-model-len 40960 --served-model-name rubricranker
BASELINES[rubricranker_8b]="use_reranker=true,reranker_type=setwise_rubricranker,reranker_model_name=rubricranker,reranker_api_url=$RERANKER_API_URL_V1,reranker_rubricranker_prompt_variant=deepresearch"

# -------------------------------- ablation models -------------------------------
# 9) RankQwen-8B (single-pass listwise; relevance-only student distilled from GPT-5.1 ranking;
#    client-side top_n truncation = return_topn). Sampling: T=0.7, top_p=0.8, top_k=20, min_p=0.
#    vllm serve /apdcephfs_zwfy4/share_303945702/wenhanliu/trained_models/relevanceranker_sft/checkpoint-xxx \
#        --port 30003 --dtype bfloat16 --max-model-len 40960 --served-model-name rankqwen
BASELINES[rankqwen_8b]="use_reranker=true,reranker_type=listwise_singlepass,reranker_model_name=rankqwen,reranker_api_url=$RERANKER_API_URL_V1,reranker_listwise_max_tokens=1024"

# Run order. Comment out any line you do not want to run.
RUN_LIST=(
    # bge_reranker_large
    # monot5_3b
    # rankt5_3b
    # rankvicuna_7b
    # rankzephyr_7b
    # setr_8b
    # rank4gen_8b
    rubricranker_8b
# -------------- ablation models -------------------------------
    # rankqwen_8b
)

for baseline in "${RUN_LIST[@]}"; do
    for task in healthbench deep_research_bench researchqa webwalker; do
        if [ "$task" = "webwalker" ]; then
            num_examples=200
        else
            num_examples=1
        fi

        echo "===== Running $MODEL on $task with reranker baseline: $baseline ====="
        overrides="$COMMON,${BASELINES[$baseline]}"
        output_name=reranker-${baseline}

        "$PYTHON_EXEC" -m workflows.$MODEL \
            generate-dataset $task \
            --num-examples $num_examples \
            --max-concurrent $MAX_CONCURRENT \
            --batch-size $MAX_CONCURRENT \
            --use-cache \
            --config $YAML_CONFIG \
            --config-overrides "$overrides" \
            --output $SAVE_FOLDER/$MODEL/$task/${output_name}.jsonl

        "$PYTHON_EXEC" scripts/evaluate.py $task eval_output/auto_search_sft/$task.jsonl
        
        done
    done
