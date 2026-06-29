backend="fsdp"
save_dir=/apdcephfs_zwfy4/share_303945702/wenhanliu/trained_models

model_name=rubricranker_qwen3_rl
step=100

checkpoint_dir=${save_dir}/${model_name}/global_step_${step}/actor
# hf_model_path=$2
target_dir=${save_dir}/${model_name}/step_${step}


python /apdcephfs_zwfy4/share_303945702/wenhanliu/searchrubrics/verl/scripts/legacy_model_merger.py merge \
    --backend $backend \
    --local_dir $checkpoint_dir \
    --target_dir $target_dir
