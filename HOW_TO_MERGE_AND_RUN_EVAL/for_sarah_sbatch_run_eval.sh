#!/bin/bash

#This part is need for OSC users
export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/fs/scratch/PAS3272/${USER}/triton_cache


export UV_CACHE_DIR=/fs/scratch/PAS3272/${USER}/.cache/uv  #control your uv caches

# Dummy key to prevent import error in safety-eval (WildGuard doesn't actually use it)
export OPENAI_API_KEY="sk-dummy-not-used"

# Disable vLLM V1 multiprocessing so EngineCore runs inline in the spawned subprocess
# rather than forking a grandchild process that loses CUDA visibility on SLURM
export VLLM_ENABLE_V1_MULTIPROCESSING=0


cd olmes/oe_eval/dependencies/safety
bash install.sh

#pip install vllm

dataset_name=(
    "gsm8k"
    "mbpp"
    "ifeval"
    "harmbench::default"
    "xstest::default"
    "codex_humaneval"           #for code generation extension
    "codex_humanevalplus"
)
# model_path=allenai/OLMo-2-0425-1B-SFT
#model_path=allenai/OLMo-2-0425-1B-SFT
#model_path=/users/PAS2930/cwang/CSE5525DefaultProject/CSE-5525-Default-Project/OSC_LLAMA_BASEMODEL_BASE
model_path=/users/PAS2930/cwang/CSE5525DefaultProject/CSE-5525-Default-Project/MERGED_SFT_MODEL_10K_mix 
output_path=/users/PAS2930/cwang/CSE5525DefaultProject/CSE-5525-Default-Project/SFTMODEL_10K_MIX_APRIL12_OLMESEVALS

for dataset in "${dataset_name[@]}"; do
    echo "Evaluating on ${dataset}..."
    if [ "${dataset}" == "gsm8k" ]; then
        num_shots=8
    elif [ "${dataset}" == "codex_humaneval" ]; then
        num_shots=3
    else
        num_shots=0
    fi
    uv run olmes \
        --model ${model_path} \
        --task ${dataset} \
        --output-dir $output_path-eval-${dataset} \
        --num-shots ${num_shots} \
        --batch-size 1 \
        --gpus 1 \
        --num-workers 1
done
