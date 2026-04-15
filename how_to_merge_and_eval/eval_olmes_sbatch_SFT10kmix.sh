#!/bin/bash
#SBATCH --job-name=10kmixSBATCHmodeljob1    # Job name
#SBATCH --output=10kmixSBATCHmodelLlama0_job_output.%j.out # Standard output and error log
#SBATCH --time=6:00:00         # Wall clock limit (4 hours)
#SBATCH --nodes=1               # Request one node
#SBATCH --gpus=1               # Request gpu
#SBATCH --ntasks=1              # Run a single task
#SBATCH --mail-type=END         # Receive email when job ends
#SBATCH --mail-user=lowkeyloki923@gmail.com # Email address
#SBATCH -A PAS3272
#SBATCH --mem=64G

source /users/PAS2930/cwang/CSE5525DefaultProject/CSE-5525-Default-Project/evals/olmes/.venv/bin/activate
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export VLLM_DISABLE_CUDA_GRAPH=1
#export VLLM_USE_TORCH_COMPILE=0
#export VLLM_USE_CUDAGRAPH=0
#export VLLM_ENABLE_PREFIX_CACHING=0
#export VLLM_ENABLE_CHUNKED_PREFILL=0
#export TORCHINDUCTOR_TRITON_CUDAGRAPH_SKIP_DYNAMIC_GRAPHS=1
#export TORCH_LOGS=recompiles" 
export HF_TOKEN=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

cd evals

bash for_sarah_sbatch_sft2.sh
