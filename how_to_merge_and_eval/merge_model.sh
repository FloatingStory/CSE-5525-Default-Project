#!/bin/bash
#SBATCH --job-name=SBATCHmergemodeljob1    # Job name
#SBATCH --output=SBATCHmergemodel_job_output.%j.out # Standard output and error log
#SBATCH --time=03:00:00         # Wall clock limit (4 hours)
#SBATCH --nodes=1               # Request one node
#SBATCH --gpus=1               # Request gpu
#SBATCH --ntasks=1              # Run a single task
#SBATCH --mail-type=END         # Receive email when job ends
#SBATCH -A PAS3272
#SBATCH --mem=64G

export HF_TOKEN=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
source .venv_for_merging_model_and_checkpoints/bin/activate

python3 merge_model_and_checkpoint.py 
