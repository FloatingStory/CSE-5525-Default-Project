#!/bin/bash
#SBATCH --job-name=OLMESEVAL_NEWESTOLMES2modeljob1    # Job name
#SBATCH --output=OLMESEVAL_NEWESTOLMES2modelLlama0_job_output.%j.out # Standard output and error log
#SBATCH --time=07:00:00         # Wall clock limit (7 hours)
#SBATCH --nodes=1               # Request one node
#SBATCH --gpus=1               # Request gpu
#SBATCH --ntasks=1              # Run a single task
#SBATCH --mail-type=END         # Receive email when job ends
#SBATCH -A PAS3272
#SBATCH --mem=64G

export HF_TOKEN=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
#source .venv_for_olmes_eval/bin/activate
source /users/PAS2930/cwang/CSE5525DefaultProject/CSE-5525-Default-Project/evals/olmes/.venv/bin/activate

cd evals

bash for_sarah_sbatch_run_eval.sh
