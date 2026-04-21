Works with python 3.11.13.

NOTE THAT: DATASETS GSM8K, IFEVAL, AND MBPP RUN FINE ON ALL OSC CLUSTERS, BUT XSTEST AND HARMBENCH RESULT IN "CUDA OUT OF MEMORY" in pitzer cluster - best fix: run olmes eval on OSC's Ascend Cluster

dashboard for OSC found: https://ondemand.osc.edu/pun/sys/dashboard , after you login you can click on Clusters then any of the shell access or...

you can go to your local device terminal and run: ssh -X -C {your_OSC_username}@ascend.osc.edu

You should be able to do setup without requesting an interactive job allocation but if you get errors, once you are logged in you can run: salloc --nodes=1 --gpus=1 --time=180 -A PAS3272
And then when you allocates you a job do: ssh {jobid}

Although not required, the load_base_llama_model.py contains the logic to load the llama-3.2-1b base model locally.

NOTE THAT YOU WANT TO MAKE SURE ALL VENV YOU ARE IN HAVE A "TRANSFORMERS" LIBRARY LESS THAN 5.0.0 to replicate (recommended to just use the olmes venv for everything - just make sure transformers version is less than 5.0.0)

```bash
#get your github token and allow access to repos, you can replace the "{token}" with your whole github token(should start with "github_pat_")

#You can download our github main branch by calling:
!git clone --recurse-submodules https://{token}@github.com/FloatingStory/CSE-5525-Default-Project.git

#or do the following to download this dpo branch
!git clone -b dpo --recurse-submodules https://{token}@github.com/FloatingStory/CSE-5525-Default-Project.git
```


To set up venv to merge please follow these instructions:
```bash
pip install uv

#download python 3.11.13 and then
#create a venv with python 3.11.13
uv venv --python 3.11.13 .venv_for_merging_model_and_checkpoints

#activate venv
source .venv_for_merging_model_and_checkpoints/bin/activate

#install proper requirement versions
python -m pip install -r ai2-olmes_venv_requirements.txt

#chelsea's read-only hugging face token, contains access to meta's llama models
export HF_TOKEN=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

#adjust adapter_path and output_dir variables in merge_model_and_checkpoint.py
#adapter_path should point to checkpoint directory holding adapters produced after running with LoRA/QLoRA
#output_dir adjust to ensure where the merged model is placed
#then run
python3 merge_model_and_checkpoint.py 

#or use sbatch:
#check merge_model.sh and adjust so the "source .../bin/activate" is set to your path and not mine
sbatch merge_model.sh
```

AFTER RUN COMPLETE: IMPORTANT ADJUSTMENTS REQUIRED TO WHERE EVER YOU SAVED THE MERGED MODEL (whatever <output_dir> is)
###### ADJUST THE FOLLOWING IN THE <output_dir> TO PREVENT ERRORS WHEN RUNNING THE OLMES EVALUATION:

Once the model is saved in the "output_dir" please do the following to allow the olmes eval dataset to run smoothly:

    1. Make a copy of chat_template_customrolecolon.jinja (and rename to chat_template.jinja) and place the chat_template.jinja file found in this folder into the "output_dir", this chat template is based off the Tinker role_colon Renderer format
    
    2. In the "output_dir" go into tokenizer_config.json and adjust the line with "tokenizer_class: <some_other_thing>" to "tokenizer_class": "PreTrainedTokenizerFast" (or make sure it is set to that by default)


===================================================================================
To set up olmes evaluation please follow these instructions, this uses a different venv:

```bash
#follow "Running Evaluations" part in the main README.md (not this README but the one provided by prof (in parent directory))

#your_path not including the start of the default project directory obtained by loading from github
#activate olmes venv, must use this one specifically as it defaults to this
source <your_path>/CSE-5525-Default-Project/evals/olmes/.venv/bin/activate

#chelsea's read-only hugging face token, contains access to meta's llama models and access needed for the olmes evals
export HF_TOKEN=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

#####versions used by chelsea in for the olmes venv can be found in ai2-olmes_venv_requirements.txt 
#If you are in the olmes venv, you can run python -m pip install -r ai2-olmes_venv_requirements.txt or use "python -m pip freeze > <file_name>.txt" and use diff to compare differences 

#assuming you are in CSE-5525-Default-Project directory
#the "for_sarah_sbatch_run_eval.sh" is similar to the run_eval.sh provided by default but you must adjust the following variables in it:
#model_path: should point to directory that holds merged model
#output_path: name of directory to place olmes eval output

#Then move the "for_sarah_sbatch_run_eval.sh" files into the "evals" directory so it is in the same directory as the olmes folder

#You can do this if you are requested and are running an interactive job:
cd evals
bash for_sarah_sbatch_run_eval.sh

#Or you can do sbatch on OSC's Ascend or Cardinal cluster:
#check eval_olmes_sbatch.sh and adjust so the "source .../bin/activate" is set to your path and not mine
sbatch eval_olmes_sbatch.sh
```

=====================

You can then extract primary scores once olmes evaluation is done by running:
```bash
#adjust the list_of_paths variable as specified in json_olmes_eval_all_metrics_extractor.py, then you can run:
python3 json_olmes_eval_all_metrics_extractor.py 
```
