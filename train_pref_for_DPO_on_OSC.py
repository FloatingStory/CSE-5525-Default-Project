"""
This module implements the PREFTrainer class for training your model using preference optimization.
"""
from datasets import load_dataset

from datetime import datetime

import chz

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import (
    DPODatasetBuilderFromComparisons,
)

from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.lr_scheduling import LRSchedule

from transformers import AutoTokenizer, AutoModelForCausalLM

from trl import DPOTrainer, DPOConfig
from torch.utils.data import DataLoader
#=====
import logging
import re
from typing import cast

import chz
import datasets
import pandas as pd

from tinker_cookbook import renderers
from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import (
    Comparison,
    LabeledComparison,
)

# from langdetect import detect

#Implementing DPO - Direct Preference Optimization
class PREFTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args

    def train(self):
        # Implement the training loop here
        trainer = DPOTrainer(model=self.model, args=self.training_args, processing_class=self.tokenizer, train_dataset=self.train_dataset)
        trainer.train()
        #saving model in OSC_DPO folder
        trainer.save_model("OSC_DPO")

#####copied and slight adjusted from train.py in dpo example provided by Tinker
@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    dataset: str = "olmo"  # or path like tinker_cookbook.preference.preference_datasets:HHHBuilder
    load_checkpoint_path: str | None = None
    renderer_name: str | None = None

    # Training parameters
    learning_rate: float = 1e-5
    lr_schedule: LRSchedule = "linear"
    # num_epochs: int | None = 1
    dpo_beta: float = 0.1           #KL-penalty coefficient in the DPO loss. Higher values penalize deviations from the reference model more strongly.
    lora_rank: int | None = 16 #8
    save_every: int | None = 0    #save checkpoint every N steps
    max_length: int | None = 512 #700 #2048 #8192
    batch_size: int = 32 #256

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Service configuration
    base_url: str | None = None

    # DPO-specific parameters
    reference_model_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps: int | None = None

#####end of referenced Tinker code


def prompt_and_reponses(examples) -> dict[str, str, str]:
    return {
        "prompt" : [{"role": "user", "content": examples["chosen"][0]["content"]}],
        "chosen": [{"role": "assistant", "content": examples["chosen"][1]["content"]}],
        "rejected": [{"role": "assistant", "content": examples["rejected"][1]["content"]}],
    }

    # return {
    #     "prompt" : f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{examples['chosen'][0]['content']}\n<|eot_id|>",
    #     "chosen": f"<|start_header_id|>assistant<|end_header_id|>\n{examples['chosen'][1]['content']}\n<|eot_id|>",
    #     "rejected": f"<|start_header_id|>assistant<|end_header_id|>\n{examples['rejected'][1]['content']}\n<|eot_id|>",
    # }

    # return {
    #     "prompt" : f"<|begin_of_text|><|start_header_id|>{examples['chosen'][0]['role']}<|end_header_id|>\n{examples['chosen'][0]['content']}\n<|eot_id|>",
    #     "chosen": f"<|start_header_id|>{examples['chosen'][1]['role']}<|end_header_id|>\n{examples['chosen'][1]['content']}\n<|eot_id|>",
    #     "rejected": f"<|start_header_id|>{examples['rejected'][1]['role']}<|end_header_id|>\n{examples['rejected'][1]['content']}\n<|eot_id|>",
    # }

    # return {
    #     "prompt" : examples["chosen"][0]["content"],
    #     "chosen": examples["chosen"][1]["content"],
    #     "rejected": examples["rejected"][1]["content"],
    # }


def load_and_preprocess_data():
    
    #load allenai/olmo-2-0425-1b-preference-mix dataset from huggingface
    dataset = load_dataset("allenai/olmo-2-0425-1b-preference-mix", split="train")#, streaming=True)

    # #only keep English examples using langdetect based on the 'chosen' content, WARNING: takes about 28 minutes on CPU
    # def is_english(example):
    #     try:
    #         #check using prompt
    #         text = example["chosen"][0]["content"]
    #         return detect(text) == "en"
    #     except:
    #         return False

    # dataset = dataset.filter(is_english)

    # test_dataset = dataset.take(256)
    # train_dataset = dataset.skip(256)

    #get column names from olmo
    original_columns = dataset.column_names

    #adjust dataset by setting each example to be what is returned by the prompt_and_responses function(3 columns: prompt, chosen response, rejected response)
    dataset = dataset.map(
        prompt_and_reponses,
        batched=False,
        remove_columns=original_columns
    )
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    print(f"\n\n=== First item in dataset ===")
    print(dataset)
    print(dataset[1])#['prompt'])

    # print(f"\n\nFirst item in dataset parsed out ===")
    # print(dataset[1]['prompt'])
    # print(dataset[1]['chosen'])
    # print(dataset[1]['rejected'])
    return dataset


if __name__ == "__main__":
    #use "huggingface-cli login" and login using a hf token in terminal for faster loading speed

    cli_config = chz.entrypoint(CLIConfig)

    #TODO:load model from checkpoint(eventually) and use for DPOTrainer
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")#checkpoint_path)
    #load corresponding tokenizer of same model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


    print("\n\n=== NOW LOADING AND PREPROCESSING DATA ===")
    training_data = load_and_preprocess_data()

    #adjust to be path to sft weights
    checkpoint_path = "./checkpoint...."

    #initialize dpo configurations(hyperparameters) for training
    training_args = DPOConfig(
        learning_rate=1e-6,
        max_length=4000,                #max length for tokenized sequence
        loss_type='sigmoid',
        output_dir="OSC_DPO", 

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  #8*4 = 32 for effective batch size
        
        logging_steps=10
        )

    #initialize PREFTrainer
    dpoModel = PREFTrainer(model=model, tokenizer=tokenizer, train_dataset=training_data, val_dataset=training_data, training_args=training_args)
    #run training
    dpoModel.train()


"""Run using command:
    python -m train_pref model_name=meta-llama/Llama-3.2-1B dataset=olmo renderer_name=role_colon learning_rate=1e-5 lora_rank=8 save_every=1000 dpo_beta=0.1 max_steps=1000

    python -m train_pref model_name=meta-llama/Llama-3.2-1B dataset=olmo renderer_name=role_colon learning_rate=1e-5 lora_rank=8 dpo_beta=0.1
"""
