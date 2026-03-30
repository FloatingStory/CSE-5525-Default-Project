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
from tinker_cookbook.recipes.preference.datasets import (
    HelpSteer3ComparisonBuilder,
    HHHComparisonBuilder,
    UltraFeedbackComparisonBuilder,
)
from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.lr_scheduling import LRSchedule

from transformers import AutoTokenizer

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
        pass

@chz.chz
class OLMOComparisonBuilder(ComparisonDatasetBuilder):
    """olmo-2-0425-1b-preference-mix dataset comparison builder."""

    #Load raw dataset and take 1024 for validation and the rest for training
    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset(
            "allenai/olmo-2-0425-1b-preference-mix", split="train"
        )
        dataset = dataset.map(lambda example: {"dummy_prompt_text": ""})
        dataset = cast(datasets.Dataset, dataset)
        dataset = dataset.shuffle(seed=0)
        test_dataset = dataset.take(1024)
        train_dataset = dataset.skip(1024)
        return train_dataset, test_dataset

    #example (dict): a single row from the HuggingFace dataset
    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        instruction = example["dummy_prompt_text"]
        chosen_response = example["chosen"][1]["content"]
        rejected_response = example["rejected"][1]["content"]

        # prompt_conversation: list[renderers.Message] = [{"role": "user", "content": instruction}]

        comparison = Comparison(
            prompt_conversation=instruction, # prompt_conversation=prompt_conversation,
            completion_A=[{"role": "assistant", "content": chosen_response}],
            completion_B=[{"role": "assistant", "content": rejected_response}],
        )
        #choose A over B
        return LabeledComparison(comparison=comparison, label="A")


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
    # lora_rank: int | None = 4
    # save_every: int | None = 0    #save checkpoint every N steps
    max_length: int | None = 1024 #8192
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


def get_dataset_builder(
    dataset: str,
    model_name: str,
    renderer_name: str,
    max_length: int | None,
    batch_size: int,
) -> ChatDatasetBuilder:
    """Get the appropriate dataset builder for DPO training."""
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    if dataset == "olmo":
        return DPODatasetBuilderFromComparisons(
            common_config=common_config, comparison_builder=OLMOComparisonBuilder()
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    

def cli_main(cli_config: CLIConfig):
    """Main CLI function that builds the full config and calls the training function."""
    # Build full config
    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_name = cli_config.model_name.replace("/", "-")
    run_name = f"{cli_config.dataset}-{model_name}-{cli_config.learning_rate}lr-{cli_config.batch_size}batch-{date_and_time}"
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"./LogPathNotFoundFolder/{run_name}"       #CHANGE PATHS
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    #documentation of Config parameters at https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/preference/train_dpo.py
    config = train_dpo.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        dataset_builder=get_dataset_builder(
            cli_config.dataset,
            cli_config.model_name,
            renderer_name,
            cli_config.max_length,
            cli_config.batch_size,
        ),
        load_checkpoint_path=cli_config.load_checkpoint_path,   #path to a checkpoint to initialize weights from.  ``None`` starts from the base model
        evaluator_builders=[],
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        # num_epochs=cli_config.num_epochs,
        dpo_beta=cli_config.dpo_beta,
        #lora_rank=cli_config.lora_rank,
        base_url=cli_config.base_url,
        # save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        reference_model_name=cli_config.reference_model_name,
        max_steps=cli_config.max_steps,
    )

    #commented out until we actually want to run the training on Tinker
    # train_dpo.main(config)



#for debugging config setup
def debug_run(cli_config: CLIConfig):
    print("\n=== DEBUG RUN (no training) ===")

    # Resolve renderer
    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )
    print(f"Renderer: {renderer_name}")

    # Build dataset builder
    dataset_builder = get_dataset_builder(
        cli_config.dataset,
        cli_config.model_name,
        renderer_name,
        cli_config.max_length,
        cli_config.batch_size,
    )
    print("Dataset builder created")

    # Get datasets
    train_ds, val_ds = dataset_builder.comparison_builder.get_train_and_test_datasets()
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds) if val_ds else 0}")

    # Peek at raw example
    print("\nSample raw example:")
    print(train_ds[1])
    print("\n\nExtract content Sample raw example:")
    print(train_ds[1]['chosen'][0]['content'])

    # Tokenizer test
    tokenizer = AutoTokenizer.from_pretrained(cli_config.model_name)
    sample_text = "This is trash"
    tokens = tokenizer(sample_text)
    print("\nTokenizer test:")
    print(tokens)

    print("\nDEBUG RUN SUCCESSFUL")

#####end of referenced code



def load_and_preprocess_data():

    #automatically select corresponding tokenizer using the dataset
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    #load allenai/olmo-2-0425-1b-preference-mix dataset from huggingface
    ds = load_dataset("allenai/olmo-2-0425-1b-preference-mix", split="train")#, streaming=True)
    #add a new column "dummy_prompt_text" filled with empty strings
    ds = ds.map(lambda example: {"dummy_prompt_text": ""})

    #use dummy_prompt_text, chosen, rejected for prompt, chosen, rejected keys respectively

    print(f"\n\nFirst item in dataset:")
    print(ds)
    return ds

if __name__ == "__main__":
    #use "huggingface-cli login" and login using a hf token in terminal for faster loading speed
    
    training_data = load_and_preprocess_data()

    cli_config = chz.entrypoint(CLIConfig)

    #debug check
    debug_run(cli_config)

    print("\n\n=== NOW RUNNING cli_main ===")
    #actual dpo tinker call
    cli_main(cli_config)

"""Run using command:

    python -m train_pref model_name=meta-llama/Llama-3.2-1B dataset=olmo renderer_name=role_colon learning_rate=1e-5 dpo_beta=0.1
"""
