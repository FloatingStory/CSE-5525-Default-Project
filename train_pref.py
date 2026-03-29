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
class ComparisonBuilder(ComparisonDatasetBuilder):
    """olmo-2-0425-1b-preference-mix dataset comparison builder."""

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset(
            "allenai/olmo-2-0425-1b-preference-mix", split="train"
        )
        dataset = cast(datasets.Dataset, dataset)
        dataset = dataset.shuffle(seed=0)
        test_dataset = dataset.take(1024)
        train_dataset = dataset.skip(1024)
        return train_dataset, test_dataset

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
        return LabeledComparison(comparison=comparison, label="A")


def load_and_preprocess_data():

    #automatically select corresponding tokenizer using the dataset
    tokenizer = AutoTokenizer.from_pretrained("allenai/olmo-2-0425-1b")

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
