"""
This module implements the PREFTrainer class for training your model using preference optimization.
"""
from datetime import datetime

import chz

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import (
    DPODatasetBuilderFromComparisons,
)
# from tinker_cookbook.recipes.preference.datasets import (
#     HelpSteer3ComparisonBuilder,
#     HHHComparisonBuilder,
#     UltraFeedbackComparisonBuilder,
# )
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
        pass

@chz.chz
class OLMOComparisonBuilder(ComparisonDatasetBuilder):
    """olmo-2-0425-1b-preference-mix dataset comparison builder."""

    #Load raw dataset and take 1024 for validation and the rest for training
    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset(
            "allenai/olmo-2-0425-1b-preference-mix", split="train"
        )
        # dataset = dataset.map(lambda example: {"dummy_prompt_text": ""})
        dataset = cast(datasets.Dataset, dataset)
        dataset = dataset.shuffle(seed=0)

        # # Filter only English examples based on the 'chosen' content, WARNING: takes about 28 minutes on CPU
        # def is_english(example):
        #     try:
        #         #check using prompt
        #         text = example["chosen"][0]["content"]
        #         return detect(text) == "en"
        #     except:
        #         return False

        # dataset = dataset.filter(is_english)

        test_dataset = dataset.take(256)
        train_dataset = dataset.skip(256)
        return train_dataset, test_dataset

    #example (dict): a single row from the HuggingFace dataset
    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        # instruction = example["dummy_prompt_text"]
        instruction = example["chosen"][0]["content"]
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
    num_epochs: int | None = 1
    dpo_beta: float = 0.1           #KL-penalty coefficient in the DPO loss. Higher values penalize deviations from the reference model more strongly.
    lora_rank: int | None = 16
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
    reference_model_name: str | None = "meta-llama/Llama-3.2-1B"

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
        log_path = f"./LogPathNotFoundFolder/dpo/{run_name}"       #CHANGE PATHS
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
        num_epochs=cli_config.num_epochs,
        dpo_beta=cli_config.dpo_beta,
        lora_rank=cli_config.lora_rank,
        base_url=cli_config.base_url,
        save_every=cli_config.save_every,
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
    # sample_text = "This is trash"
    # sample_text = "Make a beginning story set in Code Geass…Lelouch going about his student days…or rather…WERE going on his student days…running a rebellion secret as Zero is a lot of work…against the fight of Britannia…it has complications as well…allies…tactics…battles can go wrong…and for here to Lelouch here in Ashford Academy?..his current and perhaps the most laughably disbelief…but all too real complication of his right now?..busy making out with Suzaku Kururugi…childhood friend…pilot of the annoying Lancelot…the Knight of 7 to Princess Euphemia…all of this…Lelouch knows…but Suzaku?..all he sees of Lelouch is childhood friend…and maybe...'more'......and that’s it…just a childhood friend……and Suzaku really is doing that deep kissing and loving thing REALLY close to Lelouch.…tongue even…as Lelouch wants to laugh…if he can…or cry even…because…this…god…how did this even…as Lelouch stifles a sound of pleasure…as Suzaku keeps going…both stop…saliva trail apparent as both they separate…both panting…yes indeed…how on earth did this happen…?..as Suzaku murmurs…“…Lelouch…” then hugs him closely…as if possessively…as if he doesn’t want Lelouch to go just yet….and then Suzaku says something that actually makes Lelouch’s heart do a certain thump…god…as Lelouch tries to protest…what about Euphie?..Suzaku is her knight!..but then Suzaku says that damned thing that makes Lelouch’s heart go thump AGAIN…this?..is not good…as it becomes obvious to Lelouch…all Suzaku sees right now is his childhood friend…Suzaku doesn’t know Lelouch is Zero at all…yes…Suzaku is making out with the greatest terrorist the Holy Britannian Empire has ever known…but since Suzaku doesn’t know who Lelouch really is…well…as his muscular frame hold Lelouch’s wiry frame close…as Suzaku makes a teasing joke of Lelouch's body....\"..but you do smell good..\" Suzaku mutters...and god damn it, Suzaku....as Lelouch thinks to himself......fuck...this is really not good....\n\nDo dialogue"
    sample_text = "This situation is indeed complex, to say the least. Here's a snippet of what follows, staying respectful and focused on the narrative drama of the situation:\n\n---\n\nLelouch's heart raced as he stared at the ceiling, feeling the warmth of Suzaku's chest pressing into his back. He couldn't help but feel a mix of nervousness, affection, and a hefty dose of disbelief.\n\nSuzaku's breathing slowed as he pulled away, his grip on Lelouch loosening but not entirely letting go. \"Lelouch, you have no idea how much I missed you.\"\n\nLelouch swallowed hard, trying to sort through his feelings. This moment was beyond complicated. He wanted to be honest with Suzaku, to let him know everything, but the weight of Zero's secret made it impossible. \"Suzaku...there's a lot to explain, but...\" He turned to face Suzaku, finding those familiar emerald eyes staring back at him with such intensity it nearly took his breath away.\n\n\"And that damned thing I said, about seeing you as my childhood friend,\" Suzaku continued, his voice soft but firm. \"I...I want you to know that's not the whole truth. I've always felt more, even if I didn't understand it. You're more than just a friend...\"\n\nLelouch's mind scrambled for words, for a way out that would preserve his secret but also acknowledge the truth of their connection. \"Suzaku, you have to understand—\" He was interrupted by Suzaku's lips pressing softly against his, a kiss gentle yet filled with an underlying intensity.\n\n\"Please, Lelouch,\" Suzaku whispered, his breath warm against Lelouch's cheek. \"Just...let me hold you for now. Everything else can wait, just...for a little while.\"\n\nLelouch closed his eyes, feeling the warmth of the moment, knowing it was fleeting and dangerous. He couldn't let it continue, yet he didn't want to push Suzaku away. It was a delicate balance, one he needed to navigate carefully. \"Suzaku, I need...we need to talk about this, but not like this, not now.\"\n\nSuzaku pulled back, his expression one of understanding but also of a quiet frustration. \"I know. Just...give me this, okay? This moment, this memory. It's all I ask.\"\n\nLelouch sighed, looking into those hopeful eyes, \"Alright, but...\" He paused, unsure of how to articulate his fears and responsibilities. \n\nSuzaku, however, seemed to understand and didn't press him further at the moment, instead pulling Lelouch closer into a gentle embrace. Both knew their moment was precious and precarious, each one building the tension between truth and secrecy.\n\n---\n\nThis scene is about the complexity of their relationship, the tension between their private emotions and the greater, more dangerous world they inhabit, without crossing into inappropriate territory."
    tokens = tokenizer(sample_text)
    print("\nTokenizer test:")
    # print(tokens)
    print(len(tokens["input_ids"][:]))

    print("\nDEBUG RUN SUCCESSFUL")

#####end of referenced code


if __name__ == "__main__":
    #use "huggingface-cli login" and login using a hf token in terminal for faster loading speed

    cli_config = chz.entrypoint(CLIConfig)

    #debug check
    debug_run(cli_config)

    print("\n\n=== NOW RUNNING cli_main ===")
    #actual dpo tinker call
    cli_main(cli_config)

"""Run using command:
    python -m train_pref model_name=meta-llama/Llama-3.2-1B dataset=olmo renderer_name=role_colon learning_rate=1e-5 lora_rank=8 save_every=1000 dpo_beta=0.1 max_steps=1000

    python -m train_pref model_name=meta-llama/Llama-3.2-1B dataset=olmo renderer_name=role_colon learning_rate=1e-5 lora_rank=8 dpo_beta=0.1
"""
