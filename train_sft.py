import asyncio
from datetime import datetime
import datasets
from typing import cast
import chz

import tinker

from tinker_cookbook import checkpoint_utils, cli_utils, renderers
from tinker_cookbook.supervised import train
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.supervised.data import FromConversationFileBuilder, conversation_to_datum, SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig, SupervisedDataset
from tinker_cookbook.utils.lr_scheduling import LRSchedule

# api key command
# $env:TINKER_API_KEY="your_key_here"

@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    dataset: str = "tulu3"

    train_on_what: renderers.TrainOnWhat = "last_assistant_message"

    learning_rate: float = 1e-4
    num_epochs: int = 1
    batch_size: int = 32
    max_length: int = 1024

    lora_rank: int = 16

    log_path: str | None = None
    load_checkpoint_path: str | None = None
    renderer_name: str | None = "role_colon"
    base_url: str | None = None

    save_every: int = 200
    eval_every: int = 1000
    infrequent_eval_every: int = 1000

    wandb_project: str | None = None
    wandb_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "overwrite"

    max_samples: int = 0
    english_only: bool = False
    use_mixture: bool = False # instruction 40%, reasoning 30%, coding 30%

    min_text_length: int = 100
    max_text_length: int = 2000
    length_filter_on: bool = False

@chz.chz
class Olmo2Builder(ChatDatasetBuilder):
    cli_config: CLIConfig
    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:

        def isEnglish(example):
            # for message in example["messages"][:5]:
            #     if not all(ord(char) < 128 for char in message["content"]):
            #         return False
            text = " ".join([m["content"] for m in example["messages"][:5]])

            if len(text) == 0:
                return False

            ascii_ratio = sum(c.isascii() for c in text) / len(text)

            return ascii_ratio > 0.9
            
        def classify(example):
            text = example["messages"][-1]["content"].lower()

            if "def " in text or "return" in text or "code" in text or "program" in text:
                return "coding"
            elif "step by step" in text or "therefore" in text or "proof by contradiction" in text or "mathematical proof" in text:
                return "reasoning"
            else:
                return "instruction"  
            
        def isValidLength(example):
            # only check if assistant message has valid length
            messages = example["messages"]

            if len(messages) == 0:
                return False
            
            last_msg = messages[-1]
            
            if "content" not in last_msg:
                return False
            
            length = len(last_msg["content"])

            if self.cli_config.min_text_length > 0 and length < self.cli_config.min_text_length:
                return False

            if self.cli_config.max_text_length > 0 and length > self.cli_config.max_text_length:
                return False

            return True
            
        dataset = datasets.load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225")
        dataset = cast(datasets.DatasetDict, dataset)
        dataset = dataset["train"]
        dataset = dataset.shuffle(seed=0)

        # filter english if specified
        if self.cli_config.english_only:
            dataset = dataset.filter(isEnglish)
            print("[INFO] Filtered to English only. Remaining samples:", len(dataset))
        if self.cli_config.use_mixture:
            dataset = dataset.map(lambda x: {"type": classify(x)})

            instruction_ds = dataset.filter(lambda x: x["type"] == "instruction")
            reasoning_ds = dataset.filter(lambda x: x["type"] == "reasoning")
            coding_ds = dataset.filter(lambda x: x["type"] == "coding")

            max_n = self.cli_config.max_samples
            if (max_n == 0):
                max_n = len(dataset)

            inst_n = int(max_n * 0.4)
            reas_n = int(max_n * 0.3)
            code_n = int(max_n * 0.3)

            instruction_ds = instruction_ds.select(range(min(len(instruction_ds), inst_n)))
            reasoning_ds = reasoning_ds.select(range(min(len(reasoning_ds), reas_n)))
            coding_ds = coding_ds.select(range(min(len(coding_ds), code_n)))

            dataset = datasets.concatenate_datasets([instruction_ds, reasoning_ds, coding_ds])
            dataset = dataset.shuffle(seed=0)
        if self.cli_config.length_filter_on:
            before_len = len(dataset)
            dataset = dataset.filter(isValidLength)
            after_len = len(dataset)

            print(f"[INFO] Length filter applied: {before_len} -> {after_len}")
            print(f"[INFO] min_length={self.cli_config.min_text_length}, max_length={self.cli_config.max_text_length}")
        if self.cli_config.max_samples > 0:
            dataset = dataset.select(range(min(len(dataset), self.cli_config.max_samples)))

        test_ds = dataset.take(2000)
        train_ds = dataset.skip(2000)

        train_on_what = (
            renderers.TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        return SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        ), SupervisedDatasetFromHFDataset(
            test_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )

          
def get_dataset_builder(cfg: CLIConfig):
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.renderer_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        train_on_what=cfg.train_on_what,
    )

    if cfg.dataset == "tulu3":
        return chat_datasets.Tulu3Builder(common_config=common_config)

    elif cfg.dataset.endswith(".jsonl"):
        return FromConversationFileBuilder(
            common_config=common_config,
            file_path=cfg.dataset,
        )
    elif cfg.dataset == "allenai/tulu-3-sft-olmo-2-mixture-0225":
        return Olmo2Builder(common_config=common_config, cli_config=cfg)

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


def main(cli_config: CLIConfig):

    model_name_safe = cli_config.model_name.replace("/", "-")
    run_name = f"sft-{model_name_safe}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    log_path = cli_config.log_path or f"./logs/{run_name}"

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    config = train.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        dataset_builder=get_dataset_builder(cli_config),
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        learning_rate=cli_config.learning_rate,
        lr_schedule="linear",
        num_epochs=cli_config.num_epochs,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        lora_rank=cli_config.lora_rank,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
        infrequent_eval_every=cli_config.infrequent_eval_every,
    )

    asyncio.run(train.main(config))


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    main(cfg)




