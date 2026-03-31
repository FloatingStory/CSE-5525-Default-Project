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
from tinker_cookbook.supervised.nll_evaluator import NLLEvaluator
from tinker_cookbook.utils.lr_scheduling import LRSchedule


@chz.chz
class Olmo2Builder(ChatDatasetBuilder):
    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        dataset = datasets.load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225")
        dataset = cast(datasets.DatasetDict, dataset)
        dataset = dataset["train"]
        dataset = dataset.shuffle(seed=0)
        test_ds = dataset.take(1024)
        train_ds = dataset.skip(1024)

        train_on_what = (
            renderers.TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )

        def map_fn(row: dict):
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        return SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        ), SupervisedDatasetFromHFDataset(
            test_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )


@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    dataset: str = "allenai/tulu-3-sft-olmo-2-mixture-0225"

    train_on_what: renderers.TrainOnWhat = "last_assistant_message"
    batch_size: int = 32
    max_length: int = 1024
    load_checkpoint_path: str | None = None
    renderer_name: str | None = "role_colon"


def get_dataset_builder(cfg: CLIConfig):
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.renderer_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        train_on_what=cfg.train_on_what,
    )

    if cfg.dataset == "allenai/tulu-3-sft-olmo-2-mixture-0225":
        return Olmo2Builder(common_config=common_config)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


async def run_nll_eval(cli_config: CLIConfig):
    builder = get_dataset_builder(cli_config)
    _, test_dataset = builder()

    # evaluator 생성
    evaluator = NLLEvaluator.from_dataset(test_dataset)

    # TrainingClient 생성 및 checkpoint 로드
    client = tinker.TrainingClient(
        load_checkpoint_path=cli_config.load_checkpoint_path
    )
    evaluator.client = client

    print("===== Tinker NLL / Loss Evaluation =====")
    results = await evaluator()
    print(results)


def main(cfg: CLIConfig):
    asyncio.run(run_nll_eval(cfg))


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    main(cfg)