from datasets import load_dataset
from train_sft_local import SFTLocalTrainer

dataset = load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225")

train_dataset = dataset["train"].select(range(500))

trainer = SFTLocalTrainer(
    model_name="distilgpt2",
    train_dataset=train_dataset,
    val_dataset=None,
    training_args={
        "batch_size": 2,
        "learning_rate": 5e-5,
        "num_epochs": 10,
        "max_length": 128
    }
)

trainer.train()