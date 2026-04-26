from datasets import load_dataset
from transformers import AutoTokenizer
from train_sft import SFTTrainer

dataset = load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225")

train_dataset = dataset["train"].select(range(50))

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

trainer = SFTTrainer(
    model_name="meta-llama/Llama-3.2-1B",
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    val_dataset=None,
    training_args={
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_epochs": 1,
        "max_length": 512
    }
)

trainer.train()