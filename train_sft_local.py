import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm


class SFTLocalTrainer:
    def __init__(self, model_name, train_dataset, val_dataset, training_args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Using device:", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args

    def format_example(self, example):
        messages = example["messages"]

        user_msg = ""
        assistant_msg = ""

        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                assistant_msg = msg["content"]

        prompt = f"### Instruction:\n{user_msg}\n\n### Response:\n"
        full_text = prompt + assistant_msg

        tokens = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.training_args["max_length"],
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        prompt_len = len(self.tokenizer(prompt)["input_ids"])
        labels[:prompt_len] = -100

        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def collate_fn(self, batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device)
        }

    def preprocess_dataset(self, dataset):
        return [self.format_example(x) for x in dataset]

    def train(self):
        batch_size = self.training_args["batch_size"]
        lr = self.training_args["learning_rate"]
        epochs = self.training_args["num_epochs"]

        train_data = self.preprocess_dataset(self.train_dataset)

        dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        optimizer = AdamW(self.model.parameters(), lr=lr)

        self.model.train()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            total_loss = 0

            for batch in tqdm(dataloader):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )

                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Avg Loss: {avg_loss:.4f}")

            self.model.save_pretrained(f"sft_checkpoint_epoch_{epoch+1}")
            self.tokenizer.save_pretrained(f"sft_checkpoint_epoch_{epoch+1}")

        print("Training complete!")