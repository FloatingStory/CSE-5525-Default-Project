import datasets
from typing import cast
import json
from tqdm import tqdm
from langdetect import detect

MIN_TEXT_LENGTH = 100
MAX_TEXT_LENGTH = 2000

VALID_LENGTH_ONLY = True
ENGLISH_ONLY = True

OUTPUT_PATH = "filtered_dataset_sample.jsonl"
SAVE_LIMIT = 5000


def isEnglish(example):
    text = " ".join([m["content"] for m in example["messages"][:3]])

    if len(text.strip()) < 20:
        return False

    common_words = ["the", "is", "are", "and", "to", "of", "in"]
    text_lower = text.lower()
    
    if sum(word in text_lower for word in common_words) < 1:
        return False

    try:
        return detect(text) == "en"
    except:
        return False


def isValidLength(example):
    messages = example["messages"]

    if len(messages) == 0:
        return False

    last_msg = messages[-1]

    if last_msg.get("role") != "assistant":
        return False

    text = last_msg.get("content", "")
    length = len(text)

    if MIN_TEXT_LENGTH > 0 and length < MIN_TEXT_LENGTH:
        return False

    if MAX_TEXT_LENGTH > 0 and length > MAX_TEXT_LENGTH:
        return False

    return True


def main():
    print("[INFO] Loading dataset...")
    dataset = datasets.load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225")
    dataset = cast(datasets.DatasetDict, dataset)["train"]

    print("[INFO] Original size:", len(dataset))

    if VALID_LENGTH_ONLY:
        print("[INFO] Applying length filter...")
        before = len(dataset)
        dataset = dataset.filter(isValidLength)
        print(f"[INFO] Length filter: {before} -> {len(dataset)}")

    if ENGLISH_ONLY:

        print("[INFO] Applying English filter...")
        before = len(dataset)
        dataset = dataset.filter(isEnglish)
        print(f"[INFO] English filter: {before} -> {len(dataset)}")

    print("[INFO] Final size:", len(dataset))
    print("[INFO] Kept ratio:", len(dataset) / before if before > 0 else 0)

    save_n = min(len(dataset), SAVE_LIMIT)
    print(f"[INFO] Saving {save_n} samples to {OUTPUT_PATH}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i in tqdm(range(save_n)):
            row = dataset[i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("[INFO] Done!")


if __name__ == "__main__":
    main()