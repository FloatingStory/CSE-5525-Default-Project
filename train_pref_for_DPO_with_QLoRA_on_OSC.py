"""
This module implements the PREFTrainer class for training your model using preference optimization.
"""
from datasets import load_dataset

from datetime import datetime


from transformers import AutoTokenizer, AutoModelForCausalLM

from trl import DPOTrainer, DPOConfig
from torch.utils.data import DataLoader
#=====
from typing import cast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, TaskType



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
        trainer.save_model("OSC_DPO_TRAINED")


def prompt_and_reponses(examples) -> dict[str, str, str]:
    # return {
    #     "prompt" : [{"role": "user", "content": examples["chosen"][0]["content"]}],
    #     "chosen": [{"role": "assistant", "content": examples["chosen"][1]["content"]}],
    #     "rejected": [{"role": "assistant", "content": examples["rejected"][1]["content"]}],
    # }

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

    return {
        "prompt" : examples["chosen"][0]["content"].strip(),
        "chosen": " " + examples["chosen"][1]["content"].strip(),
        "rejected": " " + examples["rejected"][1]["content"].strip(),
    }



def load_and_preprocess_data():
    
    #load allenai/olmo-2-0425-1b-preference-mix dataset from huggingface
    dataset = load_dataset("allenai/olmo-2-0425-1b-preference-mix", split="train")#, streaming=True)

    #small run to test code
    dataset = dataset.take(10)


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

    #use for quantization condig
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          #best quality 4-bit quant
    bnb_4bit_use_double_quant=True,     #extra memory savings
    bnb_4bit_compute_dtype=torch.bfloat16  #or torch.float16
    )   


    #TODO:load model from checkpoint(eventually) and use for DPOTrainer
    model = AutoModelForCausalLM.from_pretrained(
        "./MERGED_SFT_MODEL",
        quantization_config=bnb_config,
        device_map="auto"
    )
    #checkpoint_path)
    # model.gradient_checkpointing_enable()
    # model.config.use_cache = False

    #PEFT - the LoRA Adapters to train on
    lora_config = LoraConfig(
    r=16,                    #rank (8–32 typical)
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,

    # target modules for attention and linear:
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj", 
        "down_proj"
    ]
    )

    model = get_peft_model(model, lora_config)
    print("======TRAINABLE PARAMETERS: ========")
    model.print_trainable_parameters()

    #load corresponding tokenizer of same model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token

    print("\n\n=== NOW LOADING AND PREPROCESSING DATA ===")
    training_data = load_and_preprocess_data()


    #initialize dpo configurations(hyperparameters) for training
    training_args = DPOConfig(
        learning_rate=1e-4,             #higher learning rate for LoRA
        max_length=1028,                #max length for tokenized sequence
        loss_type='sigmoid',
        output_dir="OSC_DPO_TRAINED", 

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  
        num_train_epochs=1,
        
        logging_steps=10,
        bf16=True,  
        optim="paged_adamw_8bit"
    )

    #initialize PREFTrainer
    dpoModel = PREFTrainer(model=model, tokenizer=tokenizer, train_dataset=training_data, val_dataset=training_data, training_args=training_args)
    #run training
    dpoModel.train()