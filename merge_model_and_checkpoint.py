from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "meta-llama/Llama-3.2-1B"
adapter_path = "./sft_best_checkpoint"
output_dir = "./MERGED_SFT_MODEL"

#load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="cpu"
)

#load adapter(LoRA) on top
model_adapt = PeftModel.from_pretrained(model, adapter_path, device_map="cpu")

#merge adapter into base model
model = model_adapt.merge_and_unload()

#save merged model
model.save_pretrained(output_dir)

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# with open("chat_template_llama3.jinja", "r", encoding="utf-8") as f:
#     chat_template = f.read()
# tokenizer.chat_template = chat_template

# tokenizer.tokenizer_class = "PreTrainedTokenizerFast"

#save tokenizer
tokenizer.save_pretrained(output_dir)


#once the model is saved in the "output_dir" please do the following to allow the olmes eval dataset to run smoothly:
    #1. Make a copy and place the chat_template.jinja file from the Llama-3.2-1b-Instruct model into the "output_dir"
    #2. In the "output_dir" go into tokenizer_config.json and adjust to "tokenizer_class": "PreTrainedTokenizerFast"