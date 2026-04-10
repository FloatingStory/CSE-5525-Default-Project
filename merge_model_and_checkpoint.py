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

#save tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(output_dir)