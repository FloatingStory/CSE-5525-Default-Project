from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "meta-llama/Llama-3.2-1B"
adapter_path = "./unzipped_checkpoint_directory"
output_dir = "./output_directory_for_model"

#load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto",
    device_map="auto"
)

#load adapter(LoRA) on top
model = PeftModel.from_pretrained(model, adapter_path)

#merge adapter into base model
model = model.merge_and_unload()

#save merged model
model.save_pretrained(output_dir)

#save tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(output_dir)