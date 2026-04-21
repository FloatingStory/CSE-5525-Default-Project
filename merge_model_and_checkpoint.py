from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download

#====== IMPORTANT: must be in an environment where transformers version is less than 5.0.0

#use for SFT
base_model_name = snapshot_download("meta-llama/Llama-3.2-1B")

#use for DPO
#base_model_name = "./MERGED_SFT_FULLRUN_rolecolon_checkpointFINAL_attemptTOMERGEWITHSAMETRANSFORMERVERSION"

adapter_path = "./sft_15000_normal_rolecolon_part"
output_dir = "./MERGED_SFT_checkpoint15000_rolecolon_VALIDMERGE"

#load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="cpu",
    torch_dtype="auto"
)
model.eval()

#load adapter(LoRA) on top
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

#merge adapter into base model
model = model.merge_and_unload()

#save merged model
model.save_pretrained(output_dir, safe_serialization=True)

#save tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

#apply custom chat template with this name in current directory
#with open("chat_template_rolecolon.jinja", "r", encoding="utf-8") as f:
#    chat_template = f.read()
#tokenizer.chat_template = chat_template

#tokenizer.tokenizer_class = "PreTrainedTokenizerFast"

tokenizer.save_pretrained(output_dir)