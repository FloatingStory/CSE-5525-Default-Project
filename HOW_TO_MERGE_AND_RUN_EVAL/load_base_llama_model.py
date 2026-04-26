from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

#load base model to local folder for testing, make sure transformers library is less than 5.0.0 for consistency
if __name__ == "__main__":

    #model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")#checkpoint_path)
    snapshot_download(repo_id="meta-llama/Llama-3.2-1B", local_dir="./OSC_LLAMA_BASEMODEL_BASE")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.save_pretrained("OSC_LLAMA_BASEMODEL_BASE")
