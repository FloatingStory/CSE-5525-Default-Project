from transformers import AutoTokenizer, AutoModelForCausalLM

#load base model to local folder for testing
if __name__ == "__main__":

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")#checkpoint_path)
    #load corresponding tokenizer of same model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model.save_pretrained("OSC_LLAMA_BASEMODEL")