from tinker_cookbook.renderers.llama3 import Llama3Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers.role_colon import RoleColonRenderer

def main():
    # 1. Load tokenizer (must match the model you're targeting)
    tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B")

    # 2. Initialize renderer
    renderer =Llama3Renderer(tokenizer)
    # renderer = RoleColonRenderer(tokenizer)

    # 3. Define a sample conversation
    messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "What is 2+2?"}
    ]

    # 4. Build the prompt (this is the IMPORTANT call)
    model_input = renderer.build_generation_prompt(messages)

    # 5. Collect all tokens from chunks
    all_tokens = []
    for chunk in model_input.chunks:
        if hasattr(chunk, "tokens"):
            all_tokens.extend(chunk.tokens)

    # 6. Decode to string
    prompt_str = tokenizer.decode(all_tokens)

    # 7. Print result
    print("==== FULL PROMPT ====")
    print(prompt_str)

    print(repr(prompt_str))


if __name__ == "__main__":
    main()


# <|begin_of_text|><|start_header_id|>system<|end_header_id|>

# You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

# Explain gravity in one sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# Hello! How can I help you today?<|eot_id|><|start_header_id|>user<|end_header_id|>

# Explain gravity simply.<|eot_id|><|start_header_id|>assistant<|end_header_id|>




#llama3 renderer
# {%- if messages[0]['role'] == 'system' -%}
# <|begin_of_text|>
# <|start_header_id|>system<|end_header_id|>

# {{ messages[0]['content'] }}<|eot_id|>
# {%- set messages = messages[1:] -%}
# {%- else -%}
# <|begin_of_text|>
# {%- endif -%}

# {%- for message in messages %}
# <|start_header_id|>{{ message['role'] }}<|end_header_id|>

# {{ message['content'] }}<|eot_id|>
# {%- endfor %}

# <|start_header_id|>assistant<|end_header_id|>


#role colon renderer
# {{ bos_token }}{% for message in messages %}
# {% if message['role'] == 'system' -%}
# System: {{ message['content'] }}
# {%- elif message['role'] == 'user' -%}
# User: {{ message['content'] }}
# {%- elif message['role'] == 'assistant' -%}
# Assistant: {{ message['content'] }}
# {%- endif %}

# {% endfor -%}Assistant:


#better. attempt
# {{ bos_token }}{%- for message in messages -%}
# {{ message['role'].capitalize() ~ ': ' ~ message['content'] ~ '\n\n' }}
# {%- endfor -%}
# {{ "Assistant: " }}

#some other
# {{- bos_token -}}{%- for message in messages -%}
# {{ message['role'].capitalize() ~ ': ' ~ message['content'] ~ '\n\n' }}
# {%- endfor -%}
# {{ "Assistant: " }}
