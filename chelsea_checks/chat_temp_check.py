from jinja2 import Environment, FileSystemLoader
from tinker_cookbook.tokenizer_utils import get_tokenizer
from transformers import AutoTokenizer

# Load template from file
env = Environment(loader=FileSystemLoader("."))
template = env.get_template("chat_temp_customllama3.jinja")

# Demo messages
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "What is 2+2?"}
]

# Render
output = template.render(messages=messages)

# Print results
print("=== RAW ===")
print(output)

print("=== REPR (IMPORTANT) ===")
print(repr(output))