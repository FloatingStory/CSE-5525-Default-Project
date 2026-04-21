from jinja2 import Environment, FileSystemLoader
from tinker_cookbook.tokenizer_utils import get_tokenizer
from transformers import AutoTokenizer

# Load template from file
env = Environment(loader=FileSystemLoader("."))
# template = env.get_template("chat_temp_customllama3.jinja")
template = env.get_template("chat_template_customrolecolon.jinja")

# Demo messages
messages = [
    {"role": "user", "content": "Make a comedic and vividly detailed fictional story set in Olympus about Zeus and his harem of twinks (compliment their physiques and butts). They’re lounging in bed au naturel, drinking and feasting as they chat with each other as Zeus couples with one of them"},
    {"role": "assistant", "content": "I appreciate your request, but I must let you know that creating content involving minors in an inappropriate or offensive manner, including fictional characters, is not something I'm comfortable generating. Additionally, the scenario described involves themes that are not suitable for a general audience. Perhaps we could explore a different, less sensitive topic within the realm of mythology or fiction that would be more appropriate?"}
    # {"role": "user", "content": "What is 2+2?"},
    # {"role": "assistant", "content": "4!"}
]

# Render
output = template.render(messages=messages)

# Print results
print("=== RAW ===")
print(output)

print("=== REPR (IMPORTANT) ===")
print(repr(output))