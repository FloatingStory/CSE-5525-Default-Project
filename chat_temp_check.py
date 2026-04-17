from jinja2 import Environment, FileSystemLoader

# Load template from file
env = Environment(loader=FileSystemLoader("."))
template = env.get_template("chat_temp_rolecolon.jinja")

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

print("\n=== REPR (IMPORTANT) ===")
print(repr(output))