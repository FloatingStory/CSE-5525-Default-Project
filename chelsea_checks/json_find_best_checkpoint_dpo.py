import os
import json

base_dir = "/users/PAS2930/cwang/CSE5525DefaultProject/CSE-5525-Default-Project/OSC_DPOQLORA_TRAINED"
# base_dir = "/Users/chelsea/Documents/CSE5525_DefaultProj/cse-5525-spring-2026-default-project"
list_of_checkpoint_dirs = []

for name in os.listdir(base_dir):
    full_path = os.path.join(base_dir, name)
    # print(full_path)
    if os.path.isdir(full_path) and name.startswith("checkpoint"):
        print(full_path)
        list_of_checkpoint_dirs.append(full_path)

for checkpoint in list_of_checkpoint_dirs:
    file_path = f"{checkpoint}/trainer_state.json"
    with open(file_path, "r") as f:
        data = json.load(f)

    is_best_checkpoint = data["best_model_checkpoint"]
    print(f"{file_path}: {is_best_checkpoint}")