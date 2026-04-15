import json
import os

print("\n\n")

#folder_path = "./SBATCH_BASEMODEL_ALLDATASETS_OLMESEVAL-eval-gsm8k/metrics-all.jsonl"

#adjust to include name of directories holding the olmes evaluation <path>-eval-<dataset> is how it is named, list_of_paths is the <path> part
list_of_paths = [
    "SBATCH_BASEMODEL_ALLDATASETS_OLMESEVAL",
    "OSC_QUARTERSFTMODEL_SBATCH",
    "SFTMODEL_10K_ENG_MIX_OLMESEVALS",
    "SFTMODEL_10K_ENG_OLMESEVALS",
    "SFTMODEL_10K_MIX_APRIL12_OLMESEVALS",
    "SFTMODEL_10K_NORMAL_OLMESEVALS",
    "SFTMODEL_10K_NORMAL_lr5e-5_rank16_OLMESEVALS"
]

list_of_datasets = [
    "gsm8k",
    "ifeval",
    "mbpp",
    "harmbench::default",
    "xstest::default"
]

#FOR .JSONL files, assumes directory is located in the same directory as this file, feel free to adjust
#extracts primary scores from metrics-all.jsonl
for dataset in list_of_datasets:
    for filename in list_of_paths:
        with open(f"./{filename}-eval-{dataset}/metrics-all.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                model_file = data.get("model_config", {}).get("model")
                model_file = model_file.split("/")[-1]  #get directory that holds model used for eval

                task = data.get("task_config", {}).get("task_name")
                score = data.get("metrics", {}).get("primary_score")
                print(f"Model {model_file} for dataset {task}: {score}")
                # print(json.dumps(data, indent=2))     #print each item in jsonl
    print("\n")


# #FOR .JSONL files
# with open("./SBATCH_BASEMODEL_ALLDATASETS_OLMESEVAL-eval-gsm8k/metrics-all.jsonl", "r") as f:
#     for line in f:
#         data = json.loads(line)
#         model_file = data.get("model_config", {}).get("model")
#         model_file = model_file.split("/")[-1]  #get directory that holds model used for eval

#         task = data.get("task_config", {}).get("task_name")
#         score = data.get("metrics", {}).get("primary_score")
#         print(f"Model {model_file} for dataset {task}: {score}")
#         # print(json.dumps(data, indent=2))     #print each item in jsonl


#FOR .JSON files
# folder_path = "directory_that_holds all .json files"
# for filename in os.listdir(folder_path):
#     if filename.endswith(".json"):
#         file_path = os.path.join(folder_path, filename)
        
#         with open(file_path, "r") as f:
#             data = json.load(f)
        
#         primary_score = data["metrics"]["primary_score"]
#         print(f"{filename}: {primary_score}")