import json
import os

print("\n\n")

#assumes in local directory, but path_name given 
list_of_paths = [
    "BASEMODEL_CUSTOMROLECOLONCHATTEMP_ALL5DATASETS",
    "SFTMODEL_FULLROLECOLONRUN_CHECKPOINT3000_VALIDMERGE",
    "SFTMODEL_FULLROLECOLONRUN_CHECKPOINT7000_VALIDMERGE",
    "SFTMODEL_FULLROLECOLONRUN_CHECKPOINT15000_VALIDMERGE",
    "SFTMODEL_FULLROLECOLONRUN_CHECKPOINTFINAL_additionalconfigsfromtransformers4library",
    "SFTMODEL_FULLROLECOLONRUN_CHECKPOINTFINAL_withllama3chattemplate",
    "DPOMODEL_FULLSFTROLECOLONRUN_DPOTINKERCHECKPOINT3000"
]

#list of tasks names done by OLMES eval
list_of_datasets = [
    "gsm8k",
    "ifeval",
    "mbpp",
    "harmbench::default",
    "xstest::default",
    "codex_humanevalplus"
]

#FOR .JSONL files
#extracts primary scores from metric-all.jsonl

for dataset in list_of_datasets:
    for filename in list_of_paths:
        path = f"./{filename}-eval-{dataset}/metrics-all.jsonl"
        try:
            with open(path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    model_file = data.get("model_config", {}).get("model")
                    model_file = model_file.split("/")[-1]  #get directory that holds model used for eval

                    task = data.get("task_config", {}).get("task_name")
                    score = data.get("metrics", {}).get("primary_score")
                    print(f"{filename} Model {model_file} for dataset {task}: {score}")
        except FileNotFoundError:
            print(f"Skipping missing file: {path}")
    print("\n\n")
