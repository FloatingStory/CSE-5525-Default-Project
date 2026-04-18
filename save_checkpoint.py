import json
import requests
import tinker

from tinker_cookbook import checkpoint_utils

LOG_PATH = "logs\sft-meta-llama-Llama-3.2-1B-20260330-103522"
METRICS_FILE = f"{LOG_PATH}/metrics.jsonl"
OUTPUT_FILE = "sft_best_checkpoint.tar.gz"
CHECKPOINTS_FILE = f"{LOG_PATH}/checkpoints.jsonl"

# # =========================
# # 1. find best step from metrics.jsonl
# # =========================
# best_step = None
# best_loss = float("inf")

# with open(METRICS_FILE, "r") as f:
#     for line in f:
#         data = json.loads(line)

#         loss = data.get("train_mean_nll")
#         step = data.get("step")

#         if loss is None or step is None:
#             continue

#         if loss < best_loss:
#             best_loss = loss
#             best_step = step

# print(f"[INFO] Best step: {best_step}, loss: {best_loss}")

# # =========================
# # 2. list checkpoints in the log directory
# # =========================
# best_ckpt = None
# min_diff = float("inf")

# with open(CHECKPOINTS_FILE, "r") as f:
#     for line in f:
#         data = json.loads(line)

#         ckpt_step = data.get("step")
#         tinker_path = data.get("sampler_path")
#         name = data.get("name")

#         if ckpt_step is None:
#             try:
#                 ckpt_step = int(name.split("_")[-1])
#             except:
#                 continue

#         diff = abs(ckpt_step - best_step)

#         if diff < min_diff:
#             min_diff = diff
#             best_ckpt = data

# if best_ckpt is None:
#     raise RuntimeError("No matching checkpoint found")

# print(f"[INFO] Selected checkpoint: {best_ckpt['name']}")
# print(f"[INFO] Step diff: {min_diff}")
# print(f"[INFO] Tinker path: {best_ckpt['sampler_path']}")

# =========================
# 3. download checkpoint
# =========================
service_client = tinker.ServiceClient()
rest_client = service_client.create_rest_client()

future = rest_client.get_checkpoint_archive_url_from_tinker_path(
    # best_ckpt['sampler_path']
    "tinker://b5ec366a-0335-5964-ac0a-9abc3b5289db:train:0/sampler_weights/final"
)

url = future.result().url

print("[INFO] Downloading...")

r = requests.get(url)
r.raise_for_status()

with open(OUTPUT_FILE, "wb") as f:
    f.write(r.content)

print(f"[DONE] Saved to {OUTPUT_FILE}")



# =========================
# checkpoint output
# [INFO] Best step: 30, loss: 0.1397583782672882
# [INFO] Selected checkpoint: 000050
# [INFO] Step diff: 20
# [INFO] Tinker path: tinker://92cb9574-7384-5473-8f5f-4930fcb50e2f:train:0/sampler_weights/000050
# Creating checkpoint archive for run id: 92cb9574-7384-5473-8f5f-4930fcb50e2f:train:0 checkpoint id: sampler_weights/000050 (this may take a while)
# [INFO] Downloading...
# [DONE] Saved to sft_best_checkpoint.tar.gz