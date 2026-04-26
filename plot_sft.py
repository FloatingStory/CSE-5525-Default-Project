import json
import numpy as np
import matplotlib.pyplot as plt

METRICS_FILE = r"logs\sft-meta-llama-Llama-3.2-1B-20260420-215100\metrics.jsonl"

# ============================
# 1. read file
# ============================
steps = []
nlls = []
lrs = []
times = []
test_steps = []
test_nlls = []

with open(METRICS_FILE, "r") as f:
    for line in f:
        data = json.loads(line)
        steps.append(data["step"])
        nlls.append(data["train_mean_nll"])
        lrs.append(data["learning_rate"])
        times.append(data.get("time/step", 0.0))
        if "test/nll" in data:
            test_steps.append(data["step"])
            test_nlls.append(data["test/nll"])

# ============================
# 2. smoothing & downsampling
# ============================
window = 100
sample_every = 50

def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, mode='valid')

nll_smooth = moving_average(nlls, window)[::sample_every]
lr_smooth = moving_average(lrs, window)[::sample_every]
time_smooth = moving_average(times, window)[::sample_every]
steps_smooth = moving_average(steps, window)[::sample_every]  # step도 맞춰줌

# ============================
# 3. Plot: NLL vs Step (Smoothed)
# ============================
plt.figure(figsize=(10, 5))
plt.plot(steps_smooth, nll_smooth, label="train_mean_nll (smoothed)")
plt.xlabel("Step")
plt.ylabel("NLL")
plt.title("Training NLL over Steps (Smoothed & Downsampled)")
plt.grid(True)
plt.legend()
plt.show()

# ============================
# 3.5 Plot: Test NLL vs Step (raw, no smoothing)
# ============================
plt.figure(figsize=(10, 5))
plt.plot(test_steps, test_nlls, marker="o", linestyle="-", label="test/nll")
plt.xlabel("Step")
plt.ylabel("Test NLL")
plt.title("Test NLL over Steps")
plt.grid(True)
plt.legend()
plt.show()

# ============================
# 4. Plot: Learning Rate vs Step (Smoothed)
# ============================
plt.figure(figsize=(10, 5))
plt.plot(steps_smooth, lr_smooth, label="learning_rate (smoothed)", color="orange")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule over Steps (Smoothed & Downsampled)")
plt.grid(True)
plt.legend()
plt.show()

# ============================
# 5. Plot: NLL + Learning Rate (Secondary Axis)
# ============================
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(steps_smooth, nll_smooth, color="blue", label="NLL")
ax1.set_xlabel("Step")
ax1.set_ylabel("NLL", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(steps_smooth, lr_smooth, color="orange", label="LR")
ax2.set_ylabel("Learning Rate", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")

fig.tight_layout()
plt.title("Training NLL and Learning Rate (Smoothed & Downsampled)")
plt.show()

# ============================
# 6. Optional: Time per Step
# ============================
plt.figure(figsize=(10, 5))
plt.plot(steps_smooth, time_smooth, label="time/step (smoothed)", color="green")
plt.xlabel("Step")
plt.ylabel("Time per Step (s)")
plt.title("Training Speed (Smoothed & Downsampled)")
plt.grid(True)
plt.legend()
plt.show()