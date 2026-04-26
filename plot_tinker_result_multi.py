import json
import numpy as np
import matplotlib.pyplot as plt

# ============================
# Config
# ============================
paths = {
    "No Filter": r"logs\sft-meta-llama-Llama-3.2-1B-20260419-200840\metrics.jsonl",
    "Length Only": r"logs\sft-meta-llama-Llama-3.2-1B-20260420-215100\metrics.jsonl",
    "English Only": r"logs\sft-meta-llama-Llama-3.2-1B-20260421-142656\metrics.jsonl",
    "English + Length": r"logs\sft-meta-llama-Llama-3.2-1B-20260421-233056\metrics.jsonl"
}

# ============================
# Load function
# ============================
def load_test_nll(path):
    steps = []
    nlls = []

    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            if "test/nll" in data:
                steps.append(data["step"])
                nlls.append(data["test/nll"])

    return np.array(steps), np.array(nlls)

# ============================
# Load all data
# ============================
data = {}

for name, path in paths.items():
    steps, nll = load_test_nll(path)

    print(f"{name}: {len(nll)} points")  # 🔍 debug check

    data[name] = (steps, nll)

# ============================
# Plot (RAW ONLY)
# ============================
plt.figure(figsize=(10, 6))

for name, (steps, nll) in data.items():

    if len(nll) == 0:
        continue  # skip empty logs

    plt.plot(
        steps,
        nll,
        label=name,
        linewidth=2,
        marker='o',        # points ON
        markersize=3,
        markevery=5       # reduce clutter
    )

# ============================
# Styling (paper-ready)
# ============================
plt.xlabel("Training Steps")
plt.ylabel("Test NLL")
plt.title("Test NLL Across Data Filtering Strategies")

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# ============================
# Train NLL loader (ADDED)
# ============================
def load_train_nll(path):
    steps = []
    nlls = []

    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            if "train_mean_nll" in data:
                steps.append(data["step"])
                nlls.append(data["train_mean_nll"])

    return np.array(steps), np.array(nlls)


# ============================
# Downsampling settings
# ============================
window = 100
sample_every = 50

def moving_average(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode='valid')


# ============================
# Plot: TRAIN NLL
# ============================
plt.figure(figsize=(10, 6))

for name, path in paths.items():

    steps, nll = load_train_nll(path)

    if len(nll) == 0:
        continue

    # smoothing + downsampling
    nll_smooth = moving_average(nll, window)
    steps_trim = steps[len(steps) - len(nll_smooth):]

    steps_ds = steps_trim[::sample_every]
    nll_ds = nll_smooth[::sample_every]

    plt.plot(
        steps_ds,
        nll_ds,
        label=name,
        linewidth=2
    )

# ============================
# Styling
# ============================
plt.xlabel("Training Steps")
plt.ylabel("Train NLL")
plt.title("Train NLL Across Data Filtering Strategies")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()