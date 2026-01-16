#!/usr/bin/env python3
import re
from pathlib import Path
import matplotlib.pyplot as plt

# 两个输入文件
infile_nosplit = Path("posterior_leakage_results/all_summary_7_noiseonly.txt")
infile_split = Path("posterior_leakage_results/all_summary_7_noiseonly_split.txt")

# Example line:
# eval_results_noise_2p0_abstract
pat_noise = re.compile(
    r"Posterior Dataset:\s*eval_results_noise_([0-9]+)p0_abstract"
)

def parse_file(infile: Path) -> dict:
    """解析单个文件，返回 {noise: (mean, std, median, max_pl)}"""
    text = infile.read_text(encoding="utf-8")
    lines = text.splitlines()
    data = {}

    for i, line in enumerate(lines):
        m = pat_noise.search(line)
        if not m:
            continue

        noise = int(m.group(1))
        mean = std = median = max_pl = None

        for j in range(i + 1, min(i + 40, len(lines))):
            s = lines[j].strip()
            if s.startswith("Mean PL:"):
                mean = float(s.split(":", 1)[1].strip())
            elif s.startswith("Std PL:"):
                std = float(s.split(":", 1)[1].strip())
            elif s.startswith("Median PL:"):
                median = float(s.split(":", 1)[1].strip())
            elif s.startswith("Max PL:"):
                max_pl = float(s.split(":", 1)[1].strip())

            if all(v is not None for v in [mean, std, median, max_pl]):
                break

        if mean is not None:
            data[noise] = (mean, std, median, max_pl)

    return data

# 解析两个文件
data_nosplit = parse_file(infile_nosplit)
data_split = parse_file(infile_split)

if not data_nosplit and not data_split:
    raise SystemExit("No PL data parsed from files.")

# 获取所有 noise 值的并集
all_noises = sorted(set(data_nosplit.keys()) | set(data_split.keys()))

# 提取数据
def extract_metrics(data, noises):
    means = [data.get(n, (None,))[0] for n in noises]
    stds = [data.get(n, (None, None))[1] for n in noises]
    meds = [data.get(n, (None, None, None))[2] for n in noises]
    maxs = [data.get(n, (None, None, None, None))[3] for n in noises]
    return means, stds, meds, maxs

means_nosplit, stds_nosplit, meds_nosplit, maxs_nosplit = extract_metrics(data_nosplit, all_noises)
means_split, stds_split, meds_split, maxs_split = extract_metrics(data_split, all_noises)

# 绘图
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 14))

# Mean PL
axs[0].plot(all_noises, means_nosplit, marker="o", color="C0", label="No Split")
axs[0].plot(all_noises, means_split, marker="s", color="C1", linestyle="--", label="Split")
axs[0].set_ylabel("Mean PL")
axs[0].grid(True)
axs[0].legend()

# Std PL
axs[1].plot(all_noises, stds_nosplit, marker="o", color="C0", label="No Split")
axs[1].plot(all_noises, stds_split, marker="s", color="C1", linestyle="--", label="Split")
axs[1].set_ylabel("Std PL")
axs[1].grid(True)
axs[1].legend()

# Median PL
axs[2].plot(all_noises, meds_nosplit, marker="o", color="C0", label="No Split")
axs[2].plot(all_noises, meds_split, marker="s", color="C1", linestyle="--", label="Split")
axs[2].set_ylabel("Median PL")
axs[2].grid(True)
axs[2].legend()

# Max PL
axs[3].plot(all_noises, maxs_nosplit, marker="o", color="C0", label="No Split")
axs[3].plot(all_noises, maxs_split, marker="s", color="C1", linestyle="--", label="Split")
axs[3].set_ylabel("Max PL")
axs[3].set_xlabel("Budget (ε)")
axs[3].grid(True)
axs[3].legend()

fig.suptitle("PL Statistics vs Budget: No Split vs Split Comparison")
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = Path("posterior_leakage_results/pl_vs_budget_7_noiseonly.png")
fig.savefig(out_path, dpi=200)
print(f"Saved plot to {out_path}")