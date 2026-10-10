"""Plot speed vs accuracy and model size for every exported version of the final model."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"pytorch": "tab:blue", "onnx": "tab:gray", "openvino": "tab:green", "executorch": "tab:red"}

with open(ROOT / "analysis" / "efficiency.csv") as f:
    rows = list(csv.DictReader(f))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# left: each variant as a dot, bigger dot = bigger file
for r in rows:
    runtime = r["variant"].split()[0]
    ms, acc, size = float(r["cpu_ms"]), float(r["mAP50-95"]), float(r["size_MB"])
    ax1.scatter(ms, acc, s=size * 40, color=COLORS.get(runtime, "black"), alpha=0.7)
    ax1.annotate(r["variant"], (ms, acc), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax1.set_xlabel("CPU latency on M2 (ms, lower is better)")
ax1.set_ylabel("mAP50-95 on validation")
ax1.set_title("Speed vs accuracy (dot size = file size)")
ax1.grid(alpha=0.3)

# right: file size per variant
names = [r["variant"] for r in rows]
sizes = [float(r["size_MB"]) for r in rows]
ax2.barh(names, sizes, color=[COLORS.get(n.split()[0], "black") for n in names])
ax2.invert_yaxis()
ax2.set_xlabel("size (MB)")
ax2.set_title("Model size")

plt.tight_layout()
out = ROOT / "analysis" / "efficiency.png"
plt.savefig(out, dpi=150)
print("saved", out)
