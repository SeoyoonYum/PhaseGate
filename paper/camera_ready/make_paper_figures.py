from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

# Embed TrueType fonts in PDF figures; avoid Type 3 fonts in the final paper.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = 9

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Figure 1: architectural motivation and policy tradeoff.
fig, (arch, sched) = plt.subplots(
    1, 2, figsize=(7.35, 2.22), gridspec_kw={"width_ratios": [0.36, 0.64]}
)

# (a) Discrete accelerator memory versus unified memory.
arch.set_xlim(0, 10)
arch.set_ylim(0, 10)
arch.axis("off")
arch.set_title("(a) Where contention enters", fontsize=9.2, pad=1)

def box(ax, xy, wh, label, fc="0.94", lw=0.8, fontsize=7.1):
    ax.add_patch(Rectangle(xy, wh[0], wh[1], facecolor=fc,
                           edgecolor="black", linewidth=lw))
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, label,
            ha="center", va="center", fontsize=fontsize)

arch.text(0.2, 8.95, "Server with discrete GPU", fontsize=7.7, fontweight="bold")
box(arch, (0.3, 7.1), (2.4, 1.25), "CPU search")
box(arch, (3.1, 7.1), (2.4, 1.25), "host DRAM")
box(arch, (7.2, 7.1), (2.4, 1.25), "GPU LLM")
box(arch, (7.2, 5.35), (2.4, 1.15), "device HBM", fc="0.82")
arch.add_patch(FancyArrowPatch((2.7, 7.72), (3.1, 7.72), arrowstyle="<->", lw=0.8))
arch.add_patch(FancyArrowPatch((8.4, 7.1), (8.4, 6.5), arrowstyle="<->", lw=0.8))
arch.plot([5.95, 5.95], [5.1, 8.55], color="0.35", lw=0.7, ls=":")
arch.text(5.95, 4.72, "separate pools", fontsize=6.8, ha="center")

arch.text(0.2, 3.6, "On-device unified-memory SoC", fontsize=7.7, fontweight="bold")
box(arch, (0.4, 1.65), (2.4, 1.25), "CPU search")
box(arch, (7.2, 1.65), (2.4, 1.25), "GPU LLM")
box(arch, (3.55, 0.25), (2.9, 1.25), "unified memory", fc="0.70")
arch.add_patch(FancyArrowPatch((2.8, 1.95), (4.0, 1.25), arrowstyle="->", lw=1.0))
arch.add_patch(FancyArrowPatch((7.2, 1.95), (6.0, 1.25), arrowstyle="->", lw=1.0))
arch.text(5.0, 2.2, "compete", fontsize=7.2, ha="center", fontweight="bold")

# (b) Fixed-N tradeoff and PhaseGate.
sched.set_xlim(0, 12.8)
sched.set_ylim(0, 4.65)
sched.axis("off")
sched.set_title("(b) Phase-aware admission moves the tradeoff", fontsize=9.2, pad=1)
x0, split, x1 = 2.05, 5.15, 10.45

def gpu_lane(y):
    sched.add_patch(Rectangle((x0, y), split - x0, 0.40, fill=False, linewidth=0.85))
    sched.add_patch(Rectangle((split, y), x1 - split, 0.40, fill=False, linewidth=0.85))
    sched.text((x0 + split) / 2, y + 0.20, "LLM prefill", ha="center", va="center", fontsize=6.7)
    sched.text((split + x1) / 2, y + 0.20, "LLM decode", ha="center", va="center", fontsize=6.7)

def cpu_bars(xa, xb, y, count):
    shades = [0.28, 0.40, 0.52, 0.64]
    for i in range(count):
        sched.add_patch(Rectangle((xa, y + 0.095 * i), xb - xa, 0.066,
                                  facecolor=str(shades[i]), edgecolor="black", linewidth=0.25))

rows = [(3.55, "Fixed-1"), (2.10, "Fixed-2"), (0.65, "PhaseGate 4→1")]
for y, label in rows:
    sched.text(x0 - 0.20, y + 0.20, label, fontsize=8.1, fontweight="bold",
               ha="right", va="center")
    gpu_lane(y)

cpu_bars(x0, x1, 3.18, 1)
sched.text(x1 + 0.55, 3.43, "LATENCY PASS", fontsize=6.4, va="center",
           color="#006b3c", fontweight="bold",
           bbox={"boxstyle": "round,pad=0.16", "facecolor": "#e5f5ec",
                 "edgecolor": "#006b3c", "linewidth": 0.6})
sched.text(x1 + 0.55, 3.08, "LOW retrieval", fontsize=6.4, va="center")

cpu_bars(x0, x1, 1.55, 2)
sched.text(x1 + 0.55, 1.88, "LATENCY FAIL", fontsize=6.4, va="center",
           color="#a31220", fontweight="bold",
           bbox={"boxstyle": "round,pad=0.16", "facecolor": "#fde8ea",
                 "edgecolor": "#a31220", "linewidth": 0.6})
sched.text(x1 + 0.55, 1.53, "HIGH retrieval", fontsize=6.4, va="center")

cpu_bars(x0, split, 0.10, 4)
cpu_bars(split, x1, 0.28, 1)
sched.text(x1 + 0.55, 0.58, "LATENCY PASS", fontsize=6.4, va="center",
           color="#006b3c", fontweight="bold",
           bbox={"boxstyle": "round,pad=0.16", "facecolor": "#e5f5ec",
                 "edgecolor": "#006b3c", "linewidth": 0.6})
sched.text(x1 + 0.55, 0.23, "HIGH retrieval", fontsize=6.4, va="center")
sched.add_patch(FancyArrowPatch((x1 + 0.18, 3.26), (x1 + 0.18, 0.35),
                               arrowstyle="<->", lw=0.65))
sched.text(x1 - 0.12, 2.92, "same decode\nadmission cap", fontsize=6.3,
           ha="right", va="center")

fig.tight_layout(pad=0.25, w_pad=0.55)
fig.savefig(OUT / "phasegate_overview.pdf", bbox_inches="tight")
fig.savefig(OUT / "phasegate_overview.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 2: mechanism and base-M4 output-shape sensitivity.
fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.38))

ax = axes[0]
labels = ["prefill", "decode"]
air_slowdown = np.array([5.7, 61.3])
mini_slowdown = np.array([6.9, 59.8])
x = np.arange(2)
width = 0.31
ax.bar(x - width / 2, air_slowdown, width=width, edgecolor="black",
       facecolor="0.75", label="M4 Air")
ax.bar(x + width / 2, mini_slowdown, width=width, edgecolor="black",
       facecolor="0.42", label="M4 Mini")
ax.set_xticks(x, labels)
ax.set_ylabel("p95 slowdown (%)")
ax.set_ylim(0, 72)
ax.grid(axis="y", linestyle=":", linewidth=0.6)
for offset, values in [(-width / 2, air_slowdown), (width / 2, mini_slowdown)]:
    for i, value in enumerate(values):
        ax.text(i + offset, value + 1.6, f"+{value:.2g}%", ha="center",
                va="bottom", fontsize=6.9)
ax.legend(frameon=False, fontsize=7.0, loc="upper left")
ax.set_title("(a) HNSW phase asymmetry on two M4 devices")

ax = axes[1]
x = np.arange(3)
labels = ["64", "128", "512"]
m4_ratio = np.array([2.3039, 2.0137, 1.4329])
m4_prefill = np.array([69.08, 51.60, 20.65])
ax.plot(x, m4_ratio, marker="o", markersize=4.4, linewidth=1.5,
        linestyle="-", label="M4 throughput ratio")
ax.set_xticks(x, labels)
ax.set_xlabel("output tokens")
ax.set_ylabel("PhaseGate / Fixed throughput ($\\times$)")
ax.set_ylim(1.0, 2.5)
ax.set_yticks([1.0, 1.5, 2.0, 2.5])
ax.grid(linestyle=":", linewidth=0.6)
ax2 = ax.twinx()
ax2.plot(x, m4_prefill, marker="^", markersize=4.2, linewidth=1.25,
         color="0.35", linestyle=":", label="M4 prefill fraction")
ax2.set_ylabel("prefill fraction (%)", color="0.35")
ax2.set_ylim(0, 80)
ax2.tick_params(axis="y", colors="0.35")
lines, line_labels = ax.get_legend_handles_labels()
lines2, line_labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, line_labels + line_labels2,
          frameon=False, fontsize=6.8, loc="upper right")
ax.set_title("(b) Throughput ratio versus output length")

fig.tight_layout(pad=0.52, w_pad=1.2)
fig.savefig(OUT / "mechanism_and_sensitivity.pdf", bbox_inches="tight")
fig.savefig(OUT / "mechanism_and_sensitivity.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Appendix figure: base-M2 output-length replication, kept separate from M4.
fig, ax = plt.subplots(figsize=(3.55, 2.25))
m2_tokens = np.array([128, 512])
m2_ratio = np.array([1.8050, 1.3653])
ax.plot(m2_tokens, m2_ratio, marker="s", markersize=5.0, linewidth=1.5,
        color="#d95f02", linestyle="-", label="M2 throughput ratio")
for token_count, ratio in zip(m2_tokens, m2_ratio):
    ax.annotate(f"{ratio:.2g}$\\times$", (token_count, ratio),
                xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=7.4)
ax.set_xticks(m2_tokens, ["128", "512"])
ax.set_xlabel("output tokens")
ax.set_ylabel("PhaseGate / Fixed throughput ($\\times$)")
ax.set_ylim(1.0, 2.1)
ax.set_yticks([1.0, 1.5, 2.0])
ax.grid(linestyle=":", linewidth=0.6)
ax.legend(frameon=False, fontsize=7.1, loc="upper right")
ax.set_title("M2 Mini output-length replication")
fig.tight_layout(pad=0.55)
fig.savefig(OUT / "m2_output_length.pdf", bbox_inches="tight")
fig.savefig(OUT / "m2_output_length.png", dpi=300, bbox_inches="tight")
plt.close(fig)
