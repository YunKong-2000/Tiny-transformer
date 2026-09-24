"""Regenerate checked-in documentation figures (optional dependency: matplotlib)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, ORANGE, GRAY, DARK = "#dbeafe", "#dcfce7", "#ffedd5", "#e2e8f0", "#172554"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    return fig, ax


def box(ax, x, y, w, h, text, color=BLUE, size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", linewidth=1.1, edgecolor="#64748b", facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=DARK, fontsize=size)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.5})


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


fig, ax = canvas(12, 10)
ax.text(0.2, 9.6, "Decoder-only Transformer: shapes and residual paths", fontsize=17, weight="bold", color=DARK)
items = [(8.4, "Token IDs  $[B,L]$", GRAY), (7.25, "Embedding  $[V,d_{\\mathrm{model}}]$\noutput $[B,L,d_{\\mathrm{model}}]$", BLUE),
         (5.85, "Decoder Block $\\times N_{\\mathrm{layers}}$ (= 8)\noutput $[B,L,d_{\\mathrm{model}}]$", GREEN), (4.45, "Final RMSNorm\n$[B,L,d_{\\mathrm{model}}]$", BLUE),
         (3.05, "LM Head (tied embedding weight)\n$[B,L,d_{\\mathrm{model}}] \\to [B,L,V]$", BLUE), (1.65, "Cross Entropy\nlogits + shifted targets -> scalar loss", ORANGE)]
for index, (y, text, color) in enumerate(items):
    box(ax, 0.3, y, 5, 0.85, text, color)
    if index:
        arrow(ax, 2.8, items[index - 1][0], 2.8, y + 0.85)
ax.text(0.4, 0.8, "Training: all $L$ positions predict in parallel.\nGeneration: use the last position to choose a new token.", fontsize=10, color=DARK)
ax.plot([5.7, 5.7], [0.8, 9.1], color="#cbd5e1", linestyle="--")
ax.text(7.0, 8.9, "Inside one pre-norm block", fontsize=13, weight="bold", color=DARK)
ys = [7.75, 6.65, 5.55, 4.45, 3.35, 2.25, 1.15]
texts = ["$x$  $[B,L,d_{\\mathrm{model}}]$", "RMSNorm", "QKV -> RoPE -> Attention -> O", "Add residual: x + attention(x)", "RMSNorm", "Gate / Up -> SiLU * Up -> Down", "Add residual: h + FFN(h)"]
for index, (y, text) in enumerate(zip(ys, texts)):
    box(ax, 6.5, y, 5, 0.65, text, GREEN if index in (3, 6) else BLUE, 10)
    if index:
        arrow(ax, 9.0, ys[index - 1], 9.0, y + 0.65)
ax.plot([6.5, 6.05, 6.05, 6.5], [8.07, 8.07, 4.77, 4.77], color="#ea580c", lw=1.5)
ax.plot([11.5, 11.9, 11.9, 11.5], [4.77, 4.77, 1.47, 1.47], color="#ea580c", lw=1.5)
save(fig, "architecture")

fig, ax = canvas(12, 5.5)
ax.text(0.2, 5.1, "Causal attention: rows are queries, columns are keys", fontsize=17, weight="bold", color=DARK)
for origin, past, qlen, klen, title in [(0.9, 0, 4, 4, "Prefill: $L_q = L_k = 4$"), (6.1, 4, 2, 6, "Cached chunk: past = 4, $L_q = 2$, $L_k = 6$")]:
    ax.text(origin, 4.4, title, fontsize=11, weight="bold", color=DARK)
    for row in range(qlen):
        ax.text(origin - 0.25, 3.65 - row * 0.65, str(past + row), ha="right", va="center")
        for col in range(klen):
            x, y = origin + col * 0.65, 3.35 - row * 0.65
            allowed = col <= past + row
            ax.add_patch(Rectangle((x, y), 0.6, 0.6, facecolor=GREEN if allowed else GRAY, edgecolor="white"))
            ax.text(x + 0.3, y + 0.3, "yes" if allowed else "no", ha="center", va="center", fontsize=9)
            if row == 0:
                ax.text(x + 0.3, 4.1, str(col), ha="center", fontsize=10)
ax.text(0.9, 0.65, "Allowed iff key_position <= past_length + query_index.\nMasked scores become -infinity BEFORE softmax.", fontsize=11, color=DARK)
ax.text(6.1, 1.35, "Single-token decode sees every cached key.\nA naive non-square triangular mask is wrong.\nIsolated packing also requires same document ID.", fontsize=11, color=DARK)
save(fig, "causal-mask")

fig, ax = canvas(12, 5.8)
ax.text(0.2, 5.35, "KV cache: prefill once, append one position per decode step", fontsize=16, weight="bold", color=DARK)
for y, title, live in [(4.05, "After prefill", 4), (2.8, "After one decode", 5), (1.55, "After two decodes", 6)]:
    ax.text(0.25, y + 0.32, title, fontsize=11, va="center", color=DARK)
    for col in range(8):
        color = BLUE if col < 4 else (GREEN if col < live else GRAY)
        box(ax, 3.0 + col * 1.02, y, 0.88, 0.65, f"K{col},V{col}" if col < live else "free", color, 10)
ax.text(0.3, 0.5, "Per-layer storage: $[B,H,C,d_h]$. Only K and V are cached.\nRotated K is stored once. Cache length advances after all layers complete.", fontsize=11, color=DARK)
save(fig, "kv-cache")

fig, ax = canvas(12, 7)
ax.text(0.2, 6.55, "CuTe mental model: coordinate -> layout -> storage offset", fontsize=16, weight="bold", color=DARK)
for origin, strides, title in [(0.7, (3, 1), "shape=(2,3), stride=(3,1)"), (6.4, (1, 2), "shape=(2,3), stride=(1,2)")]:
    ax.text(origin, 5.8, title, fontsize=12, weight="bold", color=DARK)
    for row in range(2):
        for col in range(3):
            offset = row * strides[0] + col * strides[1]
            box(ax, origin + col * 1.35, 4.35 - row * 1.05, 1.15, 0.85,
                f"({row},{col})\noffset {offset}", GREEN if (row, col) == (1, 0) else BLUE, 11)
    ax.text(origin, 2.7, f"offset(i,j) = {strides[0]} * i + {strides[1]} * j", fontsize=12, color=DARK)
box(ax, 0.7, 0.6, 3.0, 1.05, "Logical tensor\nwhich element?", BLUE)
box(ax, 4.4, 0.6, 3.0, 1.05, "Thread / value mapping\nwhich thread owns it?", GREEN)
box(ax, 8.1, 0.6, 3.0, 1.05, "Physical memory\nwhich address / transaction?", ORANGE)
arrow(ax, 3.75, 1.1, 4.35, 1.1)
arrow(ax, 7.45, 1.1, 8.05, 1.1)
save(fig, "cute-layout")
print(f"Wrote SVG and PNG figures to {OUT}")
