"""Regenerate the report's roadmap and Graph history from its measurement data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'attention_bf16_measurements.json').read_text())
ROWS = [r for r in DATA['measurements'] if r['id'].startswith('G')]
FONTS = {f.name for f in font_manager.fontManager.ttflist}
for name in ('Arial Unicode MS', 'Noto Sans CJK SC', 'Heiti TC', 'Heiti SC'):
    if name in FONTS:
        plt.rcParams['font.family'] = name
        break
else:
    raise RuntimeError('Install a Chinese font: Noto Sans CJK SC or Arial Unicode MS.')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
TEAL, GRAY, INK = '#087f8c', '#a9b6bf', '#172a36'

fig, ax = plt.subplots(figsize=(13.5, 10.2), dpi=130)
for i, r in enumerate(ROWS):
    ax.barh(i - .19, r['sdpa_us'], height=.33, color=GRAY,
            label='同轮 SDPA' if i == 0 else None)
    ax.barh(i + .19, r['student_us'], height=.33, color=TEAL,
            label='student' if i == 0 else None)
    for offset, key in ((-.19, 'sdpa_us'), (.19, 'student_us')):
        ax.text(r[key] + 1.3, i + offset, f"{r[key]:.2f}", va='center', fontsize=11)
ax.set_yticks(range(len(ROWS)), [f"{r['id']} {r['change']}" + ('*' if r['mode']=='graph*' else '') for r in ROWS])
ax.invert_yaxis()
ax.set_xlim(0, 111)
ax.set_xlabel('耗时 µs · 越低越好', loc='right', color='#536b7d')
ax.xaxis.grid(True, color='#e5ecf1')
ax.set_axisbelow(True)
ax.tick_params(axis='both', length=0, pad=9)
for spine in ('top', 'right', 'left'):
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#cbd9e3')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='lower left', bbox_to_anchor=(0, 1.015),
          ncol=2, frameon=False, columnspacing=4)
fig.subplots_adjust(left=.30, right=.975, top=.935, bottom=.075)
fig.savefig(ROOT / 'assets/attention_graph_path.png', facecolor='white')
plt.close(fig)

latest = ROWS[-1]['student_us']
fig, ax = plt.subplots(figsize=(13.5, 2.3), dpi=130)
ax.set_xlim(0, 4.24)
ax.set_ylim(0, 1)
ax.axis('off')
boxes = [
    ('01 正确性闭环', 'BF16 绑定与边界', '先保证可以比较'),
    ('02 定位 GPU 开销', 'Graph 与 NCU', '排除测量误导'),
    ('03 调整资源与流水', 'tile 预取 swizzle', '逐项评估取舍'),
    ('04 减少非 MMA 工作', 'mask 索引 归约 特化', f'最新 {latest:.2f} µs'),
]
for i, lines in enumerate(boxes):
    x = i * 1.08
    ax.add_patch(Rectangle((x, .12), 1, .8, facecolor='#edf4f7', edgecolor='#cbdbe4'))
    for y, text in zip((.74, .51, .28), lines):
        ax.text(x+.5, y, text, ha='center', va='center', color=INK, fontsize=12.5)
    if i < 3:
        ax.annotate('', xy=(x+1.08, .52), xytext=(x+1, .52),
                    arrowprops=dict(arrowstyle='-|>', color=TEAL))
fig.subplots_adjust(left=.004, right=.996, top=.98, bottom=.02)
fig.savefig(ROOT / 'assets/attention_roadmap.png', facecolor='white')
plt.close(fig)
