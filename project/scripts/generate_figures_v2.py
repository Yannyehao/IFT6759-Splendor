"""Generate Yan's presentation figures in unified PPT style.
Style: white background, blue/dark palette, clean bars, CI boxes, callout boxes."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = "project/docs/figures/v2"
os.makedirs(OUT, exist_ok=True)

# Style matching the PPT
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'font.family': 'sans-serif',
    'font.size': 11,
    'figure.dpi': 200,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors from the PPT style
C = {
    'primary': '#2B5797',      # dark blue
    'secondary': '#4BACC6',    # light blue
    'accent': '#F79646',       # orange
    'success': '#4CAF50',      # green
    'danger': '#E74C3C',       # red
    'gray': '#95A5A6',         # gray
    'dark': '#2C3E50',         # dark
    'light': '#ECF0F1',        # light bg
    'gold': '#F1C40F',         # gold
    'bg_box': '#F5F7FA',       # box background
}

def add_callout_box(ax, x, y, w, h, text, icon='', bg='#F5F7FA', border='#2B5797'):
    """Add a callout box at the bottom of chart."""
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=bg, edgecolor=border, linewidth=1.5,
                                    transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, f'{icon} {text}', transform=ax.transAxes,
            ha='center', va='center', fontsize=9, fontweight='bold', color=C['dark'])

# ============================================================
# 1. AlphaZero Failure — clean table style
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

data = [
    ['V3 Shaped (30 iter, 50 sims)', '34.0%', '4.0%', 'Reward shaping insufficient'],
    ['V4 Warm-start (60 iter)', '6.0%', '2.0%', 'PPO distillation hurt performance'],
    ['V3-long (60 iter)', '10.0%', '0.0%', 'More training did not help'],
    ['Stage C (40 iter, 50 sims)', '30.0%', '5.0%', 'Best canonical run'],
]
cols = ['AlphaZero Variant', 'vs Random', 'vs Greedy', 'Diagnosis']

table = ax.table(cellText=data, colLabels=cols, loc='center', cellLoc='center',
                 colWidths=[0.30, 0.12, 0.12, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.2)

for j in range(len(cols)):
    table[0, j].set_facecolor(C['primary'])
    table[0, j].set_text_props(color='white', fontweight='bold', fontsize=11)

for i in range(1, len(data)+1):
    for j in range(len(cols)):
        table[i, j].set_facecolor('white' if i % 2 == 1 else C['light'])
        table[i, j].set_edgecolor('#ddd')
    table[i, 2].set_text_props(color=C['danger'], fontweight='bold')

fig.suptitle('AlphaZero: All Variants Failed', fontsize=16, fontweight='bold',
             color=C['dark'], y=0.95)
fig.text(0.5, 0.08, '⚠  All variants < 10% vs Greedy  |  PPO Score-based already achieves 75.8%',
         ha='center', fontsize=10, color=C['danger'], fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDEDEC', edgecolor=C['danger']))

plt.tight_layout(rect=[0, 0.12, 1, 0.92])
plt.savefig(f'{OUT}/02_alphazero_failure.png', bbox_inches='tight')
plt.close()
print("1. AlphaZero failure done")

# ============================================================
# 2. PPO+Lookahead Architecture
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.axis('off')
ax.set_xlim(0, 12)
ax.set_ylim(-0.5, 4.5)

boxes = [
    (0.3, 1.8, 2.0, 1.2, 'Legal Actions\n(30-80)', C['gray']),
    (3.0, 1.8, 2.0, 1.2, 'PPO Filter\nTop K=15', C['primary']),
    (5.8, 1.8, 2.2, 1.2, '1-Step Forward\nSimulation', C['secondary']),
    (8.6, 1.8, 2.2, 1.2, 'Event\nEvaluation', C['success']),
]
for x, y, w, h, text, color in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')

# Best Action box
rect = mpatches.FancyBboxPatch((8.6, 0), 2.2, 1.0, boxstyle="round,pad=0.15",
                                facecolor=C['gold'], edgecolor=C['dark'], linewidth=2)
ax.add_patch(rect)
ax.text(9.7, 0.5, '★ Best Action', ha='center', va='center',
        fontsize=13, fontweight='bold', color=C['dark'])

# Arrows
arrow_kw = dict(arrowstyle='->', color=C['dark'], lw=2)
ax.annotate('', xy=(3.0, 2.4), xytext=(2.3, 2.4), arrowprops=arrow_kw)
ax.annotate('', xy=(5.8, 2.4), xytext=(5.0, 2.4), arrowprops=arrow_kw)
ax.annotate('', xy=(8.6, 2.4), xytext=(8.0, 2.4), arrowprops=arrow_kw)
ax.annotate('', xy=(9.7, 1.8), xytext=(9.7, 1.0), arrowprops=arrow_kw)

# Formula
ax.text(6, -0.2, 'Score = 0.3 × PPO_prob  +  0.5 × event_reward  +  0.2 × future_value',
        fontsize=11, ha='center', style='italic', color=C['dark'],
        bbox=dict(boxstyle='round,pad=0.4', facecolor=C['bg_box'], edgecolor='#ccc'))

ax.set_title('PPO + Lookahead Architecture', fontsize=16, fontweight='bold',
             color=C['dark'], pad=15)
plt.tight_layout()
plt.savefig(f'{OUT}/03_lookahead_architecture.png', bbox_inches='tight')
plt.close()
print("2. Architecture done")

# ============================================================
# 3. Lookahead Results Overview
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.5, 1]})

# Left: bar chart
matchups = ['PPO alone\nvs Greedy', 'PPO+Lookahead\nvs Greedy', 'Lookahead\nvs PPO (H2H)', 'Self-play\n(sanity)']
rates = [78.0, 91.9, 82.2, 53.5]
colors = [C['gray'], C['primary'], C['secondary'], C['gray']]

bars = ax1.bar(matchups, rates, color=colors, width=0.55, edgecolor='white', linewidth=1.5)
ax1.set_ylabel('Win Rate (%)', fontsize=12, color=C['dark'])
ax1.set_ylim(0, 108)
ax1.axhline(y=50, color='#ccc', linestyle='--', alpha=0.7)
ax1.tick_params(colors=C['dark'])

for bar, val in zip(bars, rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=12, color=C['dark'])

# +14pp annotation
ax1.annotate('+14pp', xy=(1, 92), xytext=(0.5, 100),
             fontsize=12, fontweight='bold', color=C['primary'],
             arrowprops=dict(arrowstyle='->', color=C['primary'], lw=2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#DBEAFE', edgecolor=C['primary']))

ax1.set_title('Win Rate by Matchup (n=1000)', fontsize=13, fontweight='bold', color=C['dark'])

# Right: CI box
ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

ci_data = [
    ('vs Greedy', '91.9%', '[90.0%, 93.4%]', 8.5),
    ('vs PPO H2H', '82.2%', '[79.7%, 84.5%]', 6.5),
    ('Self-play', '53.5%', '[46.6%, 60.3%]', 4.5),
    ('K=30 best', '93.0%', '[91.2%, 94.4%]', 2.5),
]

ax2.text(5, 9.5, 'Statistical Significance (95% CI)', ha='center', fontsize=12,
         fontweight='bold', color=C['dark'])

for label, val, ci, y in ci_data:
    ax2.text(0.5, y + 0.3, label, fontsize=10, color=C['dark'])
    # CI bar
    lo, hi = float(ci.split(',')[0].strip('[% ')), float(ci.split(',')[1].strip(']% '))
    bar_x = (lo - 40) / 6  # scale to 0-10
    bar_w = (hi - lo) / 6
    rect = mpatches.Rectangle((bar_x + 1, y - 0.15), bar_w, 0.3,
                               facecolor=C['secondary'], alpha=0.7)
    ax2.add_patch(rect)
    ax2.text(9, y, f'{val}', fontsize=11, fontweight='bold', color=C['primary'],
             ha='right')

rect_bg = mpatches.FancyBboxPatch((0, 0.5), 10, 9.5, boxstyle="round,pad=0.3",
                                   facecolor=C['bg_box'], edgecolor='#ddd', linewidth=1)
ax2.add_patch(rect_bg)
ax2.set_zorder(-1)

plt.tight_layout()
plt.savefig(f'{OUT}/10_lookahead_results.png', bbox_inches='tight')
plt.close()
print("3. Lookahead results done")

# ============================================================
# 4. Curriculum Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
labels = ['PPO alone', '+ Lookahead']
v1 = [78.0, 91.9]
curriculum = [73.7, 91.5]
x = np.arange(len(labels))
w = 0.28

bars1 = ax.bar(x - w/2, v1, w, label='Original V1 (trained vs Random)', color=C['primary'], edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + w/2, curriculum, w, label='Curriculum (trained vs Greedy)', color=C['secondary'], edgecolor='white', linewidth=1.5)

ax.set_ylabel('Win Rate vs Greedy (%)', fontsize=12, color=C['dark'])
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)
ax.set_ylim(65, 102)
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax.set_title('Curriculum Training: Does the Base Model Matter?', fontsize=14, fontweight='bold', color=C['dark'])

for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height()}%', ha='center', va='bottom', fontweight='bold', fontsize=12, color=C['dark'])

# Annotations
ax.annotate('≈ Same!', xy=(1.14, 91.7), fontsize=13, fontweight='bold', color=C['success'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor=C['success']))
ax.annotate('-4.3pp', xy=(0.14, 75.5), fontsize=11, fontweight='bold', color=C['danger'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FDEDEC', edgecolor=C['danger']))

# Bottom callout
fig.text(0.5, 0.02, '💡  PPO is just a candidate filter. The evaluation function does the real work.',
         ha='center', fontsize=10, fontweight='bold', color=C['primary'],
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#DBEAFE', edgecolor=C['primary']))

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(f'{OUT}/09_curriculum_comparison.png', bbox_inches='tight')
plt.close()
print("4. Curriculum comparison done")

# ============================================================
# 5. Component Ablation
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
configs = ['Full\n(baseline)', 'No event\nreward', 'No PPO\nvalue', 'No PPO\nprob', 'Event\nonly', 'K=5', 'K=30']
rates = [91.9, 76.1, 92.8, 91.0, 91.4, 87.7, 93.0]
deltas = [0, -15.8, +0.9, -0.9, -0.5, -4.2, +1.1]

bar_colors = []
for d in deltas:
    if d == 0: bar_colors.append(C['primary'])
    elif d < -5: bar_colors.append(C['danger'])
    elif d < 0: bar_colors.append(C['accent'])
    else: bar_colors.append(C['success'])

bars = ax.bar(configs, rates, color=bar_colors, width=0.55, edgecolor='white', linewidth=1.5)
ax.set_ylabel('Win Rate vs Greedy (%)', fontsize=12, color=C['dark'])
ax.set_title('Component Ablation: What Matters Most?', fontsize=14, fontweight='bold', color=C['dark'])
ax.set_ylim(70, 100)
ax.axhline(y=91.9, color=C['primary'], linestyle='--', alpha=0.3)

for bar, val, d in zip(bars, rates, deltas):
    label = f'{val}%'
    if d != 0:
        label += f'\n({d:+.1f}pp)'
    color = C['danger'] if d < -5 else C['dark']
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, label,
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

fig.text(0.5, 0.02, '🚨  Removing event reward = -16pp  |  PPO value is useless (+0.9pp without it)  |  Event alone ≈ full model',
         ha='center', fontsize=9.5, fontweight='bold', color=C['danger'],
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDEDEC', edgecolor=C['danger']))

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(f'{OUT}/04_component_ablation.png', bbox_inches='tight')
plt.close()
print("5. Component ablation done")

# ============================================================
# 6. Event Leave-One-Out
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
events = ['buy_card', 'reach_15', 'buy_reserved', 'scarcity_take', 'engine_spike',
          'block_reserve', 'score_up', 'take_gems', 'reserve_card']
loo_deltas = [-5.5, -2.5, -1.5, -1.3, -1.1, -0.7, +0.5, +0.7, +1.1]
loo_rates = [86.4, 89.4, 90.4, 90.6, 90.8, 91.2, 92.4, 92.6, 93.0]

bar_colors = []
for d in loo_deltas:
    if d < -3: bar_colors.append(C['danger'])
    elif d < 0: bar_colors.append(C['accent'])
    else: bar_colors.append(C['success'])

bars = ax.barh(events, loo_deltas, color=bar_colors, height=0.6, edgecolor='white', linewidth=1.5)
ax.set_xlabel('Impact on Win Rate (pp)', fontsize=12, color=C['dark'])
ax.set_title('Event Leave-One-Out: Which Signal Matters Most?', fontsize=14, fontweight='bold', color=C['dark'])
ax.axvline(x=0, color=C['dark'], linewidth=1)
ax.set_xlim(-7, 2.5)

for bar, d, r in zip(bars, loo_deltas, loo_rates):
    x = d - 0.2 if d < 0 else d + 0.1
    ha = 'right' if d < 0 else 'left'
    ax.text(x, bar.get_y() + bar.get_height()/2, f'{d:+.1f}pp ({r}%)',
            ha=ha, va='center', fontsize=10, fontweight='bold', color=C['dark'])

fig.text(0.5, 0.02, '🏆  buy_card is the MVP (-5.5pp)  |  ❌ reserve_card is noise (+1.1pp when removed)',
         ha='center', fontsize=9.5, fontweight='bold', color=C['primary'],
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#DBEAFE', edgecolor=C['primary']))

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(f'{OUT}/05_event_leave_one_out.png', bbox_inches='tight')
plt.close()
print("6. Event leave-one-out done")

# ============================================================
# 7. Final Summary Table
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

data = [
    ['PPO Sparse', '~3%', '—', 'Win/loss reward only'],
    ['PPO Score (V4a)', '75.8%', '[73.0%, 78.4%]', 'Score progress reward'],
    ['PPO Event (V5)', '78.0%', '[75.3%, 80.5%]', '9-dim event reward shaping'],
    ['AlphaZero (best)', '<10%', '—', 'MCTS 50 sims (failed)'],
    ['PPO+Lookahead', '91.9%', '[90.0%, 93.4%]', '1-step search + event eval'],
    ['PPO+Lookahead K=30', '93.0%', '[91.2%, 94.4%]', 'Best configuration'],
]
cols = ['Agent', 'vs Greedy', '95% CI', 'Method']

table = ax.table(cellText=data, colLabels=cols, loc='center', cellLoc='center',
                 colWidths=[0.25, 0.13, 0.20, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.0)

for j in range(len(cols)):
    table[0, j].set_facecolor(C['primary'])
    table[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, len(data)+1):
    for j in range(len(cols)):
        table[i, j].set_edgecolor('#ddd')
        if i % 2 == 0:
            table[i, j].set_facecolor(C['light'])

# Highlight best rows
for j in range(len(cols)):
    table[5, j].set_facecolor('#DBEAFE')
    table[6, j].set_facecolor('#E8F5E9')

# AlphaZero red
table[4, 1].set_text_props(color=C['danger'], fontweight='bold')

fig.suptitle('Final Agent Comparison (n=1000 games each)', fontsize=14,
             fontweight='bold', color=C['dark'], y=0.95)
fig.text(0.5, 0.06, '35+ experiments  |  30,000+ evaluation games  |  111 human-vs-AI battles',
         ha='center', fontsize=10, color=C['primary'], fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#DBEAFE', edgecolor=C['primary']))

plt.tight_layout(rect=[0, 0.1, 1, 0.92])
plt.savefig(f'{OUT}/08_final_summary.png', bbox_inches='tight')
plt.close()
print("7. Final summary done")

print(f"\nAll v2 figures saved to {OUT}/")
