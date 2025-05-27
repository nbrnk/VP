# 必要なライブラリを再インポート（リセットされたため）
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- RAVE のデータ ---
rave_data = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 1]
])

# --- VQ-RAVE のデータ ---
vq_rave_data = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1]
])

from statsmodels.stats.proportion import proportion_confint

def bernoulli_confidence_interval(data, alpha=0.01):  # 99% CI → alpha=0.01
    flat = data.flatten()
    successes = flat.sum()
    n = flat.size
    ci_low, ci_upp = proportion_confint(successes, n, alpha=alpha, method='beta')
    mean = successes / n
    return mean, mean - ci_low, ci_upp - mean
# RAVE の 99% Clopper-Pearson信頼区間
rave_mean, rave_err_low, rave_err_up = bernoulli_confidence_interval(rave_data)

# VQ-RAVE の 99% Clopper-Pearson信頼区間
vq_rave_mean, vq_rave_err_low, vq_rave_err_up = bernoulli_confidence_interval(vq_rave_data)

# 図の作成
fig, ax = plt.subplots(figsize=(6, 2))

# 背景線
for x in [0, 0.5, 1.0]:
    ax.axvline(x, linestyle='dashed', color='gray', linewidth=1.5, alpha=1.0, zorder=0)

# プロット（左右で非対称な誤差バーを表示）
ax.errorbar(rave_mean, 0.5, xerr=[[rave_err_low], [rave_err_up]],
            fmt='o', capsize=5, label='RAVE', zorder=1)

ax.errorbar(vq_rave_mean, -0.5, xerr=[[vq_rave_err_low], [vq_rave_err_up]],
            fmt='o', capsize=5, label='VQ-RAVE', zorder=1)

# ラベルなどは従来どおり
ax.text(rave_mean, 0.6, "RAVE", fontsize=18, ha='center', zorder=2)
ax.text(vq_rave_mean, -0.4, "VQ-RAVE", fontsize=18, ha='center', zorder=2)

ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)
ax.set_yticks([])
ax.set_xticks([0, 0.5, 1.0])
ax.set_xlabel("Rhythmic Fidelity", fontsize=18, fontweight='bold')
ax.tick_params(axis='x', labelsize=16)

ax.text(0, 0, "No", fontsize=18, va='center', ha='right', zorder=2)
ax.text(1, 0, "Yes", fontsize=18, va='center', ha='left', zorder=2)

# 枠線調整
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)

# 保存と表示
plt.savefig("result1_bernoulli.pdf", bbox_inches="tight")
plt.show()
