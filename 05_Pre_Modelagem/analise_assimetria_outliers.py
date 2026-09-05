"""
Análise de Assimetria e Outliers - Dataset PRF
=================================================
Variáveis alvo: mortos, feridos_graves, feridos_leves, ilesos, ignorados,
                feridos, veiculos (contagens por acidente)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

# ── Configuração visual ──────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.15)
PALETTE = "mako"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "graficos_assimetria_outliers")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Carregamento ─────────────────────────────────────────────────────────────
# Procura df_prf_processed.csv na pasta local, na raiz ou em 02_Transformacao
possible_paths = [
    os.path.join(BASE_DIR, "df_prf_processed.csv"),
    os.path.join(BASE_DIR, "..", "df_prf_processed.csv"),
    os.path.join(BASE_DIR, "..", "02_Transformacao", "df_prf_processed.csv")
]
CSV_PATH = next((p for p in possible_paths if os.path.exists(p)), possible_paths[1])
print(f"Carregando dataset de: {CSV_PATH}...")
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Shape: {df.shape}")
print("Colunas:", list(df.columns))

# ── Identificar variáveis alvo (numéricas inteiras de contagem) ───────────────
CANDIDATE_TARGETS = [
    "mortos", "feridos_graves", "feridos_leves", "ilesos",
    "ignorados", "feridos", "veiculos"
]
TARGET_COLS = [c for c in CANDIDATE_TARGETS if c in df.columns]

# Se não encontrar nenhum dos candidatos, usa todas as colunas numéricas inteiras
if not TARGET_COLS:
    TARGET_COLS = [
        c for c in df.select_dtypes(include=["int64", "int32", "float64"]).columns
    ]

print(f"\nVariáveis alvo identificadas: {TARGET_COLS}")

# ── 1. Estatísticas de Assimetria ────────────────────────────────────────────
print("\n" + "="*60)
print("ANÁLISE DE ASSIMETRIA (Skewness)")
print("="*60)
stats_rows = []
for col in TARGET_COLS:
    s = df[col].dropna()
    sk = s.skew()
    ku = s.kurtosis()
    mean = s.mean()
    median = s.median()
    std = s.std()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    n_outliers_iqr = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
    z_scores = np.abs(stats.zscore(s, nan_policy="omit"))
    n_outliers_z = (z_scores > 3).sum()
    stats_rows.append({
        "Variável":      col,
        "Média":         format(mean, ".3f"),
        "Mediana":       format(median, ".3f"),
        "Desvio Padrão": format(std, ".3f"),
        "Skewness":      round(sk, 4),
        "Kurtosis":      round(ku, 4),
        "Tipo de Cauda": (
            "Simétrica (|sk|<0.5)"     if abs(sk) < 0.5 else
            "Assimetria Moderada"       if abs(sk) < 1.0 else
            "Cauda Direita Forte (+)"   if sk > 0 else
            "Cauda Esquerda Forte (-)"
        ),
        "Outliers IQR":   int(n_outliers_iqr),
        "Outliers Z>3":   int(n_outliers_z),
        "% Outlier IQR": round(100 * n_outliers_iqr / len(s), 2),
    })
    print(f"\n{col}")
    print(f"  Média={mean:.3f}  Mediana={median:.3f}  Std={std:.3f}")
    print(f"  Skewness={sk:.4f}  Kurtosis={ku:.4f}")
    print(f"  Outliers IQR: {int(n_outliers_iqr)} ({100*n_outliers_iqr/len(s):.2f}%)")
    print(f"  Outliers Z>3: {int(n_outliers_z)}")

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(OUTPUT_DIR, "estatisticas_assimetria.csv"), index=False)
print(f"\nEstatísticas salvas em: {OUTPUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS DE DISTRIBUIÇÃO / CAUDA
# ─────────────────────────────────────────────────────────────────────────────
n_cols = 2
n_rows = (len(TARGET_COLS) + 1) // n_cols

# ── Fig 1: KDE + Histograma (cauda de distribuição) ──────────────────────────
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()
fig.suptitle(
    "Distribuição das Variáveis Alvo — Análise de Cauda (Skewness)",
    fontsize=16, fontweight="bold", y=1.01
)

for i, col in enumerate(TARGET_COLS):
    ax = axes[i]
    s = df[col].dropna()
    color = sns.color_palette(PALETTE, len(TARGET_COLS))[i]

    sns.histplot(s, kde=True, bins=40, ax=ax, color=color, alpha=0.7,
                 stat="density", line_kws={"lw": 2.2})

    # Linhas de média e mediana
    ax.axvline(s.mean(),   color="red",    linestyle="--", lw=1.8, label=f"Média={s.mean():.2f}")
    ax.axvline(s.median(), color="orange", linestyle="-.",  lw=1.8, label=f"Mediana={s.median():.2f}")

    sk = s.skew()
    direction = "→ Cauda Direita" if sk > 0 else ("← Cauda Esquerda" if sk < 0 else "Simétrica")
    ax.set_title(f"{col}  [sk={sk:.3f}  {direction}]", fontsize=11)
    ax.set_xlabel(col)
    ax.set_ylabel("Densidade")
    ax.legend(fontsize=9)

# Oculta eixos vazios
for j in range(len(TARGET_COLS), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
path_dist = os.path.join(OUTPUT_DIR, "01_distribuicao_cauda.png")
plt.savefig(path_dist, dpi=150, bbox_inches="tight")
plt.close()
print(f"Salvo: {path_dist}")

# ── Fig 2: Log-scale distribution (evidencia a cauda pesada) ─────────────────
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()
fig.suptitle(
    "Distribuição em Escala Log — Evidência de Cauda Pesada",
    fontsize=16, fontweight="bold", y=1.01
)
for i, col in enumerate(TARGET_COLS):
    ax = axes[i]
    s = df[col].dropna()
    color = sns.color_palette("rocket", len(TARGET_COLS))[i]
    # Deslocamos +1 para evitar log(0)
    sns.histplot(s + 1, bins=40, ax=ax, color=color, alpha=0.75,
                 log_scale=True, stat="density")
    sk = s.skew()
    ax.set_title(f"{col}  (sk={sk:.3f})", fontsize=11)
    ax.set_xlabel(f"{col} + 1 (log-scale)")
    ax.set_ylabel("Densidade")
for j in range(len(TARGET_COLS), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
path_log = os.path.join(OUTPUT_DIR, "02_distribuicao_log_scale.png")
plt.savefig(path_log, dpi=150, bbox_inches="tight")
plt.close()
print(f"Salvo: {path_log}")

# ── Fig 3: QQ-Plot (desvio da normalidade / cauda) ───────────────────────────
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()
fig.suptitle(
    "Q-Q Plot — Desvio da Normalidade (Análise de Cauda)",
    fontsize=16, fontweight="bold", y=1.01
)
for i, col in enumerate(TARGET_COLS):
    ax = axes[i]
    s = df[col].dropna().values
    (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm")
    ax.scatter(osm, osr, s=8, alpha=0.4,
               color=sns.color_palette("viridis", len(TARGET_COLS))[i])
    ax.plot(osm, slope * np.array(osm) + intercept, color="red", lw=1.8, label=f"r²={r**2:.3f}")
    sk = pd.Series(s).skew()
    ax.set_title(f"{col}  (sk={sk:.3f})", fontsize=11)
    ax.set_xlabel("Quantis Teóricos")
    ax.set_ylabel("Quantis Amostrados")
    ax.legend(fontsize=9)
for j in range(len(TARGET_COLS), len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
path_qq = os.path.join(OUTPUT_DIR, "03_qq_plot.png")
plt.savefig(path_qq, dpi=150, bbox_inches="tight")
plt.close()
print(f"Salvo: {path_qq}")

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS DE OUTLIERS
# ─────────────────────────────────────────────────────────────────────────────

# ── Fig 4: Boxplot comparativo (visão geral dos outliers) ────────────────────
# Normaliza para z-score para comparar variáveis em escalas diferentes
df_norm = df[TARGET_COLS].copy()
for col in TARGET_COLS:
    df_norm[col] = stats.zscore(df[col].dropna().reindex(df.index), nan_policy="omit")

fig, ax = plt.subplots(figsize=(13, 6))
sns.boxplot(data=df_norm[TARGET_COLS], palette=PALETTE, width=0.55,
            flierprops=dict(marker="o", markersize=2, alpha=0.4), ax=ax,
            order=TARGET_COLS)
ax.axhline(3,  color="red",    linestyle="--", lw=1.5, label="Z=±3 (threshold)")
ax.axhline(-3, color="red",    linestyle="--", lw=1.5)
ax.axhline(1.5, color="orange", linestyle=":",  lw=1.2, label="Z=±1.5")
ax.axhline(-1.5, color="orange", linestyle=":", lw=1.2)
ax.set_title("Boxplot — Outliers por Variável (Z-score normalizado)", fontsize=14, fontweight="bold")
ax.set_xlabel("Variável")
ax.set_ylabel("Z-score")
ax.legend()
plt.tight_layout()
path_box = os.path.join(OUTPUT_DIR, "04_boxplot_outliers_zscore.png")
plt.savefig(path_box, dpi=150, bbox_inches="tight")
plt.close()
print(f"Salvo: {path_box}")

# ── Fig 5: Violin + Strip plot (distribuição completa + outliers pontuais) ───
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows))
axes = axes.flatten()
fig.suptitle(
    "Violin Plot — Distribuição Completa e Outliers (IQR × 1.5)",
    fontsize=15, fontweight="bold", y=1.01
)
for i, col in enumerate(TARGET_COLS):
    ax = axes[i]
    s = df[[col]].dropna()
    q1, q3 = s[col].quantile(0.25), s[col].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = s[col][(s[col] < low) | (s[col] > high)]

    color = sns.color_palette(PALETTE, len(TARGET_COLS))[i]
    sns.violinplot(y=s[col], ax=ax, color=color, alpha=0.75, inner="quartile",
                   linewidth=1.5)
    # Pontos dos outliers
    if len(outliers) > 0:
        # Limita a 500 pts para performance
        sample_out = outliers.sample(min(500, len(outliers)), random_state=42)
        ax.scatter([0] * len(sample_out), sample_out.values,
                   color="crimson", s=10, alpha=0.5, zorder=5, label=f"{len(outliers)} outliers")
    ax.axhline(high, color="red",  linestyle="--", lw=1.4, label=f"Q3+1.5IQR={high:.1f}")
    ax.axhline(low,  color="blue", linestyle="--", lw=1.4, label=f"Q1-1.5IQR={low:.1f}")
    ax.set_title(col, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel(col)
    ax.legend(fontsize=8)
for j in range(len(TARGET_COLS), len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
path_violin = os.path.join(OUTPUT_DIR, "05_violin_outliers.png")
plt.savefig(path_violin, dpi=150, bbox_inches="tight")
plt.close()
print(f"Salvo: {path_violin}")

# ── Fig 6: Contagem de outliers por variável (barra) ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Quantidade e Proporção de Outliers por Variável", fontsize=14, fontweight="bold")

x = stats_df["Variável"]
y_abs = stats_df["Outliers IQR"]
y_pct = stats_df["% Outlier IQR"]
colors = sns.color_palette(PALETTE, len(stats_df))

ax1 = axes[0]
bars1 = ax1.bar(x, y_abs, color=colors, edgecolor="white", linewidth=0.5)
ax1.bar_label(bars1, fmt="%d", fontsize=9)
ax1.set_title("Outliers (contagem absoluta — IQR)", fontsize=11)
ax1.set_xlabel("Variável")
ax1.set_ylabel("Quantidade")
ax1.tick_params(axis="x", rotation=30)

ax2 = axes[1]
bars2 = ax2.bar(x, y_pct, color=colors, edgecolor="white", linewidth=0.5)
ax2.bar_label(bars2, fmt="%.2f%%", fontsize=9)
ax2.set_title("Outliers (proporção — IQR)", fontsize=11)
ax2.set_xlabel("Variável")
ax2.set_ylabel("% do total de registros")
ax2.tick_params(axis="x", rotation=30)

plt.tight_layout()
path_bar = os.path.join(OUTPUT_DIR, "06_outliers_barra.png")
plt.savefig(path_bar, dpi=150, bbox_inches="tight")
plt.close()
print(f"Salvo: {path_bar}")

# ── Fig 7: Scatter Plots – Outliers IQR coloridos ────────────────────────────
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows))
axes = axes.flatten()
fig.suptitle(
    "Scatter — Outliers Identificados pelo Critério IQR × 1.5",
    fontsize=15, fontweight="bold", y=1.01
)
for i, col in enumerate(TARGET_COLS):
    ax = axes[i]
    s = df[col].dropna().reset_index(drop=True)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    is_out = (s < low) | (s > high)

    normal  = s[~is_out]
    outlier = s[is_out]

    ax.scatter(normal.index,  normal.values,  s=4,  alpha=0.3, color="steelblue",  label="Normal")
    ax.scatter(outlier.index, outlier.values, s=12, alpha=0.7, color="crimson",    label=f"Outlier ({len(outlier)})")
    ax.axhline(high, color="red",  linestyle="--", lw=1.3)
    ax.axhline(low,  color="blue", linestyle="--", lw=1.3)
    ax.set_title(col, fontsize=11)
    ax.set_xlabel("Índice")
    ax.set_ylabel(col)
    ax.legend(fontsize=8)
for j in range(len(TARGET_COLS), len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
path_scatter = os.path.join(OUTPUT_DIR, "07_scatter_outliers_IQR.png")
plt.savefig(path_scatter, dpi=150, bbox_inches="tight")
plt.close()
print(f"Salvo: {path_scatter}")

print("\n" + "="*60)
print("Análise concluída. Todos os gráficos salvos em:")
print(OUTPUT_DIR)
print("="*60)
print(stats_df.to_string(index=False))
