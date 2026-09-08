# -*- coding: utf-8 -*-
"""
05_Pre_Modelagem/gerar_figura5_distribuicao_v6.py
==================================================
Gera a Figura 5 oficial do TCC para a Arquitetura V6:
'Figura 5 - Distribuição e Cauda da Variável Alvo'

Painel Analítico Triplo:
(A) Histograma + Curva de Densidade (KDE):
    Destaca a massa de zeros (70,61%), Média = 0,997 e Mediana = 0,000.
(B) Função de Cauda Complementar (CCDF em escala log):
    Evidencia a cauda longa com P95 = 5, P99 = 15, P99.9 = 40 e Máx = 77.
(C) Q-Q Plot contra a Distribuição Normal:
    Comprova graficamente a falha de premissa da regressão linear.
"""

import os
import sys
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'df_model_ready_v6.csv')
out_dirs = [
    os.path.join(BASE_DIR, 'graficos_v6'),
    os.path.join(BASE_DIR, '05_Pre_Modelagem')
]

for d in out_dirs:
    os.makedirs(d, exist_ok=True)

print(f'Carregando dados de: {data_path}...')
df = pd.read_csv(data_path, low_memory=False)
y = df['quantidade_acidentes'].values

# Métricas fundamentais
n_total = len(y)
mean_val = np.mean(y)
med_val = np.median(y)
sk_val = stats.skew(y)
ku_val = stats.kurtosis(y)
p95 = np.percentile(y, 95)
p99 = np.percentile(y, 99)
p999 = np.percentile(y, 99.9)
max_val = np.max(y)
zero_pct = (y == 0).sum() / n_total * 100

print(f'N = {n_total:,} observações')
print(f'Média = {mean_val:.4f} | Mediana = {med_val:.4f}')
print(f'Zeros Reais = {zero_pct:.2f}% ({np.sum(y == 0):,} trechos/mês)')
print(f'Skewness = {sk_val:.4f} | Kurtosis = {ku_val:.4f}')
print(f'P95 = {p95:.1f} | P99 = {p99:.1f} | P99.9 = {p999:.1f} | Máx = {max_val}')

# Configuração estética
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 200,
    'font.family': 'sans-serif'
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(
    'Figura 5 - Distribuição e Cauda da Variável Alvo',
    fontsize=15, fontweight='bold', y=0.98
)

# -----------------------------------------------------------------------------
# GRÁFICO 1: Histograma + Curva de Densidade
# -----------------------------------------------------------------------------
ax1 = axes[0]
bins_range = np.arange(-0.5, 25.5, 1.0)
counts, bin_edges = np.histogram(y, bins=bins_range, density=True)

# Barras do histograma
ax1.bar(
    bin_edges[:-1] + 0.5, counts, width=0.85,
    color='#0284c7', edgecolor='black', alpha=0.75, label='Densidade Real'
)

# Curva suave KDE sobre amostra (para agilidade computacional)
np.random.seed(42)
sample_y = np.random.choice(y, size=min(40000, n_total), replace=False)
kde = stats.gaussian_kde(sample_y, bw_method=0.25)
x_kde = np.linspace(0, 25, 300)
ax1.plot(x_kde, kde(x_kde), color='#0369a1', lw=2.5, label='Curva de Densidade (KDE)')

# Linhas verticais de Média e Mediana
ax1.axvline(
    mean_val, color='#dc2626', linestyle='--', lw=2.2,
    label=f'Média = {mean_val:.2f}'
)
ax1.axvline(
    med_val, color='#16a34a', linestyle='-.', lw=2.2,
    label=f'Mediana = {med_val:.1f} (70.6% Zeros)'
)

ax1.set_title('(A) Histograma e Curva de Densidade (KDE)', fontweight='bold')
ax1.set_xlabel('Quantidade de Acidentes (Trecho de 10 km / Mês)')
ax1.set_ylabel('Densidade')
ax1.set_xlim(-0.8, 20)
ax1.legend(loc='upper right', fontsize=9.5)
ax1.grid(True, linestyle='--', alpha=0.5)

# -----------------------------------------------------------------------------
# GRÁFICO 2: Função de Cauda Complementar (CCDF em Escala Log)
# -----------------------------------------------------------------------------
ax2 = axes[1]
unique_x = np.sort(np.unique(y))
# CCDF(x) = P(X >= x)
ccdf_y = np.array([(y >= val).sum() / n_total for val in unique_x])

ax2.step(unique_x, ccdf_y, where='post', color='#2563eb', lw=2.4, label='CCDF Amostral P(X ≥ x)')

# Linhas de percentis críticos
ax2.axvline(p95, color='#f59e0b', linestyle=':', lw=1.8, label=f'P95 = {int(p95)} acidentes (5% da malha)')
ax2.axvline(p99, color='#8b5cf6', linestyle=':', lw=1.8, label=f'P99 = {int(p99)} acidentes (1% da malha)')
ax2.axvline(p999, color='#dc2626', linestyle=':', lw=1.8, label=f'P99.9 = {int(p999)} acidentes (Top 0.1%)')

# Destaque do ponto máximo
ax2.scatter([max_val], [1.0 / n_total], color='crimson', s=55, zorder=5)
ax2.annotate(
    f'Máximo: {max_val} ac/mês',
    xy=(max_val, 1.0 / n_total),
    xytext=(max_val - 25, 5e-5),
    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
    fontsize=9, fontweight='bold'
)

ax2.set_yscale('log')
ax2.set_title('(B) Função de Cauda (CCDF em Escala Log)', fontweight='bold')
ax2.set_xlabel('Quantidade de Acidentes (x)')
ax2.set_ylabel('Probabilidade P(X ≥ x) — Log')
ax2.set_xlim(-1, 82)
ax2.set_ylim(1e-6, 1.2)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

# -----------------------------------------------------------------------------
# GRÁFICO 3: Q-Q Plot vs. Normal Gaussiana
# -----------------------------------------------------------------------------
ax3 = axes[2]
# Amostragem para plot ágil e nítido
sample_qq = np.random.choice(y, size=min(15000, n_total), replace=False)
(osm, osr), (slope, intercept, r) = stats.probplot(sample_qq, dist='norm')

ax3.scatter(osm, osr, s=6, color='#0284c7', alpha=0.35, label='Quantis Amostrais Reais')
x_ref = np.linspace(np.min(osm), np.max(osm), 200)
ax3.plot(x_ref, slope * x_ref + intercept, color='#dc2626', lw=2.2, label=f'Linha Normal Teórica (r²={r**2:.3f})')

annot_text = f"Skewness = {sk_val:.3f}\nKurtosis = {ku_val:.2f}\nAssimetria Severa (+)"
ax3.text(
    0.05, 0.78,
    annot_text,
    transform=ax3.transAxes,
    fontsize=9.5, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef3c7', edgecolor='#d97706', alpha=0.9)
)

ax3.set_title('(C) Q-Q Plot vs. Distribuição Normal', fontweight='bold')
ax3.set_xlabel('Quantis Teóricos da Gaussiana')
ax3.set_ylabel('Quantis Observados da Amostra')
ax3.set_ylim(-2, max_val + 5)
ax3.legend(loc='upper left', fontsize=9.5)
ax3.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])

for d in out_dirs:
    p = os.path.join(d, 'figura5_distribuicao_cauda_v6.png')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    print(f'Gráfico salvo em: {p}')

plt.close()
print('Geração da Figura 5 concluída com sucesso!')
