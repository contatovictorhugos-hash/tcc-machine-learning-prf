# -*- coding: utf-8 -*-
"""
06_Modelagem/gerar_figura18_residuos_v6.py
===========================================
Gera a Figura 18 oficial do TCC para a Arquitetura V6:
'Figura 18 – Análise de resíduos do Modelo V6 Oficial: histograma de erros e dispersão predito versus real'

Painel Analítico Duplo:
(A) Histograma de Resíduos (Y - Y_pred):
    Mostra a altíssima concentração em torno do zero (93,59% em [-2, +2]),
    média residual de +0,002 e a cauda positiva em eventos de pico.
(B) Gráfico de Dispersão Predito vs. Real:
    Mostra a aderência à linha diagonal (y=x), com dispersão reduzida na faixa
    rotineira e o efeito da média da floresta na cauda rara (> 40 acidentes).
"""

import os
import sys
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, d2_tweedie_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, '06_Modelagem', 'random_forest_v6_tcc.pkl')
feat_path = os.path.join(BASE_DIR, '06_Modelagem', 'v6_feature_names.pkl')
data_path = os.path.join(BASE_DIR, '05_Pre_Modelagem', 'df_model_ready_v6.csv')

out_dirs = [
    os.path.join(BASE_DIR, 'documento_tcc', 'figuras'),
    os.path.join(BASE_DIR, '06_Modelagem', 'graficos_modelo_v6')
]

for d in out_dirs:
    os.makedirs(d, exist_ok=True)

print("Carregando modelo e dados V6...")
model = joblib.load(model_path)
feature_names = joblib.load(feat_path)
df = pd.read_csv(data_path, low_memory=False)

# Conjunto de Teste: anos 2024 e 2025 (avaliação fora do tempo)
df_test = df[df['ano'].isin([2024, 2025])].copy()
y_test = df_test['quantidade_acidentes'].values

cat_cols = ['estacao_ano', 'uf']
cols_to_drop = ['quantidade_acidentes', 'ano', 'mes', 'br', 'trecho_10km',
                'total_mortos', 'total_feridos_graves', 'total_veiculos', 'is_zero_real']

df_test_enc = pd.get_dummies(df_test.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False)
df_test_enc = df_test_enc.reindex(columns=feature_names, fill_value=0).astype(float)
X_test = df_test_enc.values

print(f"Executando predição para {len(y_test):,} observações de teste...")
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# Estatísticas
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
d2 = d2_tweedie_score(y_test, y_pred, power=1)
n_obs = len(y_test)
res_mean = np.mean(residuals)
res_median = np.median(residuals)
pct_within_1 = np.mean(np.abs(residuals) <= 1) * 100
pct_within_2 = np.mean(np.abs(residuals) <= 2) * 100

print(f"N Teste: {n_obs:,}")
print(f"MAE: {mae:.4f} | R2: {r2:.4f} | D2 Poisson: {d2:.4f}")
print(f"Média dos resíduos: {res_mean:.4f} | Mediana: {res_median:.4f}")
print(f"Erros em [-1, +1]: {pct_within_1:.2f}% | Erros em [-2, +2]: {pct_within_2:.2f}%")

# Configuração visual profissional
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 13,
    'font.family': 'sans-serif'
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=200)

# =============================================================================
# (A) HISTOGRAMA DE RESÍDUOS
# =============================================================================
ax1 = axes[0]
bins = np.linspace(-6, 8, 57) # intervalos finos de 0.25
n, b, patches = ax1.hist(residuals, bins=bins, color='#0284c7', edgecolor='#0369a1', alpha=0.75, label='Frequência de Trechos/Mês')

# Destaque para a linha de resíduo zero
ax1.axvline(0, color='#dc2626', linestyle='--', linewidth=2, label='Erro Residual Zero ($Y = \\hat{Y}$)')

# Faixa de maior densidade [-2, +2]
ax1.axvspan(-2, 2, color='#e0f2fe', alpha=0.4, label='Faixa $\\pm 2$ acidentes (93,6%)')

ax1.set_title('(A) Distribuição dos Erros Residuais ($Y - \\hat{Y}$)', fontweight='bold', pad=10)
ax1.set_xlabel('Erro Residual (Acidentes Reais - Acidentes Previstos)')
ax1.set_ylabel('Frequência de Observações')
ax1.set_xlim(-6.5, 8.5)
ax1.legend(loc='upper right', fontsize=9.2)
ax1.grid(True, linestyle='--', alpha=0.5)

# =============================================================================
# (B) DISPERSÃO PREDITO VS. REAL
# =============================================================================
ax2 = axes[1]

# Scatter plot com transparência controlada
ax2.scatter(
    y_test, y_pred,
    alpha=0.20, s=9, color='#1e40af', edgecolors='none',
    label='Trechos Rodoviários Mensais (N = 135.720)'
)

# Linha de predição perfeita y = x
max_axis = 80
ax2.plot([0, max_axis], [0, max_axis], color='#dc2626', linestyle='--', linewidth=2, label='Predição Perfeita ($Y = \\hat{Y}$)')

# Linha indicativa da faixa onde começa a suavização extrema
ax2.axvline(40, color='#d97706', linestyle=':', linewidth=1.5, label='Início de Cauda Rara ($Y > 40$)')

ax2.set_title('(B) Dispersão Predito versus Real ($\\hat{Y}$ vs. $Y$)', fontweight='bold', pad=10)
ax2.set_xlabel('Quantidade Real de Acidentes ($Y$)')
ax2.set_ylabel('Quantidade Prevista pelo Modelo ($\\hat{Y}$)')
ax2.set_xlim(-1, 82)
ax2.set_ylim(-1, 50)
ax2.legend(loc='lower right', fontsize=9.2)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

for d in out_dirs:
    target_file = os.path.join(d, 'figura18_analise_residuos_v6.png')
    plt.savefig(target_file, dpi=200, bbox_inches='tight')
    print(f"Figura 18 salva com sucesso em: {target_file}")

plt.close()
print("Processo finalizado com sucesso!")
