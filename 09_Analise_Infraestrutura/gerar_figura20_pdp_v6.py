# -*- coding: utf-8 -*-
"""
09_Analise_Infraestrutura/gerar_figura20_pdp_v6.py
===================================================
Gera a Figura 20 oficial do TCC para a Arquitetura V6:
'Figura 20: Curva de Dependência Parcial (PDP) do ICP: Modelo V6 Oficial'

Mapeia a sensibilidade marginal do Índice de Condição do Pavimento (ICP)
de 0 a 100 na predição de acidentes/mês nos trechos monitorados pelo DNIT.
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, '06_Modelagem', 'random_forest_v6_tcc.pkl')
feat_path = os.path.join(BASE_DIR, '06_Modelagem', 'v6_feature_names.pkl')
data_path = os.path.join(BASE_DIR, '05_Pre_Modelagem', 'df_model_ready_v6.csv')

out_dirs = [
    os.path.join(BASE_DIR, 'documento_tcc', 'figuras'),
    os.path.join(BASE_DIR, '09_Analise_Infraestrutura')
]

for d in out_dirs:
    os.makedirs(d, exist_ok=True)

print("Carregando modelo e dados para PDP V6...")
model = joblib.load(model_path)
features = joblib.load(feat_path)
df = pd.read_csv(data_path, low_memory=False)

df_dnit = df[df['has_dnit_data'] == 1].copy()
sample_size = min(15000, len(df_dnit))
df_sample = df_dnit.sample(n=sample_size, random_state=42).copy()

cols_to_drop = [
    'quantidade_acidentes', 'ano', 'mes', 'br', 'trecho_10km',
    'total_mortos', 'total_feridos_graves', 'total_veiculos', 'is_zero_real'
]
cat_cols = ['estacao_ano', 'uf']

df_sample_enc = pd.get_dummies(df_sample.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False)
X_base = df_sample_enc.reindex(columns=features, fill_value=0)

icp_grid = np.arange(0, 105, 10)
pdp_vals = []

print("Calculando predições marginais para o grid de ICP...")
for val in icp_grid:
    X_sim = X_base.copy()
    X_sim['icp'] = float(val)
    preds = model.predict(X_sim.astype(float).values)
    pdp_vals.append(preds.mean())

# Configuração visual
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'sans-serif'
})

fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)

ax.plot(icp_grid, pdp_vals, color='#1d4ed8', linewidth=2.5, marker='o', label='Modelo V6 Oficial (Poisson)')
ax.fill_between(icp_grid, pdp_vals, alpha=0.12, color='#1d4ed8')

# Limiares conceituais do DNIT
ax.axvline(x=20, color='#dc2626', linestyle='--', linewidth=1.5, label=r'Limiar Péssimo (ICP $\leq$ 20)')
ax.axvline(x=40, color='#f59e0b', linestyle='--', linewidth=1.5, label='Limiar Ruim (ICP = 40)')
ax.axvline(x=60, color='#ca8a04', linestyle=':', linewidth=1.5, label='Limiar Regular (ICP = 60)')
ax.axvline(x=80, color='#16a34a', linestyle='--', linewidth=1.5, label='Limiar Bom (ICP = 80)')

ax.set_title('Curva de Dependência Parcial (PDP) do ICP: Modelo V6 Oficial', fontweight='bold', pad=12)
ax.set_xlabel('Índice de Condição do Pavimento (ICP) [0 = Crítico / 100 = Excelente]', fontweight='bold')
ax.set_ylabel('Acidentes Previstos / Mês', fontweight='bold')
ax.set_ylim(0.95, 1.35)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.95)

plt.tight_layout()

for d in out_dirs:
    target = os.path.join(d, 'figura20_pdp_icp_v6.png')
    plt.savefig(target, bbox_inches='tight')
    print(f"Salvo em: {target}")

plt.close()
print("Figura 20 gerada com sucesso!")
