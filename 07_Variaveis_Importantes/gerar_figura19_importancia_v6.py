# -*- coding: utf-8 -*-
"""
07_Variaveis_Importantes/gerar_figura19_importancia_v6.py
=========================================================
Gera a Figura 19 oficial do TCC para a Arquitetura V6:
'Figura 19 – Top 20 variáveis por importância preditiva no Modelo V6'

Plota o gráfico de barras horizontais com as 20 variáveis mais importantes
(Mean Decrease Impurity / MDI sob critério de Poisson) no Random Forest V6.
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

out_dirs = [
    os.path.join(BASE_DIR, 'documento_tcc', 'figuras'),
    os.path.join(BASE_DIR, '07_Variaveis_Importantes')
]

for d in out_dirs:
    os.makedirs(d, exist_ok=True)

print("Carregando modelo e features do Modelo V6 Oficial...")
model = joblib.load(model_path)
features = joblib.load(feat_path)

importances = model.feature_importances_
df_imp = pd.Series(importances * 100, index=features).sort_values(ascending=False)

top20 = df_imp.head(20).iloc[::-1] # inverte para a maior ficar no topo

# Configurações visuais
plt.rcParams.update({
    'font.size': 10.5,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'sans-serif'
})

fig, ax = plt.subplots(figsize=(10, 8.2), dpi=200)

# Cores em gradiente com destaque suave
colors = plt.cm.viridis(np.linspace(0.20, 0.88, len(top20)))

bars = ax.barh(top20.index, top20.values, color=colors, edgecolor='#1e293b', linewidth=0.8, alpha=0.88)

# Rótulo de valores nas barras
for b in bars:
    w = b.get_width()
    txt = f"{w:.2f}%".replace('.', ',') if w < 1 else f"{w:.1f}%".replace('.', ',')
    ax.text(
        w + 0.35, b.get_y() + b.get_height()/2,
        txt,
        va='center', ha='left',
        fontsize=9.5, fontweight='bold', color='#0f172a'
    )

ax.set_xlabel('Importância Relativa (%) - Redução de Impureza de Poisson (MDI)', fontweight='bold', labelpad=10)
ax.set_title('Top 20 Variáveis por Importância Preditiva: Modelo V6 Oficial', fontweight='bold', pad=12)
ax.set_xlim(0, max(top20.values) * 1.15)
ax.grid(axis='x', linestyle='--', alpha=0.5)

# Remover bordas desnecessárias
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

for d in out_dirs:
    target_file = os.path.join(d, 'figura19_importancia_variaveis_v6.png')
    plt.savefig(target_file, dpi=200, bbox_inches='tight')
    print(f"Figura 19 salva em: {target_file}")

# Também salvar como feature_importance_v6.png para manter compatibilidade
comp_file = os.path.join(BASE_DIR, '07_Variaveis_Importantes', 'feature_importance_v6.png')
plt.savefig(comp_file, dpi=200, bbox_inches='tight')
print(f"Cópia salva em: {comp_file}")

plt.close()
print("Geração da Figura 19 concluída com sucesso!")
