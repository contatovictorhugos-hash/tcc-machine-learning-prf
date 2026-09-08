# -*- coding: utf-8 -*-
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out_dirs = [
    'documento_tcc/figuras',
    'graficos_v6'
]
for d in out_dirs:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'font.family': 'sans-serif'
})

# FIGURA 21
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)

trechos_ac = [
    'BR-469/PR km 100\n(ICP Atual: 20,0)',
    'BR-174/RR km 0\n(ICP Atual: 18,2)',
    'BR-70/GO km 450\n(ICP Atual: 18,8)'
]

pred_orig_ac = [3.80, 2.85, 0.54]
pred_ref_ac = [3.31, 2.42, 0.19]
reducoes_ac = [-0.49, -0.44, -0.35]
pcts_ac = [-12.9, -15.3, -64.3]

x = np.arange(len(trechos_ac))
width = 0.35

bars1 = ax.bar(x - width/2, pred_orig_ac, width, label='Condição Atual (ICP Crítico <= 20)', color='#ef4444', edgecolor='black', alpha=0.9)
bars2 = ax.bar(x + width/2, pred_ref_ac, width, label='Após Restauração (ICP = 70)', color='#10b981', edgecolor='black', alpha=0.9)

ax.set_ylabel('Acidentes Previstos / Mês', fontweight='bold')
ax.set_title('Top 3 Trechos com Maior Queda de Acidentes ao Restaurar Pavimento (ICP <= 20 para 70)', fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(trechos_ac, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.95)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
ax.set_ylim(0, 4.5)

for i, bar in enumerate(bars1):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.08, f'{h:.2f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

for i, bar in enumerate(bars2):
    h = bar.get_height()
    red = reducoes_ac[i]
    pct = pcts_ac[i]
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.08, f'{h:.2f}\n({red:+.2f} | {pct:.1f}%)', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#047857')

plt.tight_layout()

for d in out_dirs:
    p = os.path.join(d, 'figura21_top3_reducao_acidentes_v6.png')
    plt.savefig(p, bbox_inches='tight')
    print('Salvo:', p)

plt.close()

# FIGURA 22
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)

trechos_vit = [
    'BR-277/PR km 580\n(ICP Atual: 19,5)',
    'BR-469/PR km 100\n(ICP Atual: 20,0)',
    'BR-101/BA km 500\n(ICP Atual: 18,1)'
]

vit_orig = [2.67, 1.41, 3.05]
vit_ref = [1.97, 0.94, 2.64]
reducoes_vit = [-0.70, -0.47, -0.42]
pcts_vit = [-26.2, -33.1, -13.7]

x = np.arange(len(trechos_vit))
width = 0.35

bars1 = ax.bar(x - width/2, vit_orig, width, label='Vítimas Severas Atuais (ICP Crítico <= 20)', color='#b91c1c', edgecolor='black', alpha=0.9)
bars2 = ax.bar(x + width/2, vit_ref, width, label='Após Restauração (ICP = 70)', color='#059669', edgecolor='black', alpha=0.9)

ax.set_ylabel('Vítimas Severas Previstas / Mês (Mortos + Graves)', fontweight='bold')
ax.set_title('Top 3 Trechos com Maior Queda no Número de Vítimas Severas (Modelo V6)', fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(trechos_vit, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.95)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
ax.set_ylim(0, 3.7)

for i, bar in enumerate(bars1):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.08, f'{h:.2f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

for i, bar in enumerate(bars2):
    h = bar.get_height()
    red = reducoes_vit[i]
    pct = pcts_vit[i]
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.08, f'{h:.2f}\n({red:+.2f} | {pct:.1f}%)', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#065f46')

plt.tight_layout()

for d in out_dirs:
    p = os.path.join(d, 'figura22_top3_reducao_vitimas_v6.png')
    plt.savefig(p, bbox_inches='tight')
    print('Salvo:', p)

plt.close()
print('Figuras geradas com sucesso!')
