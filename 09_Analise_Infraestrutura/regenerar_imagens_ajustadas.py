# -*- coding: utf-8 -*-
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

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

# =============================================================================
# 1. FIGURA: TOP 3 VÍTIMAS SEVERAS (AJUSTE DE LEGENDA SEM SOBREPOSIÇÃO)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5.0), dpi=200)

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
ax.set_title('Top 3 Trechos com Maior Queda no Número de Vítimas Severas (Modelo V6)', fontweight='bold', pad=14)
ax.set_xticks(x)
ax.set_xticklabels(trechos_vit, fontweight='bold')

# Limite do eixo Y expandido para 4.2 para acomodar a legenda perfeitamente no topo central
ax.set_ylim(0, 4.3)
ax.legend(loc='upper center', ncol=2, frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc', fontsize=9.5)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

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

# =============================================================================
# 2. FIGURA: EFEITO PELTZMAN (TÍTULO LIMPO "Efeito de Peltzman...")
# =============================================================================
print("Carregando dados e modelo para Efeito Peltzman...")
df = pd.read_csv('05_Pre_Modelagem/df_model_ready_v6.csv', low_memory=False)
model_pure = joblib.load('06_Modelagem/random_forest_v6_pure_infra.pkl')
feat_pure = joblib.load('06_Modelagem/v6_pure_feature_names.pkl')

cols_to_drop = ['quantidade_acidentes', 'ano', 'mes', 'br', 'trecho_10km', 'total_mortos', 'total_feridos_graves', 'total_veiculos', 'is_zero_real']
cat_cols = ['estacao_ano', 'uf']

dnit_valid = df[df['has_dnit_data'] == 1].copy()
sample_dnit = dnit_valid.sample(n=min(10000, len(dnit_valid)), random_state=42)
sample_enc_p = pd.get_dummies(sample_dnit.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False).reindex(columns=feat_pure, fill_value=0)

icp_grid = np.arange(0, 105, 20)
pdp_pure = []
for val in icp_grid:
    s_p = sample_enc_p.copy()
    s_p['icp'] = float(val)
    pdp_pure.append(model_pure.predict(s_p.astype(float).values).mean())

dnit_valid['faixa_icp'] = pd.cut(
    dnit_valid['icp'],
    bins=[-1, 20, 40, 60, 80, 100],
    labels=['Péssimo\n(0-20)', 'Ruim\n(21-40)', 'Regular\n(41-60)', 'Bom\n(61-80)', 'Ótimo\n(81-100)']
)

tab_sev = dnit_valid.groupby('faixa_icp', observed=False).agg(
    total_acidentes=('quantidade_acidentes', 'sum'),
    total_mortos=('total_mortos', 'sum'),
    total_feridos_graves=('total_feridos_graves', 'sum'),
    media_acidentes_pred=('quantidade_acidentes', 'mean')
).reset_index()

tab_sev['taxa_mortalidade_pct'] = (tab_sev['total_mortos'] / tab_sev['total_acidentes']) * 100
tab_sev['taxa_vitimas_severas_pct'] = ((tab_sev['total_mortos'] + tab_sev['total_feridos_graves']) / tab_sev['total_acidentes']) * 100

fig, (ax_p1, ax_p2, ax_p3) = plt.subplots(1, 3, figsize=(18, 6.2))
cores_faixas = ['#991b1b', '#d97706', '#ca8a04', '#16a34a', '#15803d']

# Painel 1: O Paradoxo da Frequência
bars1 = ax_p1.bar(tab_sev['faixa_icp'], pdp_pure[0:5], color='#3b82f6', edgecolor='black', alpha=0.85, width=0.55)
ax_p1.set_title('1. Paradoxo da Frequência\n(Compensação de Risco)', fontweight='bold', fontsize=12, pad=12)
ax_p1.set_ylabel('Frequência Média Predita (ocorrências/mês)', fontweight='bold')
ax_p1.set_xlabel('Condição do Pavimento (ICP)', fontweight='bold')
ax_p1.set_ylim(0, 2.15)
ax_p1.grid(True, linestyle='--', alpha=0.5, axis='y')
for bar in bars1:
    yval = bar.get_height()
    ax_p1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.04, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10.5)

texto_p1 = (
    'Compensação de Velocidade:\n'
    'Asfalto bom estimula maior velocidade de fluxo,\n'
    'elevando a ocorrência de colisões leves.'
)
ax_p1.text(0.5, 0.20, texto_p1,
           transform=ax_p1.transAxes, ha='center', fontsize=9.5, fontweight='bold', color='#1e3a8a',
           linespacing=1.3,
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f7ff', edgecolor='#2563eb', alpha=0.95))

# Painel 2: Taxa de Letalidade da PRF (%)
bars2 = ax_p2.bar(tab_sev['faixa_icp'], tab_sev['taxa_mortalidade_pct'], color=cores_faixas, edgecolor='black', alpha=0.85, width=0.55)
ax_p2.set_title('2. Taxa de Letalidade Real (%)\n(Mortos / Total de Acidentes)', fontweight='bold', fontsize=12, pad=12)
ax_p2.set_ylabel('Taxa de Mortalidade (%)', fontweight='bold')
ax_p2.set_xlabel('Condição do Pavimento (ICP)', fontweight='bold')
ax_p2.set_ylim(0, 14.5)
ax_p2.grid(True, linestyle='--', alpha=0.5, axis='y')
for bar in bars2:
    yval = bar.get_height()
    ax_p2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.35, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10.5)

dif_let = ((tab_sev.loc[0, 'taxa_mortalidade_pct'] - tab_sev.loc[3, 'taxa_mortalidade_pct']) / tab_sev.loc[3, 'taxa_mortalidade_pct']) * 100
texto_p2 = (
    f'Péssimo é +{dif_let:.1f}% Mais Letal que o Bom:\n'
    'Desvio de crateras e estouro de pneus causam\n'
    'invasão da contramão e choques frontais fatais.'
)
ax_p2.text(0.5, 0.20, texto_p2,
           transform=ax_p2.transAxes, ha='center', fontsize=9.5, fontweight='bold', color='#991b1b',
           linespacing=1.3,
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#fef2f2', edgecolor='#dc2626', alpha=0.95))

# Painel 3: Índice de Vítimas Severas (%)
bars3 = ax_p3.bar(tab_sev['faixa_icp'], tab_sev['taxa_vitimas_severas_pct'], color=cores_faixas, edgecolor='black', alpha=0.85, width=0.55)
ax_p3.set_title('3. Índice de Vítimas Severas (%)\n(Mortos + Feridos Graves por Acidente)', fontweight='bold', fontsize=12, pad=12)
ax_p3.set_ylabel('Vítimas Severas / Acidentes (%)', fontweight='bold')
ax_p3.set_xlabel('Condição do Pavimento (ICP)', fontweight='bold')
ax_p3.set_ylim(0, 50.0)
ax_p3.grid(True, linestyle='--', alpha=0.5, axis='y')
for bar in bars3:
    yval = bar.get_height()
    ax_p3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10.5)

sev_pessimo = tab_sev.loc[0, 'taxa_vitimas_severas_pct']
sev_bom = tab_sev.loc[3, 'taxa_vitimas_severas_pct']
texto_p3 = (
    'Preservação da Integridade Física:\n'
    f'Restauração (Péssimo -> Bom) reduz traumas\n'
    f'de {sev_pessimo:.1f}% para {sev_bom:.1f}%, evitando mortes e sequelas.'
)
ax_p3.text(0.5, 0.20, texto_p3,
           transform=ax_p3.transAxes, ha='center', fontsize=9.5, fontweight='bold', color='#14532d',
           linespacing=1.3,
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0fdf4', edgecolor='#16a34a', alpha=0.95))

# Título limpo e objetivo conforme solicitado pelo usuário
plt.suptitle('Efeito de Peltzman e a Severidade Real do Pavimento',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

targets_peltzman = [
    'documento_tcc/figuras/figura22_efeito_peltzman_v6.png',
    'documento_tcc/figuras/efeito_peltzman_e_severidade_dnit_v6.png',
    'graficos_v6/efeito_peltzman_e_severidade_dnit_v6.png'
]
for p in targets_peltzman:
    plt.savefig(p, bbox_inches='tight', dpi=200)
    print('Salvo:', p)

plt.close()
print('Todas as imagens foram regeneradas com sucesso!')
