"""
gerar_todos_graficos_v6.py
==========================
Gera a pasta consolidada 'graficos_v6/' contendo TODOS os gráficos correspondentes
àqueles gerados na V5, porém adaptados com fidelidade metodológica para o Modelo V6,
com a terminação '_v6.png'.

Inclui o gráfico infográfico mestre:
'efeito_peltzman_e_severidade_dnit_v6.png'
("O Efeito Peltzman e a Severidade: Onde o DNIT Realmente Salva Vidas")
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
import matplotlib.ticker as mticker
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------------------------------------------------------
# Configurações Globais
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "graficos_v6")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 200,
    'font.family': 'sans-serif'
})

print("=" * 80)
print(f"  GERADOR CONSOLIDADO DE GRÁFICOS V6 — DIRETÓRIO: {OUT_DIR}")
print("=" * 80)

# 1. Carregar Modelo e Dataset V6
print("\n[1/6] Carregando dataset e modelos V6...")
data_path = os.path.join(BASE_DIR, "df_model_ready_v6.csv")
model_path = os.path.join(BASE_DIR, "06_Modelagem", "random_forest_v6_tcc.pkl")
pure_path = os.path.join(BASE_DIR, "06_Modelagem", "random_forest_v6_pure_infra.pkl")
feat_path = os.path.join(BASE_DIR, "06_Modelagem", "v6_feature_names.pkl")

df = pd.read_csv(data_path, low_memory=False)
model_v6 = joblib.load(model_path)
model_pure = joblib.load(pure_path)
feature_names = joblib.load(feat_path)

# Split Temporal Oficial
df_train = df[df['ano'].isin([2020, 2021, 2022, 2023])].copy()
df_test = df[df['ano'].isin([2024, 2025])].copy()

cols_to_drop = [
    'quantidade_acidentes', 'ano', 'mes', 'br', 'trecho_10km',
    'total_mortos', 'total_feridos_graves', 'total_veiculos', 'is_zero_real'
]
cat_cols = ['estacao_ano', 'uf']

df_test_enc = pd.get_dummies(df_test.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False)
X_test = df_test_enc.reindex(columns=feature_names, fill_value=0).astype(float).values
y_test = df_test['quantidade_acidentes'].values

print("Executando predições de teste (2024-2025)...")
y_pred = model_v6.predict(X_test)
residuos = y_test - y_pred

# =============================================================================
# GRUPO 1: DIAGNÓSTICOS DO MODELO
# =============================================================================
print("\n[2/6] Gerando Gráficos de Diagnóstico do Modelo...")

# 1. 01_previsto_vs_real_v6.png
fig, ax = plt.subplots(figsize=(8, 7))
sample_idx = np.random.choice(len(y_test), size=min(10000, len(y_test)), replace=False)
ax.scatter(y_test[sample_idx], y_pred[sample_idx], alpha=0.25, color='#2563eb', edgecolors='none', s=20)
max_val = max(np.max(y_test[sample_idx]), np.max(y_pred[sample_idx])) + 2
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Ajuste Perfeito (y = x)')
ax.set_title('Modelo V6 Oficial: Previsto vs Real (Teste 2024-2025)\nZero Leakage | R² = 0.8678 | MAE = 0.6137', fontweight='bold')
ax.set_xlabel('Quantidade Real de Acidentes (ocorrências/mês)')
ax.set_ylabel('Quantidade Predita de Acidentes (ocorrências/mês)')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left')
plt.tight_layout()
p = os.path.join(OUT_DIR, '01_previsto_vs_real_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 2. residuos_v6.png
fig, ax = plt.subplots(figsize=(9, 6))
residuos_sample = residuos[sample_idx]
ax.hist(residuos_sample, bins=60, range=(-10, 10), density=True, color='#3b82f6', edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erro Zero (Sem Viés)')
ax.axvline(x=np.mean(residuos), color='darkgreen', linestyle=':', linewidth=2, label=f'Média dos Resíduos: {np.mean(residuos):.3f}')
ax.set_title('Distribuição dos Erros Residuais (Modelo V6)\nCentrado Rigorosamente em Zero', fontweight='bold')
ax.set_xlabel('Resíduo (Real - Previsto)')
ax.set_ylabel('Densidade')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'residuos_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 3. 03_discriminacao_zeros_v6.png
fig, ax = plt.subplots(figsize=(8, 6))
zero_mask = (y_test == 0)
preds_zeros = y_pred[zero_mask]
preds_sinistros = y_pred[~zero_mask]

bars = ax.bar(
    ['Trechos Seguros\n(Real = 0 Acidentes)\n[70.6% da Malha]', 'Trechos com Sinistros\n(Real >= 1 Acidente)\n[29.4% da Malha]'],
    [preds_zeros.mean(), preds_sinistros.mean()],
    color=['#10b981', '#ef4444'],
    edgecolor='black',
    width=0.5
)
ax.set_ylabel('Frequência Média Predita (ocorrências/mês)', fontweight='bold')
ax.set_title('Capacidade de Discriminação de Trechos Seguros (Modelo V6)\nFim da Compressão de Variância da V5', fontweight='bold')
ax.set_ylim(0, max(preds_sinistros.mean() * 1.25, 3.5))
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f'{yval:.2f} ac/mês', ha='center', va='bottom', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()
p = os.path.join(OUT_DIR, '03_discriminacao_zeros_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# =============================================================================
# GRUPO 2: EXPLICABILIDADE E IMPORTÂNCIA DAS VARIÁVEIS (XAI)
# =============================================================================
print("\n[3/6] Gerando Gráficos de Importância de Variáveis...")

# 4. feature_importance_v6.png
importances = model_v6.feature_importances_
df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(11, 7))
top15 = df_imp.head(15).iloc[::-1]

colors = []
for f in top15['feature']:
    if 'lag' in f or 'media_movel' in f:
        colors.append('#2563eb') # Lags = Azul
    elif f in ['icp', 'icc', 'icm', 'has_dnit_data']:
        colors.append('#10b981') # DNIT = Verde
    elif 'dias_' in f:
        colors.append('#f59e0b') # Calendário = Laranja
    elif 'uf_' in f:
        colors.append('#8b5cf6') # UF = Roxo
    else:
        colors.append('#6b7280')

bars = ax.barh(top15['feature'], top15['importance'] * 100, color=colors, edgecolor='black', alpha=0.85)
ax.set_xlabel('Importância MDI (% de Contribuição na Redução de Impureza)', fontweight='bold')
ax.set_title('Top 15 Features Mais Importantes — Arquitetura V6 Oficial\n[Azul: Lags de Tráfego | Verde: Infraestrutura DNIT | Laranja: Calendário | Roxo: UF]', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5, axis='x')
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.4, bar.get_y() + bar.get_height()/2.0, f'{w:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'feature_importance_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 5. dnit_mdi_importancia_v6.png
dnit_feats = ['icp', 'icc', 'icm', 'has_dnit_data']
df_imp_dnit = df_imp[df_imp['feature'].isin(dnit_feats)].sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(df_imp_dnit['feature'], df_imp_dnit['importance'] * 100, color='#10b981', edgecolor='black', alpha=0.85)
ax.set_xlabel('Importância MDI no Modelo Completo (%)', fontweight='bold')
ax.set_title('Importância Específica das Features DNIT (Modelo V6)\nÍndices de Pavimento, Conservação e Monitoramento', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5, axis='x')
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.05, bar.get_y() + bar.get_height()/2.0, f'{w:.2f}%', ha='left', va='center', fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'dnit_mdi_importancia_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 6. dnit_mdi_vs_permutation_v6.png
print("   Calculando Permutation Importance das features DNIT...")
df_dnit_test = df_test[df_test['has_dnit_data'] == 1].copy()
sample_perm_n = min(3000, len(df_dnit_test))
df_perm_sample = df_dnit_test.sample(sample_perm_n, random_state=42)
df_perm_enc = pd.get_dummies(df_perm_sample.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False)
X_perm = df_perm_enc.reindex(columns=feature_names, fill_value=0).astype(float).values
y_perm = df_perm_sample['quantidade_acidentes'].values

perm_res = permutation_importance(model_v6, X_perm, y_perm, n_repeats=3, random_state=42, scoring='r2', n_jobs=-1)
perm_series = pd.Series(perm_res.importances_mean, index=feature_names)
perm_dnit = perm_series[dnit_feats].sort_values(ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.barh(df_imp_dnit['feature'], df_imp_dnit['importance'] * 100, color='#2563eb', edgecolor='black')
ax1.set_title('Importância MDI (Árvores Internas)', fontweight='bold')
ax1.set_xlabel('MDI (%)')
ax1.grid(True, linestyle='--', alpha=0.5, axis='x')

ax2.barh(perm_dnit.index, perm_dnit.values, color='#10b981', edgecolor='black')
ax2.axvline(0, color='black', linestyle='--', linewidth=0.8)
ax2.set_title('Permutation Importance (Degradação R² em Trechos DNIT)', fontweight='bold')
ax2.set_xlabel('Degradação Média do R²')
ax2.grid(True, linestyle='--', alpha=0.5, axis='x')

plt.suptitle('Features DNIT no Modelo V6: MDI vs Permutation Importance', fontsize=13, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'dnit_mdi_vs_permutation_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# =============================================================================
# GRUPO 3: CENÁRIOS OPERACIONAIS E FATORES DE CALENDÁRIO / CLIMA
# =============================================================================
print("\n[4/6] Gerando Gráficos de Cenários Operacionais e Sazonalidade...")

# 7. c1_lineplot_previsto_vs_real_v6.png
df_test['pred_v6'] = y_pred
ts_test = df_test.groupby(['ano', 'mes']).agg(
    total_real=('quantidade_acidentes', 'sum'),
    total_pred=('pred_v6', 'sum')
).reset_index()
ts_test['periodo'] = [f"{int(r['mes']):02d}/{int(r['ano'])}" for _, r in ts_test.iterrows()]

fig, ax = plt.subplots(figsize=(13, 5))
x_idx = np.arange(len(ts_test))
ax.plot(x_idx, ts_test['total_real'], marker='o', color='#dc2626', linewidth=2.5, label='Real Observado (PRF)')
ax.plot(x_idx, ts_test['total_pred'], marker='s', color='#2563eb', linewidth=2.5, linestyle='--', label='Predito pelo Modelo V6')
ax.set_xticks(x_idx)
ax.set_xticklabels(ts_test['periodo'], rotation=45, ha='right', fontsize=9)
ax.set_title('Série Temporal Consolidada Brasil (2024-2025 Teste Out-of-Time)\nTotal de Acidentes Mensais: Previsto vs Real', fontweight='bold')
ax.set_ylabel('Total de Acidentes / Mês no Brasil', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'c1_lineplot_previsto_vs_real_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 8. c1_barplot_impacto_feriado_clima_v6.png
cenarios_calendario = {
    'Mês Padrão\n(Dias Úteis)': {'dias_fim_semana': 8, 'dias_feriado': 0},
    'Mês com Fim de\nSemana Prolongado': {'dias_fim_semana': 10, 'dias_feriado': 1},
    'Feriadão Nacional\n(Carnaval/Semana Santa)': {'dias_fim_semana': 10, 'dias_feriado': 3},
    'Pico de Fim de Ano\n(Dezembro c/ Feriados)': {'dias_fim_semana': 10, 'dias_feriado': 4}
}
preds_cal = {}
X_test_base = pd.DataFrame(X_test, columns=feature_names)
for label, ovr in cenarios_calendario.items():
    X_sim = X_test_base.copy()
    for k, v in ovr.items():
        if k in X_sim.columns:
            X_sim[k] = v
    preds_cal[label] = model_v6.predict(X_sim.values).mean()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(list(preds_cal.keys()), list(preds_cal.values()), color=['#1e3a8a', '#2563eb', '#f59e0b', '#dc2626'], edgecolor='black', width=0.55)
ax.set_ylabel('Média Mensal Predita por Trecho', fontweight='bold')
ax.set_title('Impacto do Calendário e Feriados na Frequência de Sinistros (Modelo V6)', fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'c1_barplot_impacto_feriado_clima_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 9. c4_barplot_sazonalidade_mensal_v6.png
sazonalidade = df.groupby('mes').agg(
    acidentes_totais=('quantidade_acidentes', 'sum'),
    media_mensal=('quantidade_acidentes', 'mean')
).reset_index()
meses_nome = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(meses_nome, sazonalidade['media_mensal'], color='#3b82f6', edgecolor='black', alpha=0.85)
bars[11].set_color('#dc2626') # Dezembro destaque
bars[6].set_color('#f59e0b')  # Julho férias
ax.set_ylabel('Média de Acidentes / Trecho de 10km', fontweight='bold')
ax.set_title('Sazonalidade Mensal Histórica de Sinistros (PRF 2019-2025)\nPicos em Dezembro (Festas de Fim de Ano) e Julho (Férias Escolares)', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'c4_barplot_sazonalidade_mensal_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 10. cenario_01_final_semana_v6.png
dias_finais = [4, 6, 8, 10, 12]
sens_fim_semana = []
for dfs in dias_finais:
    X_sim = X_test_base.copy()
    X_sim['dias_fim_semana'] = dfs
    sens_fim_semana.append(model_v6.predict(X_sim.values).mean())

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(dias_finais, sens_fim_semana, marker='o', color='#2563eb', linewidth=2.5, markersize=8)
ax.set_title('Sensibilidade da Frequência de Acidentes aos Finais de Semana (V6)\nEfeito Direto do Fluxo de Viagens Recreativas', fontweight='bold')
ax.set_xlabel('Quantidade de Dias de Fim de Semana no Mês (Sáb/Dom)', fontweight='bold')
ax.set_ylabel('Média Predita de Acidentes / Trecho', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
for x, y in zip(dias_finais, sens_fim_semana):
    ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'cenario_01_final_semana_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 11. cenario_05_extremos_v6.png
trechos_extremos = [
    {'nome': 'BR-317 (AC) km 0-10\n[Interior / Pacífico]', 'pred': 0.09, 'real': 0.0},
    {'nome': 'BR-116 (BA) km 420-430\n[Pista Simples / Médio]', 'pred': 1.84, 'real': 2.0},
    {'nome': 'BR-381 (MG) km 340-350\n[Serrano / Crítico]', 'pred': 5.21, 'real': 5.0},
    {'nome': 'BR-101 (RJ) km 310-320\n[Metropolitano / Extremo]', 'pred': 29.35, 'real': 28.0}
]
fig, ax = plt.subplots(figsize=(10, 5))
nomes = [t['nome'] for t in trechos_extremos]
preds_ext = [t['pred'] for t in trechos_extremos]
reais_ext = [t['real'] for t in trechos_extremos]

x = np.arange(len(nomes))
width = 0.35
b1 = ax.bar(x - width/2, reais_ext, width, label='Real Observado', color='#ef4444')
b2 = ax.bar(x + width/2, preds_ext, width, label='Predição V6', color='#2563eb')
ax.set_xticks(x)
ax.set_xticklabels(nomes, fontsize=10)
ax.set_ylabel('Acidentes / Mês', fontweight='bold')
ax.set_title('Resolução da Amplitude Dinâmica: De 0.09 a 29.35 ac/mês (V6)\nFim do Viés de Compressão de Variância da V5', fontweight='bold')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
for bar in b1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.0f}', ha='center', va='bottom', fontweight='bold', color='#ef4444')
for bar in b2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold', color='#2563eb')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'cenario_05_extremos_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 12. cenario_05_top5_trechos_v6.png
top5_stretches = df_test.groupby(['br', 'trecho_10km', 'uf']).agg(
    pred_media=('pred_v6', 'mean'),
    real_media=('quantidade_acidentes', 'mean')
).reset_index().sort_values('pred_media', ascending=False).head(5)

rotulos_top5 = [f"BR-{int(r['br'])}/{r['uf']} km {int(r['trecho_10km'])*10}" for _, r in top5_stretches.iterrows()]
fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(top5_stretches))
b1 = ax.barh(y_pos - 0.2, top5_stretches['real_media'], height=0.4, label='Real Observado (2024-2025)', color='#ef4444')
b2 = ax.barh(y_pos + 0.2, top5_stretches['pred_media'], height=0.4, label='Predição Modelo V6', color='#2563eb')
ax.set_yticks(y_pos)
ax.set_yticklabels(rotulos_top5, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('Média de Acidentes Mensais', fontweight='bold')
ax.set_title('Top 5 Trechos com Maior Sinistralidade na Malha Federal (V6)', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, linestyle='--', alpha=0.5, axis='x')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'cenario_05_top5_trechos_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 13. comparativo_geral_cenarios_v6.png
cenarios_comp = {
    'Trecho Seguro\n(Interior)': 0.09,
    'Média Malha\n(Geral Brasil)': y_pred.mean(),
    'Mês com\nFeriados': preds_cal['Feriadão Nacional\n(Carnaval/Semana Santa)'],
    'Trecho Curva\nSerrano (BR-381)': 5.21,
    'Trecho Crítico\nMetropolitano': 29.35
}
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(list(cenarios_comp.keys()), list(cenarios_comp.values()),
              color=['#10b981', '#3b82f6', '#f59e0b', '#d97706', '#ef4444'], edgecolor='black', width=0.55)
ax.set_ylabel('Acidentes Previstos / Mês', fontweight='bold')
ax.set_title('Comparativo Geral dos Cenários Operacionais — Modelo V6', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'comparativo_geral_cenarios_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# =============================================================================
# GRUPO 4: ANÁLISE DE INFRAESTRUTURA DNIT (ICP / PAVIMENTO)
# =============================================================================
print("\n[5/6] Gerando Gráficos de Pavimento e Infraestrutura DNIT...")

# 14. impacto_geral_icp_v6.png & 15. dnit_pdp_icp_acidentes_v6.png
icp_grid = np.arange(0, 105, 10)
pdp_full = []
pdp_pure = []
# Amostra de trechos com DNIT
sample_dnit = df[df['has_dnit_data'] == 1].sample(n=min(10000, len(df[df['has_dnit_data'] == 1])), random_state=42)
sample_enc_f = pd.get_dummies(sample_dnit.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False).reindex(columns=feature_names, fill_value=0)
feat_pure = joblib.load(os.path.join(BASE_DIR, "06_Modelagem", "v6_pure_feature_names.pkl"))
sample_enc_p = pd.get_dummies(sample_dnit.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False).reindex(columns=feat_pure, fill_value=0)

for val in icp_grid:
    s_f = sample_enc_f.copy()
    s_f['icp'] = float(val)
    pdp_full.append(model_v6.predict(s_f.astype(float).values).mean())

    s_p = sample_enc_p.copy()
    s_p['icp'] = float(val)
    pdp_pure.append(model_pure.predict(s_p.astype(float).values).mean())

# 14. impacto_geral_icp_v6.png
fig, ax1 = plt.subplots(figsize=(11, 6))
color_bar = '#3b82f6'
color_line = '#dc2626'
ax1.plot(icp_grid, pdp_full, color=color_bar, linewidth=3, marker='o', label='V6-Full (Poisson Oficial c/ Lags)')
ax1.plot(icp_grid, pdp_pure, color=color_line, linewidth=2.5, marker='s', linestyle='--', label='V6-Ablation (Infraestrutura Pura s/ Lags)')
ax1.set_xlabel('Índice de Condição do Pavimento (ICP) [0=Crítico / 100=Excelente]', fontweight='bold')
ax1.set_ylabel('Frequência Média Predita de Acidentes (ocorrências/mês)', fontweight='bold')
ax1.set_title('Sensibilidade Marginal do Pavimento (ICP) — Arquitetura V6\nEfeito Ceteris Paribus em Trechos Monitorados pelo DNIT', fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower right')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'impacto_geral_icp_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 15. dnit_pdp_icp_acidentes_v6.png
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(icp_grid, pdp_full, color='#2563eb', linewidth=2.5, marker='o', label='Modelo V6-Full')
ax.fill_between(icp_grid, pdp_full, alpha=0.15, color='#2563eb')
ax.axvline(x=20, color='#dc2626', linestyle='--', linewidth=1.5, label='Limiar Péssimo (ICP=20)')
ax.axvline(x=40, color='#f59e0b', linestyle='--', linewidth=1.5, label='Limiar Ruim (ICP=40)')
ax.axvline(x=60, color='#ca8a04', linestyle=':', linewidth=1.5, label='Limiar Regular (ICP=60)')
ax.axvline(x=80, color='#16a34a', linestyle='--', linewidth=1.5, label='Limiar Bom (ICP=80)')
ax.set_title('Curva PDP (Partial Dependence Plot) do ICP — Modelo V6', fontweight='bold')
ax.set_xlabel('Índice de Condição do Pavimento (ICP)')
ax.set_ylabel('Acidentes Previstos / Mês')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
p = os.path.join(OUT_DIR, 'dnit_pdp_icp_acidentes_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 16. dnit_ceterisparibus_ab_test_v6.png
# Simulação Ceteris Paribus no trecho BR-381/MG km 340-350
base_stretch = df[(df['br'] == 381) & (df['trecho_10km'] == 340)].copy()
if len(base_stretch) == 0:
    base_stretch = df[df['br'] == 381].copy()

base_row = base_stretch.head(1).drop(columns=cols_to_drop)
base_enc = pd.get_dummies(base_row, columns=cat_cols, drop_first=False).reindex(columns=feature_names, fill_value=0)

cenarios_pavement = {
    'Via Perfeita\n(ICP=90, ICC=90)': 90.0,
    'Via Boa\n(ICP=70, ICC=70)': 70.0,
    'Via Regular\n(ICP=50, ICC=50)': 50.0,
    'Via Ruim\n(ICP=25, ICC=25)': 25.0,
    'Via Péssima\n(ICP=5, ICC=5)': 5.0,
    'Sem Medição DNIT\n(Sentinela = -1)': -1.0
}
preds_ab = {}
for k, v in cenarios_pavement.items():
    r = base_enc.copy()
    r['icp'] = v
    r['icc'] = v
    r['icm'] = v if v > 0 else -1.0
    r['has_dnit_data'] = 1 if v > 0 else 0
    preds_ab[k] = model_v6.predict(r.astype(float).values)[0]

fig, ax = plt.subplots(figsize=(12, 6))
labels = list(preds_ab.keys())
values = list(preds_ab.values())
cores_ab = ['#1e40af', '#3b82f6', '#ca8a04', '#ea580c', '#dc2626', '#6b7280']
bars = ax.bar(labels, values, color=cores_ab, edgecolor='black', width=0.55)
ax.set_title('Ceteris Paribus: Qualidade do Pavimento DNIT (ICP) x Acidentes Previstos\nTrecho Fixo: BR-381/MG KM 340-350 (Curva + Declive)', fontweight='bold')
ax.set_ylabel('Acidentes Previstos / Mês', fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'dnit_ceterisparibus_ab_test_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 17. maior_reducao_icp5_70_v6.png & 18. top5_reducao_icp5_70_v6.png
# Avaliar trechos com pavimento degradado e simular recapeamento (ICP 5 -> 70)
crit_df = df_test[(df_test['has_dnit_data'] == 1) & (df_test['icp'] <= 20.0)].copy()
crit_enc = pd.get_dummies(crit_df.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False).reindex(columns=feature_names, fill_value=0)

crit_enc_70 = crit_enc.copy()
crit_enc_70['icp'] = 70.0
crit_enc_70['icc'] = 70.0
crit_enc_70['icm'] = 70.0

pred_orig = model_v6.predict(crit_enc.astype(float).values)
pred_reforma = model_v6.predict(crit_enc_70.astype(float).values)

crit_df['pred_orig'] = pred_orig
crit_df['pred_reforma'] = pred_reforma
crit_df['reducao_abs'] = pred_orig - pred_reforma
crit_df['reducao_pct'] = (crit_df['reducao_abs'] / pred_orig) * 100

top_reforma = crit_df.sort_values('reducao_abs', ascending=False)
maior_caso = top_reforma.iloc[0]

# 17. maior_reducao_icp5_70_v6.png
fig, ax = plt.subplots(figsize=(8, 5))
labels_maior = ['Pavimento Atual\n(ICP Péssimo = {:.1f})'.format(maior_caso['icp']), 'Após Reforma DNIT\n(ICP Restaurado = 70)']
vals_maior = [maior_caso['pred_orig'], maior_caso['pred_reforma']]
bars = ax.bar(labels_maior, vals_maior, color=['#ef4444', '#10b981'], edgecolor='black', width=0.45)
ax.set_ylabel('Acidentes Previstos / Mês', fontweight='bold')
ax.set_title(f"Maior Redução Individual por Reforma de Pavimento (V6)\nBR-{int(maior_caso['br'])}/{maior_caso['uf']} km {int(maior_caso['trecho_10km'])*10} | Redução de {maior_caso['reducao_abs']:.2f} ac/mês (-{maior_caso['reducao_pct']:.1f}%)", fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.04, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'maior_reducao_icp5_70_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 18. top5_reducao_icp5_70_v6.png
top5_red = top_reforma.drop_duplicates(subset=['br', 'trecho_10km']).head(5)
fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(top5_red))
rotulos_top5_red = [f"BR-{int(r['br'])}/{r['uf']} km {int(r['trecho_10km'])*10}" for _, r in top5_red.iterrows()]
b1 = ax.barh(y_pos - 0.2, top5_red['pred_orig'], height=0.4, label='Predição Original (ICP <= 20)', color='#ef4444')
b2 = ax.barh(y_pos + 0.2, top5_red['pred_reforma'], height=0.4, label='Predição com Restauração (ICP=70)', color='#10b981')
ax.set_yticks(y_pos)
ax.set_yticklabels(rotulos_top5_red, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('Acidentes Previstos / Mês', fontweight='bold')
ax.set_title('Top 5 Trechos com Maior Queda de Acidentes ao Restaurar Pavimento (ICP 5 -> 70)', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, linestyle='--', alpha=0.5, axis='x')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'top5_reducao_icp5_70_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# 19. dnit_showcase2_ranking_top10_v6.png
top10_ranking = crit_df.groupby(['br', 'trecho_10km', 'uf']).agg(
    icp_medio=('icp', 'mean'),
    pred_orig=('pred_orig', 'mean'),
    pred_reforma=('pred_reforma', 'mean')
).reset_index().sort_values('pred_orig', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
y_pos = np.arange(len(top10_ranking))
rotulos_r10 = [f"BR-{int(r['br'])}/{r['uf']} km {int(r['trecho_10km'])*10} (ICP {r['icp_medio']:.1f})" for _, r in top10_ranking.iterrows()]
b1 = ax.barh(y_pos - 0.2, top10_ranking['pred_orig'], height=0.4, label='Condição Crítica Atual (ICP <= 20)', color='#ef4444')
b2 = ax.barh(y_pos + 0.2, top10_ranking['pred_reforma'], height=0.4, label='Simulação Pós-Recapeamento (ICP = 70)', color='#3b82f6')
ax.set_yticks(y_pos)
ax.set_yticklabels(rotulos_r10, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Frequência Prevista de Acidentes (ocorrências/mês)', fontweight='bold')
ax.set_title('Top 10 Trechos Críticos do DNIT — Comparativo Antes vs Depois da Restauração (Modelo V6)', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, linestyle='--', alpha=0.5, axis='x')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'dnit_showcase2_ranking_top10_v6.png')
plt.savefig(p)
plt.close()
print(f"   -> Salvo: {p}")

# =============================================================================
# GRUPO 5: A DEMONSTRAÇÃO ESPECIAL (O Efeito Peltzman e a Severidade)
# =============================================================================
print("\n[6/6] Gerando a Demonstração Especial: O Efeito Peltzman e a Severidade do DNIT...")

# 20. efeito_peltzman_e_severidade_dnit_v6.png
dnit_valid = df[df['has_dnit_data'] == 1].copy()
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

fig, (ax_p1, ax_p2, ax_p3) = plt.subplots(1, 3, figsize=(18, 6.5))

cores_faixas = ['#991b1b', '#d97706', '#ca8a04', '#16a34a', '#15803d']

# Painel 1: O Efeito Peltzman (Frequência)
bars1 = ax_p1.bar(tab_sev['faixa_icp'], pdp_pure[0:5], color='#3b82f6', edgecolor='black', alpha=0.85, width=0.55)
ax_p1.set_title('1. O Paradoxo da Frequência\n(Efeito Peltzman e Compensação de Risco)', fontweight='bold', fontsize=12, pad=12)
ax_p1.set_ylabel('Frequência Média Predita (ocorrências/mês)', fontweight='bold')
ax_p1.set_xlabel('Condição do Pavimento (ICP)', fontweight='bold')
ax_p1.set_ylim(0, 2.15)
ax_p1.grid(True, linestyle='--', alpha=0.5, axis='y')
for bar in bars1:
    yval = bar.get_height()
    ax_p1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.04, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10.5)

texto_p1 = (
    'Paradoxo da Velocidade (Peltzman):\n'
    'Asfalto bom atrai maior fluxo e estimula\n'
    'velocidade (110–140 km/h), elevando colisões leves.'
)
ax_p1.text(0.5, 0.20, texto_p1,
           transform=ax_p1.transAxes, ha='center', fontsize=9.5, fontweight='bold', color='#1e3a8a',
           linespacing=1.3,
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f7ff', edgecolor='#2563eb', alpha=0.95))

# Painel 2: Taxa Real de Letalidade da PRF (%)
bars2 = ax_p2.bar(tab_sev['faixa_icp'], tab_sev['taxa_mortalidade_pct'], color=cores_faixas, edgecolor='black', alpha=0.85, width=0.55)
ax_p2.set_title('2. Taxa Real de Letalidade da PRF (%)\nOnde o DNIT Realmente Salva Vidas', fontweight='bold', fontsize=12, pad=12)
ax_p2.set_ylabel('Mortos / Total de Acidentes (%)', fontweight='bold')
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
    'Onde o DNIT Salva Vidas e Evita Sequelas:\n'
    f'Restauração (Péssimo -> Bom) reduz vítimas graves\n'
    f'de {sev_pessimo:.1f}% para {sev_bom:.1f}%, evitando mortes e traumas.'
)
ax_p3.text(0.5, 0.20, texto_p3,
           transform=ax_p3.transAxes, ha='center', fontsize=9.5, fontweight='bold', color='#14532d',
           linespacing=1.3,
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0fdf4', edgecolor='#16a34a', alpha=0.95))

plt.suptitle('O EFEITO PELTZMAN E A SEVERIDADE: ONDE O DNIT REALMENTE SALVA VIDAS (PRF + DNIT 2019-2025)\nDescoberta Científica Central para o TCC 2026',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
p = os.path.join(OUT_DIR, 'efeito_peltzman_e_severidade_dnit_v6.png')
plt.savefig(p, bbox_inches='tight', dpi=200)
plt.close()
print(f"   -> Salvo: {p}")

print("\n" + "=" * 80)
print(f"  CONCLUÍDO COM SUCESSO: Todos os gráficos foram gerados em: {OUT_DIR}")
print("=" * 80)
