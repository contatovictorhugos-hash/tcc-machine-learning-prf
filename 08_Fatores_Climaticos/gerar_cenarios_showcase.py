import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACAO
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE_DIR

# Procura dataset
possible_data = [
    os.path.join(BASE_DIR, "df_model_ready_tcc.csv"),
    os.path.join(BASE_DIR, "..", "df_model_ready_tcc.csv"),
    os.path.join(BASE_DIR, "..", "05_Pre_Modelagem", "df_model_ready_tcc.csv")
]
DATA_PATH = next((p for p in possible_data if os.path.exists(p)), possible_data[1])

# Procura modelo V5
possible_models = [
    os.path.join(BASE_DIR, "random_forest_v5_tcc.pkl"),
    os.path.join(BASE_DIR, "..", "06_Modelagem", "random_forest_v5_tcc.pkl"),
    os.path.join(BASE_DIR, "..", "06_Modelagem", "random_forest_model.pkl"),
    os.path.join(BASE_DIR, "..", "random_forest_model.pkl")
]
MODEL_PATH = next((p for p in possible_models if os.path.exists(p)), possible_models[1])

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams['figure.dpi'] = 150

print(f"Carregando dados de: {DATA_PATH}")
print(f"Carregando modelo de: {MODEL_PATH}")
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

TARGET = 'quantidade_acidentes'
y = df[TARGET]
X = df.drop(columns=[TARGET])

# Colunas one-hot
meteo_cols = [c for c in X.columns if c.startswith('condicao_metereologica_')]
fase_cols = [c for c in X.columns if c.startswith('fase_dia_')]
tracado_cols = [c for c in X.columns if c.startswith('is_tracado_via_')]
estacao_cols = [c for c in X.columns if c.startswith('estacao_ano_')]

# Baseline
pred_baseline = model.predict(X)
mediana_baseline = np.median(pred_baseline)
media_baseline = pred_baseline.mean()

print(f"Baseline -> Mediana: {mediana_baseline:.4f} | Media: {media_baseline:.4f}")
print(f"Target real -> Mediana: {y.median():.1f} | Media: {y.mean():.4f}")
print(f"Total de registros: {len(X):,}\n")

resultados_cenarios = []


def stats(pred):
    """Retorna mediana e media de um array de predicoes."""
    return np.median(pred), pred.mean()


# ============================================================
# CENARIO 1: FINAL DE SEMANA vs DIA UTIL
# ============================================================
print("="*70)
print("CENARIO 1: Impacto do Final de Semana na Sinistralidade")
print("="*70)

X_util = X.copy(); X_util['is_final_semana'] = 0
X_fds = X.copy(); X_fds['is_final_semana'] = 1

pred_util = model.predict(X_util)
pred_fds = model.predict(X_fds)

med_util, avg_util = stats(pred_util)
med_fds, avg_fds = stats(pred_fds)
diff1 = med_fds - med_util
pct1 = (diff1 / med_util) * 100

print(f"  Mediana (Dia Util):       {med_util:.4f}  (Media: {avg_util:.4f})")
print(f"  Mediana (Final de Semana): {med_fds:.4f}  (Media: {avg_fds:.4f})")
print(f"  Aumento absoluto (mediana): +{diff1:.4f} acidentes/trecho")
print(f"  Aumento percentual:         +{pct1:.1f}%")

resultados_cenarios.append({
    'cenario': 'Final de Semana vs Dia Util',
    'condicao_A': 'Dia Util', 'condicao_B': 'Final de Semana',
    'med_A': med_util, 'med_B': med_fds, 'avg_A': avg_util, 'avg_B': avg_fds,
    'diff': diff1, 'pct': pct1
})

# Grafico 1
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

bars = axes[0].bar(['Dia Util', 'Final de Semana'], [med_util, med_fds],
                    color=['#3b82f6', '#ef4444'], width=0.5, edgecolor='black', linewidth=0.8)
for bar, val, avg in zip(bars, [med_util, med_fds], [avg_util, avg_fds]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'Md: {val:.3f}\nAvg: {avg:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Mediana Predita de Acidentes', fontsize=12)
axes[0].set_title('Mediana: Dia Util vs Final de Semana', fontsize=13, fontweight='bold')
axes[0].annotate(f'+{diff1:.3f} acid. (+{pct1:.1f}%)',
                xy=(0.5, max(med_util, med_fds) * 0.55), fontsize=13, fontweight='bold',
                ha='center', color='#991b1b',
                bbox=dict(boxstyle='round,pad=0.4', fc='#fee2e2', ec='#ef4444', lw=1.5))

axes[1].hist(pred_util, bins=50, alpha=0.6, label='Dia Util', color='#3b82f6', edgecolor='black', linewidth=0.3)
axes[1].hist(pred_fds, bins=50, alpha=0.6, label='Final de Semana', color='#ef4444', edgecolor='black', linewidth=0.3)
axes[1].axvline(med_util, color='#1d4ed8', linestyle='--', lw=2, label=f'Md Util: {med_util:.2f}')
axes[1].axvline(med_fds, color='#b91c1c', linestyle='--', lw=2, label=f'Md FDS: {med_fds:.2f}')
axes[1].set_xlabel('Predicao de Acidentes', fontsize=12)
axes[1].set_ylabel('Frequencia', fontsize=12)
axes[1].set_title('Distribuicao das Predicoes', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cenario_01_final_semana.png'), dpi=150)
plt.close()
print("  -> Grafico salvo: cenario_01_final_semana.png\n")

# ============================================================
# CENARIO 2: TRACADO DA VIA - RETA vs CURVA vs INTERSECAO
# ============================================================
print("="*70)
print("CENARIO 2: Impacto do Tracado da Via (Reta vs Curva vs Intersecao)")
print("="*70)

def set_tracado(X_in, col_ativo):
    X_out = X_in.copy()
    for c in tracado_cols: X_out[c] = 0
    X_out[col_ativo] = 1
    return X_out

X_reta = set_tracado(X, 'is_tracado_via_Reta')
X_curva = set_tracado(X, 'is_tracado_via_Curva')
X_inter = set_tracado(X, 'is_tracado_via_Intersecao_de_Vias')

pred_reta = model.predict(X_reta)
pred_curva = model.predict(X_curva)
pred_inter = model.predict(X_inter)

med_reta, avg_reta = stats(pred_reta)
med_curva, avg_curva = stats(pred_curva)
med_inter, avg_inter = stats(pred_inter)

print(f"  Mediana (Reta):       {med_reta:.4f}  (Media: {avg_reta:.4f})")
print(f"  Mediana (Curva):      {med_curva:.4f}  (Media: {avg_curva:.4f})")
print(f"  Mediana (Intersecao): {med_inter:.4f}  (Media: {avg_inter:.4f})")
diff2 = med_reta - med_curva
pct2 = (diff2 / med_curva) * 100
print(f"  Reta vs Curva (mediana): {diff2:+.4f} ({pct2:+.1f}%)")

resultados_cenarios.append({
    'cenario': 'Tracado: Reta vs Curva',
    'condicao_A': 'Reta', 'condicao_B': 'Curva',
    'med_A': med_reta, 'med_B': med_curva, 'avg_A': avg_reta, 'avg_B': avg_curva,
    'diff': diff2, 'pct': pct2
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

labels_2 = ['Reta', 'Curva', 'Intersecao']
meds_2 = [med_reta, med_curva, med_inter]
avgs_2 = [avg_reta, avg_curva, avg_inter]
colors_2 = ['#ef4444', '#f59e0b', '#22c55e']

bars = axes[0].bar(labels_2, meds_2, color=colors_2, width=0.5, edgecolor='black', linewidth=0.8)
for bar, med, avg in zip(bars, meds_2, avgs_2):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'Md: {med:.3f}\nAvg: {avg:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Mediana Predita de Acidentes', fontsize=12)
axes[0].set_title('Mediana por Tracado de Via', fontsize=13, fontweight='bold')

axes[1].hist(pred_reta, bins=50, alpha=0.5, label='Reta', color='#ef4444', edgecolor='black', linewidth=0.3)
axes[1].hist(pred_curva, bins=50, alpha=0.5, label='Curva', color='#f59e0b', edgecolor='black', linewidth=0.3)
axes[1].hist(pred_inter, bins=50, alpha=0.5, label='Intersecao', color='#22c55e', edgecolor='black', linewidth=0.3)
axes[1].set_xlabel('Predicao de Acidentes', fontsize=12)
axes[1].set_ylabel('Frequencia', fontsize=12)
axes[1].set_title('Distribuicao das Predicoes por Tracado', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cenario_02_tracado_via.png'), dpi=150)
plt.close()
print("  -> Grafico salvo: cenario_02_tracado_via.png\n")

# ============================================================
# CENARIO 3: CONDICAO METEOROLOGICA - CHUVA vs CEU CLARO
# ============================================================
print("="*70)
print("CENARIO 3: Impacto da Chuva na Sinistralidade")
print("="*70)

def set_meteo(X_in, col_ativo):
    X_out = X_in.copy()
    for c in meteo_cols: X_out[c] = 0
    X_out[col_ativo] = 1
    return X_out

ceu_claro_col = [c for c in meteo_cols if 'Claro' in c or 'Sol' in c]
chuva_col = [c for c in meteo_cols if 'Chuva' == c.split('_')[-1]]
nevoeiro_col = [c for c in meteo_cols if 'Nevoeiro' in c or 'Neblina' in c]

X_claro = set_meteo(X, ceu_claro_col[0])
X_chuva = set_meteo(X, chuva_col[0])
X_nevoeiro = set_meteo(X, nevoeiro_col[0])

pred_claro = model.predict(X_claro)
pred_chuva = model.predict(X_chuva)
pred_nevoeiro = model.predict(X_nevoeiro)

med_claro, avg_claro = stats(pred_claro)
med_chuva, avg_chuva = stats(pred_chuva)
med_nevoeiro, avg_nevoeiro = stats(pred_nevoeiro)

diff3 = med_chuva - med_claro
pct3 = (diff3 / med_claro) * 100

print(f"  Mediana (Ceu Claro):  {med_claro:.4f}  (Media: {avg_claro:.4f})")
print(f"  Mediana (Chuva):      {med_chuva:.4f}  (Media: {avg_chuva:.4f})")
print(f"  Mediana (Nevoeiro):   {med_nevoeiro:.4f}  (Media: {avg_nevoeiro:.4f})")
print(f"  Chuva vs Claro (mediana): {diff3:+.4f} ({pct3:+.1f}%)")

resultados_cenarios.append({
    'cenario': 'Chuva vs Ceu Claro',
    'condicao_A': 'Ceu Claro', 'condicao_B': 'Chuva',
    'med_A': med_claro, 'med_B': med_chuva, 'avg_A': avg_claro, 'avg_B': avg_chuva,
    'diff': diff3, 'pct': pct3
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
labels_3 = ['Ceu Claro', 'Chuva', 'Nevoeiro']
meds_3 = [med_claro, med_chuva, med_nevoeiro]
avgs_3 = [avg_claro, avg_chuva, avg_nevoeiro]
colors_3 = ['#60a5fa', '#1e3a5f', '#94a3b8']

bars = axes[0].bar(labels_3, meds_3, color=colors_3, width=0.5, edgecolor='black', linewidth=0.8)
for bar, med, avg in zip(bars, meds_3, avgs_3):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'Md: {med:.3f}\nAvg: {avg:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Mediana Predita de Acidentes', fontsize=12)
axes[0].set_title('Mediana por Condicao Meteorologica', fontsize=13, fontweight='bold')

axes[1].hist(pred_claro, bins=50, alpha=0.5, label='Ceu Claro', color='#60a5fa', edgecolor='black', linewidth=0.3)
axes[1].hist(pred_chuva, bins=50, alpha=0.5, label='Chuva', color='#1e3a5f', edgecolor='black', linewidth=0.3)
axes[1].set_xlabel('Predicao de Acidentes', fontsize=12)
axes[1].set_ylabel('Frequencia', fontsize=12)
axes[1].set_title('Distribuicao: Ceu Claro vs Chuva', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cenario_03_meteorologia.png'), dpi=150)
plt.close()
print("  -> Grafico salvo: cenario_03_meteorologia.png\n")

# ============================================================
# CENARIO 4: FASE DO DIA - PLENO DIA vs PLENA NOITE
# ============================================================
print("="*70)
print("CENARIO 4: Impacto da Fase do Dia (Visibilidade)")
print("="*70)

def set_fase(X_in, col_ativo):
    X_out = X_in.copy()
    for c in fase_cols: X_out[c] = 0
    X_out[col_ativo] = 1
    return X_out

dia_col = [c for c in fase_cols if 'Pleno dia' in c][0]
noite_col = [c for c in fase_cols if 'Plena Noite' in c][0]
amanhecer_col = [c for c in fase_cols if 'Amanhecer' in c][0]
anoitecer_col = [c for c in fase_cols if 'Anoitecer' in c][0]

X_dia = set_fase(X, dia_col)
X_noite = set_fase(X, noite_col)
X_amanhecer = set_fase(X, amanhecer_col)
X_anoitecer = set_fase(X, anoitecer_col)

pred_dia = model.predict(X_dia)
pred_noite = model.predict(X_noite)
pred_amanhecer = model.predict(X_amanhecer)
pred_anoitecer = model.predict(X_anoitecer)

med_dia, avg_dia = stats(pred_dia)
med_noite, avg_noite = stats(pred_noite)
med_amanhecer, avg_amanhecer = stats(pred_amanhecer)
med_anoitecer, avg_anoitecer = stats(pred_anoitecer)

diff4 = med_noite - med_dia
pct4 = (diff4 / med_dia) * 100

print(f"  Mediana (Pleno Dia):   {med_dia:.4f}  (Media: {avg_dia:.4f})")
print(f"  Mediana (Amanhecer):   {med_amanhecer:.4f}  (Media: {avg_amanhecer:.4f})")
print(f"  Mediana (Anoitecer):   {med_anoitecer:.4f}  (Media: {avg_anoitecer:.4f})")
print(f"  Mediana (Plena Noite): {med_noite:.4f}  (Media: {avg_noite:.4f})")
print(f"  Noite vs Dia (mediana): {diff4:+.4f} ({pct4:+.1f}%)")

resultados_cenarios.append({
    'cenario': 'Plena Noite vs Pleno Dia',
    'condicao_A': 'Pleno Dia', 'condicao_B': 'Plena Noite',
    'med_A': med_dia, 'med_B': med_noite, 'avg_A': avg_dia, 'avg_B': avg_noite,
    'diff': diff4, 'pct': pct4
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
labels_4 = ['Amanhecer', 'Pleno Dia', 'Anoitecer', 'Plena Noite']
meds_4 = [med_amanhecer, med_dia, med_anoitecer, med_noite]
avgs_4 = [avg_amanhecer, avg_dia, avg_anoitecer, avg_noite]
colors_4 = ['#fbbf24', '#f59e0b', '#f97316', '#1e293b']

bars = axes[0].bar(labels_4, meds_4, color=colors_4, width=0.5, edgecolor='black', linewidth=0.8)
for bar, med, avg in zip(bars, meds_4, avgs_4):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'Md: {med:.3f}\nAvg: {avg:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Mediana Predita de Acidentes', fontsize=12)
axes[0].set_title('Mediana por Fase do Dia', fontsize=13, fontweight='bold')

axes[1].hist(pred_dia, bins=50, alpha=0.5, label='Pleno Dia', color='#f59e0b', edgecolor='black', linewidth=0.3)
axes[1].hist(pred_noite, bins=50, alpha=0.5, label='Plena Noite', color='#1e293b', edgecolor='black', linewidth=0.3)
axes[1].set_xlabel('Predicao de Acidentes', fontsize=12)
axes[1].set_ylabel('Frequencia', fontsize=12)
axes[1].set_title('Distribuicao: Dia vs Noite', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cenario_04_fase_dia.png'), dpi=150)
plt.close()
print("  -> Grafico salvo: cenario_04_fase_dia.png\n")

# ============================================================
# CENARIO 5: TEMPESTADE PERFEITA vs CENARIO IDEAL
# ============================================================
print("="*70)
print("CENARIO 5: Tempestade Perfeita vs Cenario Ideal")
print("="*70)

X_pior = X.copy()
X_pior['is_final_semana'] = 1
X_pior['is_feriado'] = 1
for c in tracado_cols: X_pior[c] = 0
X_pior['is_tracado_via_Reta'] = 1
for c in meteo_cols: X_pior[c] = 0
chuva_c = [c for c in meteo_cols if 'Chuva' == c.split('_')[-1]][0]
X_pior[chuva_c] = 1
for c in fase_cols: X_pior[c] = 0
X_pior[noite_col] = 1

X_melhor = X.copy()
X_melhor['is_final_semana'] = 0
X_melhor['is_feriado'] = 0
for c in tracado_cols: X_melhor[c] = 0
X_melhor['is_tracado_via_Curva'] = 1
for c in meteo_cols: X_melhor[c] = 0
X_melhor[ceu_claro_col[0]] = 1
for c in fase_cols: X_melhor[c] = 0
X_melhor[dia_col] = 1

pred_pior = model.predict(X_pior)
pred_melhor = model.predict(X_melhor)

med_pior, avg_pior = stats(pred_pior)
med_melhor, avg_melhor = stats(pred_melhor)
diff5 = med_pior - med_melhor
pct5 = (diff5 / med_melhor) * 100

print(f"  PIOR (Reta+FDS+Chuva+Noite+Feriado):    Md={med_pior:.4f}  (Avg={avg_pior:.4f})")
print(f"  MELHOR (Curva+Util+CeuClaro+Dia+Normal): Md={med_melhor:.4f}  (Avg={avg_melhor:.4f})")
print(f"  Diferenca absoluta (mediana):              +{diff5:.4f} acidentes")
print(f"  Aumento percentual:                        +{pct5:.1f}%")

resultados_cenarios.append({
    'cenario': 'Tempestade Perfeita vs Cenario Ideal',
    'condicao_A': 'Cenario Ideal', 'condicao_B': 'Tempestade Perfeita',
    'med_A': med_melhor, 'med_B': med_pior, 'avg_A': avg_melhor, 'avg_B': avg_pior,
    'diff': diff5, 'pct': pct5
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
labels_5 = ['Cenario Ideal\n(Curva+Util+Claro\n+Dia+Normal)',
            'Tempestade Perfeita\n(Reta+FDS+Chuva\n+Noite+Feriado)']
meds_5 = [med_melhor, med_pior]
avgs_5 = [avg_melhor, avg_pior]
colors_5 = ['#22c55e', '#dc2626']

bars = axes[0].bar(labels_5, meds_5, color=colors_5, width=0.5, edgecolor='black', linewidth=0.8)
for bar, med, avg in zip(bars, meds_5, avgs_5):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'Md: {med:.3f}\nAvg: {avg:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Mediana Predita de Acidentes', fontsize=12)
axes[0].set_title('Extremos: Melhor vs Pior Cenario', fontsize=13, fontweight='bold')
axes[0].annotate(f'+{diff5:.2f} acidentes\n(+{pct5:.0f}%)',
                xy=(0.5, max(meds_5)*0.45), fontsize=14, fontweight='bold',
                ha='center', color='#7f1d1d',
                bbox=dict(boxstyle='round,pad=0.5', fc='#fee2e2', ec='#dc2626', lw=2))

axes[1].hist(pred_melhor, bins=50, alpha=0.6, label='Cenario Ideal', color='#22c55e', edgecolor='black', linewidth=0.3)
axes[1].hist(pred_pior, bins=50, alpha=0.6, label='Tempestade Perfeita', color='#dc2626', edgecolor='black', linewidth=0.3)
axes[1].set_xlabel('Predicao de Acidentes', fontsize=12)
axes[1].set_ylabel('Frequencia', fontsize=12)
axes[1].set_title('Distribuicao dos Extremos', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cenario_05_extremos.png'), dpi=150)
plt.close()
print("  -> Grafico salvo: cenario_05_extremos.png\n")

# ============================================================
# GRAFICO CONSOLIDADO
# ============================================================
print("="*70)
print("GERANDO GRAFICO COMPARATIVO FINAL")
print("="*70)

fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(resultados_cenarios))
width = 0.35

vals_A = [r['med_A'] for r in resultados_cenarios]
vals_B = [r['med_B'] for r in resultados_cenarios]

bars_a = ax.bar(x - width/2, vals_A, width, label='Condicao A (Base/Segura)',
                color='#3b82f6', edgecolor='black', linewidth=0.6)
bars_b = ax.bar(x + width/2, vals_B, width, label='Condicao B (Risco/Alterada)',
                color='#ef4444', edgecolor='black', linewidth=0.6)

for i, (ba, bb, r) in enumerate(zip(bars_a, bars_b, resultados_cenarios)):
    ax.text(ba.get_x() + ba.get_width()/2, ba.get_height() + 0.02,
            f'{r["med_A"]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1e3a5f')
    ax.text(bb.get_x() + bb.get_width()/2, bb.get_height() + 0.02,
            f'{r["med_B"]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#7f1d1d')
    mid_x = (ba.get_x() + ba.get_width() + bb.get_x()) / 2
    top_y = max(r['med_A'], r['med_B']) + 0.15
    sign = '+' if r['diff'] > 0 else ''
    ax.text(mid_x, top_y, f'{sign}{r["diff"]:.2f}\n({sign}{r["pct"]:.0f}%)',
            ha='center', fontsize=9, fontweight='bold', color='#1e40af',
            bbox=dict(boxstyle='round,pad=0.3', fc='#eff6ff', ec='#93c5fd', lw=1))

short_labels = [f'{r["condicao_A"]}\nvs\n{r["condicao_B"]}' for r in resultados_cenarios]
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel('Mediana Predita de Acidentes', fontsize=13, fontweight='bold')
ax.set_title('Comparativo Geral: 5 Cenarios Showcase - Mediana (Modelo V5)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ymax = max(max(vals_A), max(vals_B)) * 1.3
ax.set_ylim(0, ymax)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'comparativo_geral_cenarios.png'), dpi=150)
plt.close()
print("  -> Grafico salvo: comparativo_geral_cenarios.png\n")

# ============================================================
# RELATORIO .TXT
# ============================================================
print("="*70)
print("GERANDO RELATORIO .TXT")
print("="*70)

with open(os.path.join(OUT_DIR, 'relatorio_cenarios_showcase.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 110 + "\n")
    f.write("                   RELATORIO DE CENARIOS SHOWCASE - MODELO V5 (Random Forest Temporal)\n")
    f.write("                   Projeto TCC: Predicao de Acidentes em Rodovias Federais Brasileiras\n")
    f.write("=" * 110 + "\n\n")

    f.write("METODOLOGIA: Utilizamos a MEDIANA das predicoes como metrica primaria.\n")
    f.write("Justificativa: A variavel alvo (quantidade_acidentes) possui assimetria severa\n")
    f.write("(media=3.39, mediana=2, max=77). A media e distorcida por outliers extremos.\n")
    f.write("A mediana representa o trecho TIPICO e e mais adequada para politicas publicas.\n")
    f.write("A media e reportada como referencia secundaria em cada cenario.\n\n")
    f.write(f"Base: {len(X):,} registros | Modelo: Random Forest V5 (R2=0.9726, MAE=0.4181)\n")
    f.write(f"Baseline -> Mediana Pred: {mediana_baseline:.4f} | Media Pred: {media_baseline:.4f}\n\n")
    f.write("-" * 110 + "\n\n")

    # Cenario 1
    r = resultados_cenarios[0]
    f.write("CENARIO 1: FINAL DE SEMANA vs DIA UTIL\n")
    f.write("-" * 110 + "\n")
    f.write("Relevancia Social: A PRF pode alocar mais efetivos nos finais de semana, quando\n")
    f.write("a sinistralidade e comprovadamente maior. Politica publica de fiscalizacao temporal.\n\n")
    f.write(f"  Condicao A (Dia Util):       Mediana = {r['med_A']:.4f}  |  Media = {r['avg_A']:.4f}\n")
    f.write(f"  Condicao B (Final de Semana): Mediana = {r['med_B']:.4f}  |  Media = {r['avg_B']:.4f}\n")
    f.write(f"  Impacto (mediana): {'+' if r['diff']>0 else ''}{r['diff']:.4f} acidentes ({'+' if r['pct']>0 else ''}{r['pct']:.1f}%)\n\n")
    f.write("  Interpretacao: O modelo preve um aumento na sinistralidade durante finais de\n")
    f.write("  semana, refletindo o aumento do trafego recreativo, consumo de alcool, e viagens\n")
    f.write("  de longa distancia que caracterizam esses periodos.\n\n")
    f.write("  Grafico: cenario_01_final_semana.png\n\n")
    f.write("-" * 110 + "\n\n")

    # Cenario 2
    r = resultados_cenarios[1]
    f.write("CENARIO 2: TRACADO DA VIA - RETA vs CURVA vs INTERSECAO\n")
    f.write("-" * 110 + "\n")
    f.write("Relevancia Social: A engenharia rodoviaria pode priorizar intervencoes em trechos\n")
    f.write("criticos (sinalizar limites de velocidade, instalar radares, barreiras centrais).\n\n")
    f.write(f"  Condicao A (Reta):       Mediana = {med_reta:.4f}  |  Media = {avg_reta:.4f}\n")
    f.write(f"  Condicao B (Curva):      Mediana = {med_curva:.4f}  |  Media = {avg_curva:.4f}\n")
    f.write(f"  Condicao C (Intersecao): Mediana = {med_inter:.4f}  |  Media = {avg_inter:.4f}\n")
    f.write(f"  Impacto Reta vs Curva (mediana): {'+' if r['diff']>0 else ''}{r['diff']:.4f} ({'+' if r['pct']>0 else ''}{r['pct']:.1f}%)\n\n")
    f.write("  Interpretacao: A diferenca entre tracados revela padroes de risco reais.\n")
    f.write("  A analise por mediana elimina o vies dos outliers extremos e mostra o\n")
    f.write("  impacto no trecho tipico brasileiro.\n\n")
    f.write("  Grafico: cenario_02_tracado_via.png\n\n")
    f.write("-" * 110 + "\n\n")

    # Cenario 3
    r = resultados_cenarios[2]
    f.write("CENARIO 3: CONDICAO METEOROLOGICA - CHUVA vs CEU CLARO\n")
    f.write("-" * 110 + "\n")
    f.write("Relevancia Social: Sistemas de alerta meteorologico integrados com a PRF podem\n")
    f.write("acionar protocolos de reducao de velocidade em rodovias durante chuvas.\n\n")
    f.write(f"  Condicao A (Ceu Claro): Mediana = {r['med_A']:.4f}  |  Media = {r['avg_A']:.4f}\n")
    f.write(f"  Condicao B (Chuva):     Mediana = {r['med_B']:.4f}  |  Media = {r['avg_B']:.4f}\n")
    f.write(f"  Condicao C (Nevoeiro):  Mediana = {med_nevoeiro:.4f}  |  Media = {avg_nevoeiro:.4f}\n")
    f.write(f"  Impacto Chuva vs Claro (mediana): {'+' if r['diff']>0 else ''}{r['diff']:.4f} ({'+' if r['pct']>0 else ''}{r['pct']:.1f}%)\n\n")
    f.write("  Interpretacao: A chuva impacta a sinistralidade pela reducao da visibilidade,\n")
    f.write("  aumento da distancia de frenagem (aquaplanagem), e comportamento do motorista.\n\n")
    f.write("  Grafico: cenario_03_meteorologia.png\n\n")
    f.write("-" * 110 + "\n\n")

    # Cenario 4
    r = resultados_cenarios[3]
    f.write("CENARIO 4: FASE DO DIA - PLENO DIA vs PLENA NOITE\n")
    f.write("-" * 110 + "\n")
    f.write("Relevancia Social: Investimentos em iluminacao rodoviaria, refletores e sinalizacao\n")
    f.write("noturna podem ser priorizados com base no impacto comprovado da fase do dia.\n\n")
    f.write(f"  Condicao A (Pleno Dia):   Mediana = {med_dia:.4f}  |  Media = {avg_dia:.4f}\n")
    f.write(f"  Condicao B (Amanhecer):   Mediana = {med_amanhecer:.4f}  |  Media = {avg_amanhecer:.4f}\n")
    f.write(f"  Condicao C (Anoitecer):   Mediana = {med_anoitecer:.4f}  |  Media = {avg_anoitecer:.4f}\n")
    f.write(f"  Condicao D (Plena Noite): Mediana = {med_noite:.4f}  |  Media = {avg_noite:.4f}\n")
    f.write(f"  Impacto Noite vs Dia (mediana): {'+' if r['diff']>0 else ''}{r['diff']:.4f} ({'+' if r['pct']>0 else ''}{r['pct']:.1f}%)\n\n")
    f.write("  Interpretacao: A variacao entre fases do dia captura o efeito combinado da\n")
    f.write("  visibilidade, do cansaco do motorista e do volume de trafego.\n\n")
    f.write("  Grafico: cenario_04_fase_dia.png\n\n")
    f.write("-" * 110 + "\n\n")

    # Cenario 5
    r = resultados_cenarios[4]
    f.write("CENARIO 5: TEMPESTADE PERFEITA vs CENARIO IDEAL\n")
    f.write("-" * 110 + "\n")
    f.write("Relevancia Social: Demonstra o pior cenario possivel combinando todos os fatores\n")
    f.write("de risco simultaneamente. Util para planejamento de contingencia da PRF e DNIT.\n\n")
    f.write("  TEMPESTADE PERFEITA: Reta + Final de Semana + Chuva + Plena Noite + Feriado\n")
    f.write("  CENARIO IDEAL:      Curva + Dia Util + Ceu Claro + Pleno Dia + Dia Normal\n\n")
    f.write(f"  Condicao A (Ideal):      Mediana = {r['med_A']:.4f}  |  Media = {r['avg_A']:.4f}\n")
    f.write(f"  Condicao B (Tempestade): Mediana = {r['med_B']:.4f}  |  Media = {r['avg_B']:.4f}\n")
    f.write(f"  Impacto (mediana): +{r['diff']:.4f} acidentes (+{r['pct']:.0f}%)\n\n")
    f.write("  Interpretacao: A combinacao de todos os fatores de risco simultaneos gera o\n")
    f.write("  cenario mais severo previsto pelo modelo. Este e o tipo de situacao onde\n")
    f.write("  a PRF deveria ter efetivo maximo e protocolos de emergencia ativados.\n\n")
    f.write("  Grafico: cenario_05_extremos.png\n\n")
    f.write("-" * 110 + "\n\n")

    # Tabela comparativa
    f.write("=" * 110 + "\n")
    f.write("TABELA COMPARATIVA CONSOLIDADA (MEDIANA)\n")
    f.write("=" * 110 + "\n\n")
    f.write(f"{'Cenario':<42s} | {'Md A':>8s} | {'Md B':>8s} | {'Avg A':>8s} | {'Avg B':>8s} | {'Diff Md':>8s} | {'%':>7s}\n")
    f.write("-" * 110 + "\n")
    for r in resultados_cenarios:
        sign = '+' if r['diff'] > 0 else ''
        f.write(f"{r['cenario']:<42s} | {r['med_A']:>8.4f} | {r['med_B']:>8.4f} | {r['avg_A']:>8.4f} | {r['avg_B']:>8.4f} | {sign}{r['diff']:>7.4f} | {sign}{r['pct']:>6.1f}%\n")
    f.write("-" * 110 + "\n\n")

    f.write("NOTA METODOLOGICA\n")
    f.write("-" * 110 + "\n")
    f.write("Md = Mediana (metrica primaria, robusta a outliers, representa o trecho tipico)\n")
    f.write("Avg = Media (referencia secundaria, sensivel a outliers extremos como trechos com 40+ acidentes)\n\n")
    f.write("A escolha da mediana como metrica primaria se justifica pela severa assimetria da variavel\n")
    f.write("alvo (quantidade_acidentes): 75% dos trechos tem ate 3 acidentes/mes, mas existem trechos\n")
    f.write("com ate 77 acidentes/mes que distorcem a media aritmetica.\n")
    f.write("=" * 110 + "\n")

print("  -> Relatorio salvo: relatorio_cenarios_showcase.txt")
print("\n[OK] Todos os cenarios showcase (MEDIANA) gerados com sucesso!")
