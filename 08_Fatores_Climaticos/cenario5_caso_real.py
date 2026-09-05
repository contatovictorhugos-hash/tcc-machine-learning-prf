import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE_DIR

possible_data = [
    os.path.join(BASE_DIR, "df_model_ready_tcc.csv"),
    os.path.join(BASE_DIR, "..", "df_model_ready_tcc.csv"),
    os.path.join(BASE_DIR, "..", "05_Pre_Modelagem", "df_model_ready_tcc.csv")
]
DATA_PATH = next((p for p in possible_data if os.path.exists(p)), possible_data[1])

possible_models = [
    os.path.join(BASE_DIR, "random_forest_v5_tcc.pkl"),
    os.path.join(BASE_DIR, "..", "06_Modelagem", "random_forest_v5_tcc.pkl"),
    os.path.join(BASE_DIR, "..", "06_Modelagem", "random_forest_model.pkl"),
    os.path.join(BASE_DIR, "..", "random_forest_model.pkl")
]
MODEL_PATH = next((p for p in possible_models if os.path.exists(p)), possible_models[1])

print(f"Carregando dados de: {DATA_PATH}")
print(f"Carregando modelo de: {MODEL_PATH}")
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

TARGET = 'quantidade_acidentes'
y = df[TARGET]
X = df.drop(columns=[TARGET])

meteo_cols = [c for c in X.columns if c.startswith('condicao_metereologica_')]
fase_cols = [c for c in X.columns if c.startswith('fase_dia_')]
tracado_cols = [c for c in X.columns if c.startswith('is_tracado_via_')]

dia_col = [c for c in fase_cols if 'Pleno dia' in c][0]
noite_col = [c for c in fase_cols if 'Plena Noite' in c][0]
ceu_claro_col = [c for c in meteo_cols if 'Claro' in c][0]
chuva_col = [c for c in meteo_cols if 'Chuva' == c.split('_')[-1]][0]

nomes_meses = {1:'Janeiro',2:'Fevereiro',3:'Marco',4:'Abril',5:'Maio',6:'Junho',
               7:'Julho',8:'Agosto',9:'Setembro',10:'Outubro',11:'Novembro',12:'Dezembro'}

# ============================================================
# METODOLOGIA CORRIGIDA:
# Mantemos a geometria REAL do trecho (is_tracado_via_*)
# Alteramos APENAS os fatores CONTEXTUAIS controlaveis:
#   - is_final_semana (0 ou 1)
#   - is_feriado (0 ou 1)
#   - condicao_metereologica_* (chuva vs ceu claro)
#   - fase_dia_* (noite vs dia)
# ============================================================

# Top trechos mais criticos
top_trechos = df.groupby(['br', 'trecho_10km'])[TARGET].agg(
    ['sum', 'mean', 'median', 'count']
).sort_values('sum', ascending=False).head(20)

print("TOP 10 trechos mais criticos:")
print(top_trechos.head(10).to_string())

# ============================================================
# SIMULACAO CORRIGIDA PARA CADA TRECHO CRITICO
# ============================================================
resultados = []

for (br_val, trecho_val), row_stats in top_trechos.iterrows():
    mask = (df['br'] == br_val) & (df['trecho_10km'] == trecho_val)
    df_trecho = df[mask].copy()
    X_trecho = df_trecho.drop(columns=[TARGET])
    y_trecho = df_trecho[TARGET]

    # Predicao REAL (dados originais)
    pred_real = model.predict(X_trecho)

    # TEMPESTADE PERFEITA: Manter geometria real, mudar contexto
    X_pior = X_trecho.copy()
    X_pior['is_final_semana'] = 1
    X_pior['is_feriado'] = 1
    for c in meteo_cols: X_pior[c] = 0
    X_pior[chuva_col] = 1
    for c in fase_cols: X_pior[c] = 0
    X_pior[noite_col] = 1
    # GEOMETRIA PRESERVADA - nao tocamos em is_tracado_via_*

    # CENARIO IDEAL: Manter geometria real, mudar contexto
    X_ideal = X_trecho.copy()
    X_ideal['is_final_semana'] = 0
    X_ideal['is_feriado'] = 0
    for c in meteo_cols: X_ideal[c] = 0
    X_ideal[ceu_claro_col] = 1
    for c in fase_cols: X_ideal[c] = 0
    X_ideal[dia_col] = 1
    # GEOMETRIA PRESERVADA

    pred_pior = model.predict(X_pior)
    pred_ideal = model.predict(X_ideal)

    med_real = np.median(pred_real)
    med_pior = np.median(pred_pior)
    med_ideal = np.median(pred_ideal)
    diff = med_pior - med_ideal
    pct = (diff / med_ideal) * 100 if med_ideal != 0 else 0

    resultados.append({
        'br': int(br_val),
        'trecho': int(trecho_val),
        'acidentes_reais_total': int(y_trecho.sum()),
        'acidentes_reais_media': y_trecho.mean(),
        'acidentes_reais_mediana': y_trecho.median(),
        'registros': len(df_trecho),
        'med_real': med_real,
        'med_pior': med_pior,
        'med_ideal': med_ideal,
        'avg_real': pred_real.mean(),
        'avg_pior': pred_pior.mean(),
        'avg_ideal': pred_ideal.mean(),
        'diff': diff,
        'pct': pct
    })

df_res = pd.DataFrame(resultados).sort_values('diff', ascending=False)

print("\n=== RANKING (Geometria preservada): Tempestade vs Ideal ===")
for _, row in df_res.head(5).iterrows():
    print(f"  BR-{int(row['br']):03d} km {int(row['trecho'])} | "
          f"Ideal: {row['med_ideal']:.1f} | Real: {row['med_real']:.1f} | "
          f"Pior: {row['med_pior']:.1f} | "
          f"Diff: +{row['diff']:.1f} (+{row['pct']:.0f}%) | "
          f"Reais: {int(row['acidentes_reais_total'])}")

# ============================================================
# CASO SHOWCASE: trecho com maior diferenca
# ============================================================
caso = df_res.iloc[0]
br_caso = int(caso['br'])
trecho_caso = int(caso['trecho'])

mask_caso = (df['br'] == br_caso) & (df['trecho_10km'] == trecho_caso)
df_caso = df[mask_caso]
X_caso = df_caso.drop(columns=[TARGET])

# Mes de pico
mes_pior = df_caso.groupby(['ano_base', 'mes'])[TARGET].sum().sort_values(ascending=False)
ano_pico, mes_pico = mes_pior.index[0]
acid_pico = int(mes_pior.iloc[0])
mes_nome = nomes_meses.get(int(mes_pico), str(int(mes_pico)))

# Registro especifico do pico
mask_pico = (df_caso['ano_base'] == ano_pico) & (df_caso['mes'] == mes_pico)
X_pico = X_caso[mask_pico].copy()

# Predicao real
pred_pico_real = model.predict(X_pico)[0]

# Tempestade: manter TUDO do registro real, mudar apenas contexto
X_pico_pior = X_pico.copy()
X_pico_pior['is_final_semana'] = 1
X_pico_pior['is_feriado'] = 1
for c in meteo_cols: X_pico_pior[c] = 0
X_pico_pior[chuva_col] = 1
for c in fase_cols: X_pico_pior[c] = 0
X_pico_pior[noite_col] = 1

# Ideal: manter TUDO do registro real, mudar apenas contexto
X_pico_ideal = X_pico.copy()
X_pico_ideal['is_final_semana'] = 0
X_pico_ideal['is_feriado'] = 0
for c in meteo_cols: X_pico_ideal[c] = 0
X_pico_ideal[ceu_claro_col] = 1
for c in fase_cols: X_pico_ideal[c] = 0
X_pico_ideal[dia_col] = 1

pred_pico_pior = model.predict(X_pico_pior)[0]
pred_pico_ideal = model.predict(X_pico_ideal)[0]

diff_pico = pred_pico_pior - pred_pico_ideal
pct_pico = (diff_pico / pred_pico_ideal) * 100

# Listar geometria real do trecho
registro_real = X_pico.iloc[0]
tracado_ativo = [c.replace('is_tracado_via_', '') for c in tracado_cols if registro_real[c] == 1]

print(f"\n{'='*70}")
print(f"CASO SHOWCASE (CORRIGIDO): BR-{br_caso:03d}, km {trecho_caso}")
print(f"{'='*70}")
print(f"  Geometria REAL preservada: {', '.join(tracado_ativo) if tracado_ativo else 'Nenhuma'}")
print(f"  Periodo: {mes_nome}/{int(ano_pico)}")
print(f"  Acidentes reais: {acid_pico}")
print(f"  Pred. Real:       {pred_pico_real:.1f}")
print(f"  Pred. Ideal:      {pred_pico_ideal:.1f}")
print(f"  Pred. Tempestade: {pred_pico_pior:.1f}")
print(f"  Diferenca:        +{diff_pico:.1f} acidentes (+{pct_pico:.0f}%)")

# ============================================================
# GRAFICO SHOWCASE CORRIGIDO
# ============================================================
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

fig.suptitle(
    f'ESTUDO DE CASO: BR-{br_caso:03d}, Trecho km {trecho_caso}\n'
    f'{mes_nome} de {int(ano_pico)} | Acidentes Reais: {acid_pico}',
    fontsize=16, fontweight='bold', y=0.98
)

# --- Painel 1: Barras (3 cenarios) ---
ax1 = fig.add_subplot(gs[0, 0])
labels = ['Cenario Ideal\n(contextual)', 'Situacao Real\n(dados originais)', 'Tempestade Perfeita\n(contextual)']
vals = [pred_pico_ideal, pred_pico_real, pred_pico_pior]
colors = ['#22c55e', '#3b82f6', '#dc2626']

bars = ax1.bar(labels, vals, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax1.axhline(y=acid_pico, color='#7c3aed', linestyle='--', lw=2,
            label=f'Acidentes Reais: {acid_pico}')
ax1.legend(fontsize=11, loc='upper left')
ax1.set_ylabel('Predicao de Acidentes', fontsize=12)
ax1.set_title('Predicao V5 por Cenario (Geometria Preservada)', fontsize=12, fontweight='bold')

# Anotacao da diferenca
ax1.annotate(
    f'Diferenca: +{diff_pico:.1f} acid.\n(+{pct_pico:.0f}%)',
    xy=(1.0, (pred_pico_pior + pred_pico_ideal)/2),
    fontsize=12, fontweight='bold', ha='center', color='#991b1b',
    bbox=dict(boxstyle='round,pad=0.4', fc='#fee2e2', ec='#dc2626', lw=1.5)
)

# --- Painel 2: Tabela de fatores ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')

geom_str = ', '.join(tracado_ativo) if tracado_ativo else 'N/A'
if len(geom_str) > 30:
    geom_str = geom_str[:28] + '...'

table_data = [
    ['Fator', 'Cenario Ideal', 'Tempestade Perfeita'],
    ['Geometria da Via', f'PRESERVADA', f'PRESERVADA'],
    ['Dia da Semana', 'Dia Util', 'Final de Semana'],
    ['Meteorologia', 'Ceu Claro', 'Chuva'],
    ['Fase do Dia', 'Pleno Dia', 'Plena Noite'],
    ['Feriado', 'Nao', 'Sim'],
    ['', '', ''],
    ['Predicao Modelo', f'{pred_pico_ideal:.1f} acid.', f'{pred_pico_pior:.1f} acid.'],
    ['Acidentes Reais', f'{acid_pico}', f'{acid_pico}'],
    ['Diferenca', '', f'+{diff_pico:.1f} acid. (+{pct_pico:.0f}%)'],
]

cell_colors = []
for i, row in enumerate(table_data):
    if i == 0:
        cell_colors.append(['#1e3a5f', '#166534', '#991b1b'])
    elif i == 1:
        cell_colors.append(['#f8fafc', '#eff6ff', '#eff6ff'])  # Preservada = azul claro
    elif i == 6:
        cell_colors.append(['white', 'white', 'white'])
    elif i in [7, 8, 9]:
        cell_colors.append(['#f0f9ff', '#dcfce7', '#fee2e2'])
    else:
        cell_colors.append(['#f8fafc', '#f0fdf4', '#fef2f2'])

table = ax2.table(cellText=table_data, cellColours=cell_colors,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.7)

for j in range(3):
    table[0, j].set_text_props(color='white', fontweight='bold')
for i in [1, 7, 8, 9]:
    for j in range(3):
        table[i, j].set_text_props(fontweight='bold')

ax2.set_title(f'Fatores Alterados (Geometria: {geom_str})', fontsize=12, fontweight='bold', pad=20)

# --- Painel 3: Historico temporal ---
ax3 = fig.add_subplot(gs[1, 0])

hist_data = df_caso.groupby(['ano_base', 'mes'])[TARGET].sum().reset_index()
hist_data['periodo'] = hist_data['ano_base'].astype(str) + '/' + hist_data['mes'].apply(lambda x: f'{int(x):02d}')
hist_data = hist_data.sort_values(['ano_base', 'mes']).reset_index(drop=True)

if len(hist_data) > 24:
    hist_data = hist_data.tail(24).reset_index(drop=True)

bar_colors = ['#3b82f6'] * len(hist_data)
pico_pos = None
for i, row in hist_data.iterrows():
    if row['ano_base'] == ano_pico and row['mes'] == mes_pico:
        bar_colors[i] = '#dc2626'
        pico_pos = i

ax3.bar(range(len(hist_data)), hist_data[TARGET], color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.3)
ax3.axhline(y=caso['acidentes_reais_mediana'], color='#ef4444', linestyle='--', lw=2,
            label=f'Mediana historica: {caso["acidentes_reais_mediana"]:.0f}')

if pico_pos is not None:
    ax3.annotate(f'PICO: {acid_pico}',
                 xy=(pico_pos, acid_pico), xytext=(pico_pos, acid_pico + 3),
                 ha='center', fontsize=10, fontweight='bold', color='#dc2626')

tick_step = max(1, len(hist_data)//8)
tick_positions = range(0, len(hist_data), tick_step)
ax3.set_xticks(list(tick_positions))
ax3.set_xticklabels([hist_data.iloc[i]['periodo'] for i in tick_positions], rotation=45, fontsize=9)
ax3.set_ylabel('Acidentes Reais', fontsize=12)
ax3.set_title(f'Historico Mensal: BR-{br_caso:03d} km {trecho_caso}', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)

# --- Painel 4: Resumo ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

resumo = (
    f"RESUMO EXECUTIVO\n"
    f"{'='*48}\n\n"
    f"Rodovia:   BR-{br_caso:03d}\n"
    f"Trecho:    km {trecho_caso} (10 km)\n"
    f"Geometria: {', '.join(tracado_ativo)}\n"
    f"Periodo:   {mes_nome} de {int(ano_pico)}\n\n"
    f"Acidentes reais:    {acid_pico}\n"
    f"Predicao (real):    {pred_pico_real:.1f}\n\n"
    f"{'='*48}\n"
    f"CENARIO IDEAL (fatores contextuais):\n"
    f"  Dia util + Ceu claro + Pleno dia\n"
    f"  -> {pred_pico_ideal:.1f} acidentes\n\n"
    f"TEMPESTADE PERFEITA:\n"
    f"  FDS + Chuva + Noite + Feriado\n"
    f"  -> {pred_pico_pior:.1f} acidentes\n\n"
    f"{'='*48}\n"
    f"IMPACTO CONTEXTUAL:\n"
    f"  +{diff_pico:.1f} acidentes (+{pct_pico:.0f}%)\n\n"
    f"A geometria do trecho e FIXA.\n"
    f"Os fatores contextuais (clima,\n"
    f"horario, dia) sao CONTROLAVEIS\n"
    f"pela PRF e DNIT."
)

ax4.text(0.05, 0.95, resumo, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.8', fc='#f8fafc', ec='#64748b', lw=1.5))

plt.savefig(os.path.join(OUT_DIR, 'cenario_05_caso_real.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n-> Grafico salvo: cenario_05_caso_real.png")

# ============================================================
# TOP 5 TRECHOS CORRIGIDO
# ============================================================
fig, ax = plt.subplots(figsize=(16, 8))

top5 = df_res.head(5).reset_index(drop=True)

x = np.arange(len(top5))
width = 0.25

bars_ideal = ax.bar(x - width, top5['med_ideal'], width, label='Cenario Ideal (contexto favoravel)',
                    color='#22c55e', edgecolor='black', linewidth=0.6)
bars_real = ax.bar(x, top5['med_real'], width, label='Situacao Real (dados originais)',
                   color='#3b82f6', edgecolor='black', linewidth=0.6)
bars_pior = ax.bar(x + width, top5['med_pior'], width, label='Tempestade Perfeita (pior contexto)',
                   color='#dc2626', edgecolor='black', linewidth=0.6)

for i, (_, row) in enumerate(top5.iterrows()):
    ax.text(i - width, row['med_ideal'] + 0.3, f'{row["med_ideal"]:.1f}',
            ha='center', fontsize=9, fontweight='bold', color='#166534')
    ax.text(i, row['med_real'] + 0.3, f'{row["med_real"]:.1f}',
            ha='center', fontsize=9, fontweight='bold', color='#1e3a5f')
    ax.text(i + width, row['med_pior'] + 0.3, f'{row["med_pior"]:.1f}',
            ha='center', fontsize=9, fontweight='bold', color='#991b1b')

    ax.annotate(f'+{row["diff"]:.1f}\n(+{row["pct"]:.0f}%)',
                xy=(i, max(row['med_pior'], row['med_real']) + 1.5),
                ha='center', fontsize=9, fontweight='bold', color='#7c2d12',
                bbox=dict(boxstyle='round,pad=0.3', fc='#fff7ed', ec='#f97316', lw=1))

labels_x = [f'BR-{int(row["br"]):03d}\nkm {int(row["trecho"])}\n({int(row["acidentes_reais_total"])} acid.)'
            for _, row in top5.iterrows()]
ax.set_xticks(x)
ax.set_xticklabels(labels_x, fontsize=10)
ax.set_ylabel('Mediana Predita de Acidentes', fontsize=13, fontweight='bold')
ax.set_title('Top 5 Trechos Criticos: Impacto dos Fatores Contextuais\n(Geometria Real Preservada - Modelo V5)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cenario_05_top5_trechos.png'), dpi=150)
plt.close()
print("-> Grafico salvo: cenario_05_top5_trechos.png")

# ============================================================
# ATUALIZAR RELATORIO TXT - Substituir secao do cenario 5
# ============================================================
# Ler o relatorio existente e remover a secao antiga do caso real se existir
relatorio_path = os.path.join(OUT_DIR, 'relatorio_cenarios_showcase.txt')
with open(relatorio_path, 'r', encoding='utf-8') as f:
    conteudo = f.read()

# Remover secao antiga se existir
marker = "CENARIO 5 - APROFUNDAMENTO: ESTUDO DE CASO REAL"
if marker in conteudo:
    idx = conteudo.index("=" * 110 + "\n" + marker)
    conteudo = conteudo[:idx].rstrip() + "\n"

with open(relatorio_path, 'w', encoding='utf-8') as f:
    f.write(conteudo)
    f.write("\n\n")
    f.write("=" * 110 + "\n")
    f.write("CENARIO 5 - APROFUNDAMENTO: ESTUDO DE CASO REAL (METODOLOGIA CORRIGIDA)\n")
    f.write("=" * 110 + "\n\n")

    f.write("METODOLOGIA: A geometria real do trecho (is_tracado_via_*) e PRESERVADA.\n")
    f.write("Apenas os fatores CONTEXTUAIS controlaveis sao alterados:\n")
    f.write("  - Dia da semana (util vs final de semana)\n")
    f.write("  - Feriado (sim vs nao)\n")
    f.write("  - Meteorologia (ceu claro vs chuva)\n")
    f.write("  - Fase do dia (pleno dia vs plena noite)\n\n")
    f.write("Justificativa: A geometria da via e uma propriedade FISICA do trecho que nao\n")
    f.write("pode ser alterada por politica publica de curto prazo. Zerar essas variaveis\n")
    f.write("distorceria completamente a predicao do modelo.\n\n")

    f.write("-" * 110 + "\n")
    f.write(f"TRECHO ANALISADO: BR-{br_caso:03d}, km {trecho_caso}\n")
    f.write(f"GEOMETRIA REAL:   {', '.join(tracado_ativo)}\n")
    f.write(f"PERIODO DE PICO:  {mes_nome} de {int(ano_pico)}\n")
    f.write(f"ACIDENTES REAIS:  {acid_pico} acidentes neste mes\n")
    f.write(f"HISTORICO TOTAL:  {int(caso['acidentes_reais_total'])} acidentes acumulados\n\n")

    f.write("-" * 110 + "\n")
    f.write("COMPOSICAO DOS CENARIOS (alteracoes contextuais apenas):\n")
    f.write("-" * 110 + "\n")
    f.write(f"{'Fator':<25s} | {'CENARIO IDEAL':<25s} | {'TEMPESTADE PERFEITA':<25s}\n")
    f.write("-" * 110 + "\n")
    f.write(f"{'Geometria da Via':<25s} | {'PRESERVADA (real)':<25s} | {'PRESERVADA (real)':<25s}\n")
    f.write(f"{'Dia da Semana':<25s} | {'Dia Util':<25s} | {'Final de Semana':<25s}\n")
    f.write(f"{'Meteorologia':<25s} | {'Ceu Claro':<25s} | {'Chuva':<25s}\n")
    f.write(f"{'Fase do Dia':<25s} | {'Pleno Dia':<25s} | {'Plena Noite':<25s}\n")
    f.write(f"{'Feriado':<25s} | {'Nao':<25s} | {'Sim':<25s}\n")
    f.write("-" * 110 + "\n\n")

    f.write("RESULTADO DA SIMULACAO:\n")
    f.write("-" * 110 + "\n")
    f.write(f"  Predicao Cenario Ideal:            {pred_pico_ideal:.1f} acidentes\n")
    f.write(f"  Predicao Situacao Real:            {pred_pico_real:.1f} acidentes\n")
    f.write(f"  Predicao Tempestade Perfeita:      {pred_pico_pior:.1f} acidentes\n")
    f.write(f"  Acidentes reais registrados (PRF): {acid_pico} acidentes\n\n")
    f.write(f"  IMPACTO CONTEXTUAL: +{diff_pico:.1f} acidentes (+{pct_pico:.0f}%)\n\n")

    f.write("INTERPRETACAO:\n")
    f.write("-" * 110 + "\n")
    f.write(f"Na BR-{br_caso:03d}, trecho km {trecho_caso} (geometria: {', '.join(tracado_ativo)}),\n")
    f.write("o modelo V5 demonstra que APENAS alterando fatores contextuais (sem mudar a via),\n")
    f.write(f"a sinistralidade prevista varia em {pct_pico:+.0f}%. Isso comprova que:\n\n")
    f.write("  -> A PRF pode REDUZIR acidentes priorizando fiscalizacao em:\n")
    f.write("     * Finais de semana e feriados\n")
    f.write("     * Periodos noturnos\n")
    f.write("     * Dias de chuva\n\n")
    f.write("  -> O DNIT pode investir em:\n")
    f.write("     * Iluminacao para reduzir o efeito noturno\n")
    f.write("     * Drenagem para reduzir risco em chuvas\n\n")

    f.write("=" * 110 + "\n")
    f.write("TOP 5 TRECHOS CRITICOS (Geometria preservada, fatores contextuais alterados)\n")
    f.write("=" * 110 + "\n\n")
    f.write(f"{'#':<3s} | {'BR':<8s} | {'km':<6s} | {'Md Ideal':>10s} | {'Md Real':>10s} | {'Md Pior':>10s} | {'Diff':>8s} | {'%':>7s} | {'Acid.Hist':>10s}\n")
    f.write("-" * 110 + "\n")
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        f.write(f"{i:<3d} | BR-{int(row['br']):03d}  | {int(row['trecho']):<6d} | "
                f"{row['med_ideal']:>10.1f} | {row['med_real']:>10.1f} | {row['med_pior']:>10.1f} | "
                f"+{row['diff']:>7.1f} | +{row['pct']:>5.0f}% | {int(row['acidentes_reais_total']):>10,d}\n")
    f.write("-" * 110 + "\n")
    f.write("  Graficos: cenario_05_caso_real.png | cenario_05_top5_trechos.png\n")
    f.write("=" * 110 + "\n")

print("\n[OK] Cenario 5 CORRIGIDO com geometria preservada!")
