import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ============================================================
# CONFIGURAÇÃO
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

sns.set_theme(style="whitegrid", font_scale=1.1)

print(f"Carregando dados de: {DATA_PATH}")
print(f"Carregando modelo de: {MODEL_PATH}")
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

TARGET = 'quantidade_acidentes'
y = df[TARGET]
X = df.drop(columns=[TARGET])

# ============================================================
# 1. ANÁLISE DE IMPACTO GERAL: Variando o ICP de 0 a 100
# ============================================================
print("\n[1/3] Avaliando impacto geral do ICP na variável alvo...")

icp_values = np.arange(0, 105, 5)  # 0, 5, 10, ..., 100
mean_predictions = []

for icp_val in icp_values:
    X_sim = X.copy()
    X_sim['icp'] = icp_val
    preds = model.predict(X_sim)
    mean_predictions.append(preds.mean())

df_impacto = pd.DataFrame({
    'ICP': icp_values,
    'Media_Pred_Acidentes': mean_predictions
})

# Calcular a redução percentual em relação ao ICP=0
baseline_pred = df_impacto.loc[df_impacto['ICP'] == 0, 'Media_Pred_Acidentes'].values[0]
df_impacto['Reducao_Pct'] = ((baseline_pred - df_impacto['Media_Pred_Acidentes']) / baseline_pred) * 100

print(df_impacto.to_string(index=False))

# --- Gráfico 1: Curva de Impacto Geral ---
fig, ax1 = plt.subplots(figsize=(12, 6))

color_bar = '#3b82f6'
color_line = '#ef4444'

ax1.bar(df_impacto['ICP'], df_impacto['Media_Pred_Acidentes'],
        width=4, color=color_bar, alpha=0.7, label='Média Predita de Acidentes')
ax1.set_xlabel('Índice de Condição do Pavimento (ICP)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Média Predita de Acidentes (quantidade)', color=color_bar, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color_bar)

ax2 = ax1.twinx()
ax2.plot(df_impacto['ICP'], df_impacto['Reducao_Pct'], color=color_line,
         linewidth=2.5, marker='o', markersize=5, label='Redução % vs ICP=0')
ax2.set_ylabel('Redução Percentual vs ICP=0 (%)', color=color_line, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color_line)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

plt.title('Impacto do ICP na Predição Média de Acidentes (Modelo V5)', fontsize=14, fontweight='bold')
fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.95), fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}\\impacto_geral_icp_v5.png', dpi=150)
plt.close()
print("   -> Gráfico salvo: impacto_geral_icp_v5.png")

# ============================================================
# 2. CENÁRIO DE MAIOR REDUÇÃO: ICP de 5 para 70 (por registro)
# ============================================================
print("\n[2/3] Avaliando cenário de maior redução individual (ICP 5 -> 70)...")

# Filtrar apenas registros que atualmente têm ICP == 5
mask_icp5 = X['icp'] == 5
X_icp5 = X[mask_icp5].copy()
y_icp5 = y[mask_icp5].copy()
print(f"   Registros com ICP = 5: {len(X_icp5):,}")

if len(X_icp5) == 0:
    # Se não houver registros com ICP exatamente 5, usar os mais próximos (4-6)
    mask_icp5 = (X['icp'] >= 4) & (X['icp'] <= 6)
    X_icp5 = X[mask_icp5].copy()
    y_icp5 = y[mask_icp5].copy()
    print(f"   (Expandido para ICP entre 4 e 6): {len(X_icp5):,}")

# Predição com ICP = 5 (cenário atual)
X_cenario_atual = X_icp5.copy()
X_cenario_atual['icp'] = 5
pred_atual = model.predict(X_cenario_atual)

# Predição com ICP = 70 (cenário melhorado)
X_cenario_melhor = X_icp5.copy()
X_cenario_melhor['icp'] = 70
pred_melhor = model.predict(X_cenario_melhor)

# Calcular reduções
df_reducao = pd.DataFrame({
    'idx_original': X_icp5.index,
    'br': X_icp5['br'].values,
    'trecho_10km': X_icp5['trecho_10km'].values,
    'ano_base': X_icp5['ano_base'].values,
    'mes': X_icp5['mes'].values,
    'pred_icp5': pred_atual,
    'pred_icp70': pred_melhor,
    'reducao_absoluta': pred_atual - pred_melhor,
    'reducao_pct': ((pred_atual - pred_melhor) / pred_atual) * 100,
    'acidentes_real': y_icp5.values
})

# Ordenar pela maior redução absoluta
df_reducao = df_reducao.sort_values('reducao_absoluta', ascending=False)

# Cenário de MAIOR redução
maior = df_reducao.iloc[0]
print(f"\n   CENÁRIO DE MAIOR REDUÇÃO:")
print(f"   BR: {maior['br']:.0f} | Trecho: {maior['trecho_10km']:.0f} | Ano: {maior['ano_base']:.0f} | Mês: {maior['mes']:.0f}")
print(f"   Predição ICP=5:  {maior['pred_icp5']:.2f} acidentes")
print(f"   Predição ICP=70: {maior['pred_icp70']:.2f} acidentes")
print(f"   Redução Absoluta: {maior['reducao_absoluta']:.2f} acidentes")
print(f"   Redução Percentual: {maior['reducao_pct']:.1f}%")
print(f"   Acidentes Reais: {maior['acidentes_real']:.0f}")

# --- Gráfico 2: Cenário de Maior Redução (Before vs After) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 2a: Barras comparativas do cenário de maior redução
labels = ['ICP = 5\n(Atual)', 'ICP = 70\n(Melhorado)']
vals = [maior['pred_icp5'], maior['pred_icp70']]
colors = ['#ef4444', '#22c55e']

bars = axes[0].bar(labels, vals, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
axes[0].set_ylabel('Predição de Acidentes', fontsize=12)
axes[0].set_title(
    f'Maior Redução: BR-{maior["br"]:.0f} Trecho {maior["trecho_10km"]:.0f}\n'
    f'Ano {maior["ano_base"]:.0f} Mês {maior["mes"]:.0f}',
    fontsize=13, fontweight='bold'
)
for bar, val in zip(bars, vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Anotação da redução
axes[0].annotate(
    f'Redução: {maior["reducao_absoluta"]:.2f} acidentes\n({maior["reducao_pct"]:.1f}%)',
    xy=(0.5, max(vals) * 0.5), fontsize=12, fontweight='bold',
    ha='center', color='#1e3a5f',
    bbox=dict(boxstyle='round,pad=0.4', fc='#dbeafe', ec='#3b82f6', lw=1.5)
)

# Gráfico 2b: Distribuição das reduções de todos os registros ICP=5
axes[1].hist(df_reducao['reducao_absoluta'], bins=40, color='#8b5cf6', alpha=0.8, edgecolor='black', linewidth=0.5)
axes[1].axvline(df_reducao['reducao_absoluta'].mean(), color='#ef4444', linestyle='--', linewidth=2,
                label=f'Média: {df_reducao["reducao_absoluta"].mean():.2f}')
axes[1].axvline(maior['reducao_absoluta'], color='#22c55e', linestyle='--', linewidth=2,
                label=f'Máximo: {maior["reducao_absoluta"]:.2f}')
axes[1].set_xlabel('Redução Absoluta de Acidentes (ICP 5→70)', fontsize=12)
axes[1].set_ylabel('Frequência', fontsize=12)
axes[1].set_title('Distribuição das Reduções (Todos os Registros com ICP=5)', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}\\maior_reducao_icp5_70_v5.png', dpi=150)
plt.close()
print("   -> Gráfico salvo: maior_reducao_icp5_70_v5.png")

# ============================================================
# 3. TOP 5 CENÁRIOS COM MAIOR REDUÇÃO (ICP 5 → 70)
# ============================================================
print("\n[3/3] Top 5 cenários com maior redução de acidentes (ICP 5 -> 70)...")

top5 = df_reducao.head(5).reset_index(drop=True)
top5.index = top5.index + 1  # Ranking começa em 1

print("\n   TOP 5 CENÁRIOS DE REDUÇÃO:")
print("   " + "="*90)
for i, row in top5.iterrows():
    print(f"   #{i} | BR-{row['br']:.0f} Trecho {row['trecho_10km']:.0f} "
          f"| {row['ano_base']:.0f}/{row['mes']:02.0f} "
          f"| ICP5: {row['pred_icp5']:.2f} -> ICP70: {row['pred_icp70']:.2f} "
          f"| Redução: {row['reducao_absoluta']:.2f} ({row['reducao_pct']:.1f}%) "
          f"| Real: {row['acidentes_real']:.0f}")
print("   " + "="*90)

# --- Gráfico 3: Top 5 Cenários (Barras agrupadas) ---
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(top5))
width = 0.35

bars1 = ax.bar(x - width/2, top5['pred_icp5'], width,
               label='ICP = 5 (Atual)', color='#ef4444', edgecolor='black', linewidth=0.6)
bars2 = ax.bar(x + width/2, top5['pred_icp70'], width,
               label='ICP = 70 (Melhorado)', color='#22c55e', edgecolor='black', linewidth=0.6)

# Labels
ax.set_ylabel('Predição de Acidentes', fontsize=13, fontweight='bold')
ax.set_title('Top 5 Cenários com Maior Redução de Acidentes\n(ICP 5 → 70, Modelo V5)', fontsize=14, fontweight='bold')

labels_x = [f'#{i}\nBR-{row["br"]:.0f}\nTrecho {row["trecho_10km"]:.0f}\n{row["ano_base"]:.0f}/{row["mes"]:02.0f}'
            for i, row in top5.iterrows()]
ax.set_xticks(x)
ax.set_xticklabels(labels_x, fontsize=10)

# Adicionar os valores de redução entre as barras
for i_idx, (b1, b2, (_, row)) in enumerate(zip(bars1, bars2, top5.iterrows())):
    # Valor sobre cada barra
    ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.1,
            f'{row["pred_icp5"]:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#7f1d1d')
    ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.1,
            f'{row["pred_icp70"]:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#14532d')

    # Anotação de redução
    mid_x = (b1.get_x() + b1.get_width() + b2.get_x()) / 2
    mid_y = max(row['pred_icp5'], row['pred_icp70']) + 1.2
    ax.annotate(f'↓ {row["reducao_absoluta"]:.1f}\n({row["reducao_pct"]:.0f}%)',
                xy=(mid_x, mid_y), ha='center', fontsize=9, fontweight='bold', color='#1e40af',
                bbox=dict(boxstyle='round,pad=0.3', fc='#eff6ff', ec='#93c5fd', lw=1))

ax.legend(fontsize=12, loc='upper right')
ax.set_ylim(0, top5['pred_icp5'].max() * 1.4)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}\\top5_reducao_icp5_70_v5.png', dpi=150)
plt.close()
print("   -> Gráfico salvo: top5_reducao_icp5_70_v5.png")

# ============================================================
# RESUMO ESTATÍSTICO GERAL
# ============================================================
print("\n" + "="*70)
print("RESUMO DA ANÁLISE DE IMPACTO DO ICP (Modelo V5)")
print("="*70)
print(f"Total de registros com ICP=5:       {len(df_reducao):,}")
print(f"Média de redução absoluta (5->70):   {df_reducao['reducao_absoluta'].mean():.4f} acidentes")
print(f"Média de redução percentual (5->70): {df_reducao['reducao_pct'].mean():.2f}%")
print(f"Mediana da redução absoluta:        {df_reducao['reducao_absoluta'].median():.4f} acidentes")
print(f"Desvio padrão da redução:           {df_reducao['reducao_absoluta'].std():.4f}")
print(f"Máxima redução absoluta:            {df_reducao['reducao_absoluta'].max():.4f} acidentes")
print(f"Registros com redução > 0:          {(df_reducao['reducao_absoluta'] > 0).sum():,} ({(df_reducao['reducao_absoluta'] > 0).mean()*100:.1f}%)")
print("="*70)

# Salvar tabela top 5 em CSV
top5_out = top5[['br', 'trecho_10km', 'ano_base', 'mes', 'pred_icp5', 'pred_icp70',
                 'reducao_absoluta', 'reducao_pct', 'acidentes_real']]
top5_out.to_csv(f'{OUT_DIR}\\top5_reducao_icp_v5.csv', index=False, encoding='utf-8-sig')
print("\nTabela Top 5 salva em: top5_reducao_icp_v5.csv")

print("\n[OK] Análise de impacto do ICP concluída com sucesso!")
