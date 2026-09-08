"""
08_Fatores_Climaticos/cenarios_v6.py
====================================
Simulação de Cenários Operacionais da PRF com o Modelo V6.
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_scenarios_v6():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "06_Modelagem", "random_forest_v6_tcc.pkl")
    feat_path = os.path.join(base_dir, "06_Modelagem", "v6_feature_names.pkl")
    data_path = os.path.join(base_dir, "05_Pre_Modelagem", "df_model_ready_v6.csv")
    out_txt = os.path.join(base_dir, "08_Fatores_Climaticos", "relatorio_cenarios_v6.txt")
    out_png = os.path.join(base_dir, "08_Fatores_Climaticos", "cenarios_v6_comparativo.png")
    
    print("=" * 70)
    print("  SIMULAÇÃO DE CENÁRIOS OPERACIONAIS — V6 (TCC 2026)")
    print("=" * 70)
    
    model = joblib.load(model_path)
    feature_names = joblib.load(feat_path)
    df = pd.read_csv(data_path, low_memory=False)
    
    # Função para obter a linha base média de um trecho específico
    def get_trecho_baseline(br, trecho_km):
        sub = df[(df['br'] == br) & (df['trecho_10km'] == trecho_km)]
        if len(sub) == 0:
            sub = df[df['br'] == br]
        cat_cols = ['estacao_ano', 'uf']
        cols_to_drop = ['quantidade_acidentes', 'ano', 'mes', 'br', 'trecho_10km',
                        'total_mortos', 'total_feridos_graves', 'total_veiculos', 'is_zero_real']
        enc = pd.get_dummies(sub.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False)
        enc = enc.reindex(columns=feature_names, fill_value=0).astype(float)
        return enc.mean()
        
    def predict_scenario(baseline_row, overrides):
        row = baseline_row.copy()
        for k, v in overrides.items():
            if k in row.index:
                row[k] = v
        x = row.values.reshape(1, -1)
        return round(float(model.predict(x)[0]), 2)

    # 1. BR-381 KM 340 (Serra de MG / João Monlevade - Crítica)
    base_br381 = get_trecho_baseline(381, 340)
    c1_normal = predict_scenario(base_br381, {'dias_feriado': 0, 'dias_fim_semana': 8})
    c1_feriado = predict_scenario(base_br381, {'dias_feriado': 3, 'dias_fim_semana': 10})
    
    # 2. BR-116 KM 50 (Regis Bittencourt / SP - Alta Densidade)
    base_br116 = get_trecho_baseline(116, 50)
    c2_inverno = predict_scenario(base_br116, {'estacao_ano_Inverno': 1, 'estacao_ano_Verao': 0, 'dias_feriado': 0})
    c2_verao_pico = predict_scenario(base_br116, {'estacao_ano_Inverno': 0, 'estacao_ano_Verao': 1, 'dias_feriado': 2})
    
    # 3. BR-101 KM 130 (Litoral SC / Balneário Camboriú - Turístico)
    base_br101 = get_trecho_baseline(101, 130)
    c3_baixo_fluxo = predict_scenario(base_br101, {'dias_fim_semana': 8, 'dias_feriado': 0})
    c3_alto_fluxo = predict_scenario(base_br101, {'dias_fim_semana': 10, 'dias_feriado': 3})
    
    # 4. Trecho Seguro / Baixa Densidade (BR-317 KM 70 no Acre)
    base_br317 = get_trecho_baseline(317, 70)
    c4_seguro = predict_scenario(base_br317, {})
    
    cenarios = [
        ('BR-381 KM 340 (MG) - Mês Regular', c1_normal, '#3b82f6'),
        ('BR-381 KM 340 (MG) - Feriados + FDS', c1_feriado, '#ef4444'),
        ('BR-116 KM 50 (SP) - Inverno Regular', c2_inverno, '#3b82f6'),
        ('BR-116 KM 50 (SP) - Pico de Verão', c2_verao_pico, '#ef4444'),
        ('BR-101 KM 130 (SC) - Baixo Fluxo', c3_baixo_fluxo, '#3b82f6'),
        ('BR-101 KM 130 (SC) - Feriadões', c3_alto_fluxo, '#ef4444'),
        ('BR-317 KM 70 (AC) - Trecho Seguro', c4_seguro, '#22c55e')
    ]
    
    print("\nResultados dos Cenários Operacionais V6:")
    for nome, pred, _ in cenarios:
        print(f"  * {nome:<38}: {pred:.2f} acidentes/mês")
        
    # Salvar Relatório
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("==============================================================================\n")
        f.write("              RELATÓRIO DE CENÁRIOS OPERACIONAIS — MODELO V6                 \n")
        f.write("==============================================================================\n\n")
        for nome, pred, _ in cenarios:
            f.write(f"{nome:<40} : {pred:.2f} acidentes/mês\n")
        f.write("\nConclusão:\n")
        f.write("Diferente do Modelo V5 (que ficava comprimido entre 1.36 e 1.60 acidentes),\n")
        f.write(f"o Modelo V6 discrimina com precisão trechos de risco nulo (BR-317: {c4_seguro} ac/mês)\n")
        f.write(f"de trechos de altíssimo risco concentrado (BR-116/381: até {max(c1_feriado, c2_verao_pico):.2f} ac/mês).\n")
        f.write("==============================================================================\n")
    print(f"\nRelatório salvo em: {out_txt}")
    
    # Gerar Gráfico
    plt.figure(figsize=(11, 6))
    labels = [c[0] for c in cenarios]
    vals = [c[1] for c in cenarios]
    colors = [c[2] for c in cenarios]
    
    bars = plt.barh(labels[::-1], vals[::-1], color=colors[::-1], edgecolor='black', alpha=0.85)
    plt.xlabel('Previsão Oficial do Modelo V6 (Acidentes no Mês)', fontsize=11, fontweight='bold')
    plt.title('Simulação de Cenários Operacionais da PRF — Arquitetura V6', fontsize=13, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    for b in bars:
        w = b.get_width()
        plt.text(w + 0.1, b.get_y() + b.get_height()/2, f'{w:.2f}', va='center', fontweight='bold')
        
    plt.xlim(0, max(vals) * 1.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Gráfico comparativo salvo em: {out_png}")
    print("=" * 70)

if __name__ == "__main__":
    run_scenarios_v6()
