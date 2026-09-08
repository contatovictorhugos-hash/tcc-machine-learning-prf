"""
06_Modelagem/model_report_v6.py
===============================
Geração de Diagnósticos de Resíduos e Previsão de Zeros para a Arquitetura V6.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def run_diagnostics_v6():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "06_Modelagem", "random_forest_v6_tcc.pkl")
    feat_path = os.path.join(base_dir, "06_Modelagem", "v6_feature_names.pkl")
    data_path = os.path.join(base_dir, "05_Pre_Modelagem", "df_model_ready_v6.csv")
    out_dir = os.path.join(base_dir, "06_Modelagem", "graficos_modelo_v6")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print("  DIAGNÓSTICO E DISPERSÃO DE RESÍDUOS — V6 (TCC 2026)")
    print("=" * 70)
    
    model = joblib.load(model_path)
    feature_names = joblib.load(feat_path)
    df = pd.read_csv(data_path, low_memory=False)
    
    # Test set (2024 e 2025)
    df_test = df[df['ano'].isin([2024, 2025])].copy()
    y_test = df_test['quantidade_acidentes'].values
    
    cat_cols = ['estacao_ano', 'uf']
    cols_to_drop = ['quantidade_acidentes', 'ano', 'mes', 'br', 'trecho_10km',
                    'total_mortos', 'total_feridos_graves', 'total_veiculos', 'is_zero_real']
    
    df_test_enc = pd.get_dummies(df_test.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False)
    df_test_enc = df_test_enc.reindex(columns=feature_names, fill_value=0).astype(float)
    X_test = df_test_enc.values
    
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # -------------------------------------------------------------
    # Gráfico 1: Previsto vs Real V6
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 7))
    plt.scatter(y_test, y_pred, alpha=0.15, s=8, color='#1d4ed8', edgecolors='none', label='Trechos/Meses (2024-25)')
    max_val = max(np.percentile(y_test, 99.9), np.percentile(y_pred, 99.9))
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Previsão Perfeita')
    plt.title(f'Previsto vs. Real — Arquitetura V6 (Poisson Oficial)\nMAE = {mae:.3f} | R² = {r2:.4f}', fontsize=13, fontweight='bold')
    plt.xlabel('Quantidade Real de Acidentes (Y)', fontsize=11)
    plt.ylabel('Quantidade Prevista pelo Modelo (Ŷ)', fontsize=11)
    plt.xlim(-0.5, max_val + 2)
    plt.ylim(-0.5, max_val + 2)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    p1 = os.path.join(out_dir, '01_previsto_vs_real_v6.png')
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"Salvo: {p1}")
    
    # -------------------------------------------------------------
    # Gráfico 2: Distribuição de Resíduos
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.hist(residuals, bins=60, range=(-6, 6), color='#0284c7', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Resíduo Zero')
    plt.title('Distribuição de Resíduos (Y - Ŷ) — Modelo V6', fontsize=13, fontweight='bold')
    plt.xlabel('Erro Residual (Acidentes)', fontsize=11)
    plt.ylabel('Frequência de Observações', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    p2 = os.path.join(out_dir, '02_distribuicao_residuos_v6.png')
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"Salvo: {p2}")
    
    # -------------------------------------------------------------
    # Gráfico 3: Desempenho por Faixa (Capacidade de Prever Zeros)
    # -------------------------------------------------------------
    zero_mask = (y_test == 0)
    mae_zeros = mean_absolute_error(y_test[zero_mask], y_pred[zero_mask])
    pred_zeros_mean = y_pred[zero_mask].mean()
    pred_pos_mean = y_pred[~zero_mask].mean()
    real_pos_mean = y_test[~zero_mask].mean()
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Trechos Seguros\n(Real = 0)', 'Trechos com Sinistros\n(Média Real = {:.1f})'.format(real_pos_mean)],
                   [pred_zeros_mean, pred_pos_mean], color=['#22c55e', '#ef4444'], edgecolor='black', width=0.45)
    plt.ylabel('Média Prevista pelo Modelo V6', fontsize=11)
    plt.title('Capacidade de Discriminação de Zeros Reais — Modelo V6', fontsize=13, fontweight='bold')
    plt.ylim(0, pred_pos_mean * 1.35)
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.08, f'{b.get_height():.2f}', ha='center', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    p3 = os.path.join(out_dir, '03_discriminacao_zeros_v6.png')
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f"Salvo: {p3}")
    print("=" * 70)

if __name__ == "__main__":
    run_diagnostics_v6()
