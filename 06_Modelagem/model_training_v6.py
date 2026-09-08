"""
06_Modelagem/model_training_v6.py
=================================
Treinamento Oficial da Arquitetura V6 (Random Forest Poisson + Estudo de Ablação).

Especificações Metodológicas:
- Dataset: 'df_model_ready_v6.csv' (475.020 observações, 70.61% de zeros reais).
- Split Temporal Estrito:
    * 2019: Warm-up dos Lags temporais.
    * Treino: 2020 a 2023 (271.440 observações).
    * Teste Out-of-Time: 2024 a 2025 (135.720 observações).
- Modelos Avaliados:
    1. V6-Full (Poisson): Lags + DNIT + Calendário + UF (Modelo Oficial).
    2. V6-MSE (Squared Error): Benchmark comparativo direto com V1-V5.
    3. V6-Ablation (Infraestrutura Pura): Modelo sem lags temporais.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, d2_tweedie_score

def find_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base_dir, "05_Pre_Modelagem", "df_model_ready_v6.csv"),
        os.path.join(base_dir, "df_model_ready_v6.csv")
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

def execute_training_v6():
    print("=" * 80)
    print("  TREINAMENTO OFICIAL ARQUITETURA V6 — RANDOM FOREST (TCC 2026)")
    print("=" * 80)
    
    data_path = find_dataset()
    print(f"Carregando dataset de: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERRO: Dataset não encontrado em {data_path}")
        return None
        
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Shape do Dataset: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    
    # 1. Partição Temporal
    print("\nAplicando Split Temporal (Treino 2020-2023 / Teste 2024-2025)...")
    # 2019 usado apenas para compor os lags
    df_train = df[df['ano'].isin([2020, 2021, 2022, 2023])].copy()
    df_test = df[df['ano'].isin([2024, 2025])].copy()
    
    print(f"Treino (2020-2023) : {len(df_train):,} observações ({len(df_train)/len(df)*100:.1f}%)")
    print(f"Teste  (2024-2025) : {len(df_test):,} observações ({len(df_test)/len(df)*100:.1f}%)")
    
    target_col = 'quantidade_acidentes'
    y_train = df_train[target_col].values
    y_test = df_test[target_col].values
    
    # Colunas auxiliares a remover de X
    cols_to_drop = [
        target_col, 'ano', 'mes', 'br', 'trecho_10km',
        'total_mortos', 'total_feridos_graves', 'total_veiculos', 'is_zero_real'
    ]
    
    # One-Hot Encoding em estacao_ano e uf
    cat_cols = ['estacao_ano', 'uf']
    df_train_encoded = pd.get_dummies(df_train.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False)
    df_test_encoded = pd.get_dummies(df_test.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False)
    
    # Alinhar colunas entre treino e teste
    all_feature_cols = df_train_encoded.columns.tolist()
    df_test_encoded = df_test_encoded.reindex(columns=all_feature_cols, fill_value=0)
    
    X_train_full = df_train_encoded.astype(float).values
    X_test_full = df_test_encoded.astype(float).values
    
    print(f"Número total de features no modelo completo: {len(all_feature_cols)}")
    
    # Features sem Lags (Estudo de Ablação)
    lag_cols = [c for c in all_feature_cols if 'lag' in c or 'media_movel' in c]
    pure_infra_cols = [c for c in all_feature_cols if c not in lag_cols]
    
    pure_indices = [all_feature_cols.index(c) for c in pure_infra_cols]
    X_train_pure = X_train_full[:, pure_indices]
    X_test_pure = X_test_full[:, pure_indices]
    
    results = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # =========================================================================
    # MODELO 1: V6-FULL (POISSON) — O MODELO OFICIAL CAMPEÃO
    # =========================================================================
    print("\n" + "-" * 60)
    print(">>> 1/3 Treinando Modelo V6-Full (Poisson Deviance)...")
    print("    Parâmetros: n_estimators=200, criterion='poisson', max_features='sqrt', min_samples_leaf=5, max_samples=0.9")
    
    rf_v6_poisson = RandomForestRegressor(
        n_estimators=200,
        criterion='poisson',
        max_features='sqrt',
        min_samples_leaf=5,
        max_samples=0.9,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    
    t0 = time.time()
    rf_v6_poisson.fit(X_train_full, y_train)
    t_v6 = time.time() - t0
    print(f"    Treinamento concluído em {t_v6:.1f}s.")
    
    y_pred_v6 = rf_v6_poisson.predict(X_test_full)
    r2_v6 = r2_score(y_test, y_pred_v6)
    mae_v6 = mean_absolute_error(y_test, y_pred_v6)
    mse_v6 = mean_squared_error(y_test, y_pred_v6)
    rmse_v6 = np.sqrt(mse_v6)
    d2_v6 = d2_tweedie_score(y_test, y_pred_v6, power=1) # Poisson D2 score
    
    # Acurácia de previsão em trechos seguros (onde y_test == 0)
    zero_mask = (y_test == 0)
    mae_zeros_v6 = mean_absolute_error(y_test[zero_mask], y_pred_v6[zero_mask])
    
    results['V6_Poisson'] = {
        'nome': 'V6-Full (Poisson Oficial)',
        'r2': r2_v6, 'mae': mae_v6, 'mse': mse_v6, 'rmse': rmse_v6,
        'd2_poisson': d2_v6, 'mae_zeros': mae_zeros_v6, 'tempo': t_v6,
        'features': len(all_feature_cols)
    }
    
    # Salvar modelo oficial
    v6_path = os.path.join(base_dir, 'random_forest_v6_tcc.pkl')
    joblib.dump(rf_v6_poisson, v6_path)
    print(f"    Modelo V6 Oficial salvo em: {v6_path}")
    
    # Salvar features treinadas
    feat_path = os.path.join(base_dir, 'v6_feature_names.pkl')
    joblib.dump(all_feature_cols, feat_path)

    # =========================================================================
    # MODELO 2: V6-MSE (SQUARED ERROR) — BENCHMARK DIRETO COM V1-V5
    # =========================================================================
    print("\n" + "-" * 60)
    print(">>> 2/3 Treinando Modelo V6-MSE (Benchmark Squared Error)...")
    print("    Parâmetros: n_estimators=200, criterion='squared_error', max_features='sqrt', min_samples_leaf=5, max_samples=0.9")
    
    rf_v6_mse = RandomForestRegressor(
        n_estimators=200,
        criterion='squared_error',
        max_features='sqrt',
        min_samples_leaf=5,
        max_samples=0.9,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    
    t0 = time.time()
    rf_v6_mse.fit(X_train_full, y_train)
    t_mse = time.time() - t0
    print(f"    Treinamento concluído em {t_mse:.1f}s.")
    
    y_pred_mse = rf_v6_mse.predict(X_test_full)
    r2_mse = r2_score(y_test, y_pred_mse)
    mae_mse = mean_absolute_error(y_test, y_pred_mse)
    mse_mse = mean_squared_error(y_test, y_pred_mse)
    rmse_mse = np.sqrt(mse_mse)
    d2_mse = d2_tweedie_score(y_test, y_pred_mse, power=1)
    mae_zeros_mse = mean_absolute_error(y_test[zero_mask], y_pred_mse[zero_mask])
    
    results['V6_MSE'] = {
        'nome': 'V6-MSE (Benchmark)',
        'r2': r2_mse, 'mae': mae_mse, 'mse': mse_mse, 'rmse': rmse_mse,
        'd2_poisson': d2_mse, 'mae_zeros': mae_zeros_mse, 'tempo': t_mse,
        'features': len(all_feature_cols)
    }

    # =========================================================================
    # MODELO 3: V6-ABLATION (INFRAESTRUTURA PURA — SEM LAGS)
    # =========================================================================
    print("\n" + "-" * 60)
    print(">>> 3/3 Treinando Modelo V6-Ablation (Infraestrutura Pura DNIT sem Lags)...")
    print("    Parâmetros: Apenas Pavimento + Calendário + UF (sem inércia temporal)")
    
    rf_v6_pure = RandomForestRegressor(
        n_estimators=200,
        criterion='poisson',
        max_features='sqrt',
        min_samples_leaf=5,
        max_samples=0.9,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    
    t0 = time.time()
    rf_v6_pure.fit(X_train_pure, y_train)
    t_pure = time.time() - t0
    print(f"    Treinamento concluído em {t_pure:.1f}s.")
    
    y_pred_pure = rf_v6_pure.predict(X_test_pure)
    r2_pure = r2_score(y_test, y_pred_pure)
    mae_pure = mean_absolute_error(y_test, y_pred_pure)
    mse_pure = mean_squared_error(y_test, y_pred_pure)
    rmse_pure = np.sqrt(mse_pure)
    d2_pure = d2_tweedie_score(y_test, y_pred_pure, power=1)
    mae_zeros_pure = mean_absolute_error(y_test[zero_mask], y_pred_pure[zero_mask])
    
    results['V6_PureInfra'] = {
        'nome': 'V6-Ablation (Infra Pura)',
        'r2': r2_pure, 'mae': mae_pure, 'mse': mse_pure, 'rmse': rmse_pure,
        'd2_poisson': d2_pure, 'mae_zeros': mae_zeros_pure, 'tempo': t_pure,
        'features': len(pure_infra_cols)
    }
    
    pure_path = os.path.join(base_dir, 'random_forest_v6_pure_infra.pkl')
    joblib.dump(rf_v6_pure, pure_path)
    joblib.dump(pure_infra_cols, os.path.join(base_dir, 'v6_pure_feature_names.pkl'))

    # =========================================================================
    # RELATÓRIO CONSOLIDADO E TABELA COMPARATIVA
    # =========================================================================
    print("\n" + "=" * 80)
    print("           TABELA CONSOLIDADA DE PERFORMANCE DA ARQUITETURA V6          ")
    print("=" * 80)
    print(f"{'Métrica':<22} | {'V6-Full (Poisson)':<18} | {'V6-MSE (Benchmark)':<18} | {'V6-Ablation (Infra)':<18}")
    print("-" * 80)
    print(f"{'R² (Escala Real)':<22} | {r2_v6:.4f}{'':<12} | {r2_mse:.4f}{'':<12} | {r2_pure:.4f}{'':<12}")
    print(f"{'MAE (Erro Médio)':<22} | {mae_v6:.4f} ac/mês{'':<6} | {mae_mse:.4f} ac/mês{'':<6} | {mae_pure:.4f} ac/mês{'':<6}")
    print(f"{'RMSE':<22} | {rmse_v6:.4f}{'':<12} | {rmse_mse:.4f}{'':<12} | {rmse_pure:.4f}{'':<12}")
    print(f"{'Poisson D² Score':<22} | {d2_v6:.4f}{'':<12} | {d2_mse:.4f}{'':<12} | {d2_pure:.4f}{'':<12}")
    print(f"{'MAE Trechos Seguros':<22} | {mae_zeros_v6:.4f} ac/mês{'':<6} | {mae_zeros_mse:.4f} ac/mês{'':<6} | {mae_zeros_pure:.4f} ac/mês{'':<6}")
    print(f"{'Tempo de Treino':<22} | ~{t_v6:.1f}s{'':<13} | ~{t_mse:.1f}s{'':<13} | ~{t_pure:.1f}s{'':<13}")
    print("-" * 80)
    
    # Exporta relatório em texto
    rep_path = os.path.join(base_dir, 'model_comparison_v6.txt')
    with open(rep_path, 'w', encoding='utf-8') as f_rep:
        f_rep.write("==============================================================================\n")
        f_rep.write("               RELATÓRIO OFICIAL DE PERFORMANCE DA ARQUITETURA V6             \n")
        f_rep.write("==============================================================================\n\n")
        f_rep.write(f"Dataset de Treino: 2020–2023 ({len(df_train):,} observações)\n")
        f_rep.write(f"Dataset de Teste : 2024–2025 ({len(df_test):,} observações fora do tempo)\n")
        f_rep.write(f"Percentual de Zeros no Teste: {zero_mask.sum()/len(y_test)*100:.2f}%\n\n")
        f_rep.write("TABELA DE MÉTRICAS:\n")
        f_rep.write(f"{'Métrica':<22} | {'V6-Full (Poisson)':<18} | {'V6-MSE (Benchmark)':<18} | {'V6-Ablation (Infra)':<18}\n")
        f_rep.write("-" * 80 + "\n")
        f_rep.write(f"{'R² (Escala Real)':<22} | {r2_v6:.4f}{'':<12} | {r2_mse:.4f}{'':<12} | {r2_pure:.4f}{'':<12}\n")
        f_rep.write(f"{'MAE (Erro Médio)':<22} | {mae_v6:.4f} ac/mês{'':<6} | {mae_mse:.4f} ac/mês{'':<6} | {mae_pure:.4f} ac/mês{'':<6}\n")
        f_rep.write(f"{'RMSE':<22} | {rmse_v6:.4f}{'':<12} | {rmse_mse:.4f}{'':<12} | {rmse_pure:.4f}{'':<12}\n")
        f_rep.write(f"{'Poisson D² Score':<22} | {d2_v6:.4f}{'':<12} | {d2_mse:.4f}{'':<12} | {d2_pure:.4f}{'':<12}\n")
        f_rep.write(f"{'MAE Trechos Seguros':<22} | {mae_zeros_v6:.4f} ac/mês{'':<6} | {mae_zeros_mse:.4f} ac/mês{'':<6} | {mae_zeros_pure:.4f} ac/mês{'':<6}\n")
        f_rep.write("-" * 80 + "\n")
    print(f"\nRelatório oficial exportado em: {rep_path}")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    execute_training_v6()
