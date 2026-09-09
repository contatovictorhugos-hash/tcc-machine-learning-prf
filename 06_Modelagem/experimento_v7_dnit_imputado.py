"""
06_Modelagem/experimento_v7_dnit_imputado.py
============================================
EXPERIMENTO DE LABORATÓRIO — ARQUITETURA V7 (ORACLE SIMULATION)
Objetivo: Simular o cenário contrafactual em que 100% da malha federal
dispõe de medições de pavimento do DNIT (eliminação do sentinela -1.0
via interpolação e imputação espacial-temporal) e ajuste de hiperparâmetros.

ATENÇÃO: Este script é estritamente experimental para pesquisa e testes.
NÃO altera o Modelo V6 Oficial nem o storytelling da monografia.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
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

def impute_dnit_100_percent(df_train, df_test):
    """
    Executa a simulação contrafactual de 100% de cobertura do DNIT:
    Aprende os padrões espaciais a partir do conjunto de treino (2020-2023)
    e imputa suavemente os trechos sem medição no treino e no teste.
    """
    print("\n--- Executando Imputação Espacial-Temporal Contrafactual do DNIT (100% Cobertura) ---")
    dnit_cols = ['icp', 'icc', 'icm']
    
    # Treino apenas com medições válidas (>= 0)
    train_valid = df_train[df_train['has_dnit_data'] == 1].copy()
    
    # 1. Priors por Trecho
    trecho_priors = train_valid.groupby(['br', 'trecho_10km'])[dnit_cols].mean()
    # 2. Priors por BR e UF
    bruf_priors = train_valid.groupby(['br', 'uf'])[dnit_cols].mean()
    # 3. Priors por UF
    uf_priors = train_valid.groupby('uf')[dnit_cols].mean()
    # 4. Global Priors
    global_priors = train_valid[dnit_cols].mean()
    
    print(f"Médias Globais de Referência DNIT: ICP={global_priors['icp']:.1f}, ICC={global_priors['icc']:.1f}, ICM={global_priors['icm']:.1f}")
    
    def apply_imputation(target_df, name=""):
        df_out = target_df.copy()
        missing_mask = (df_out['has_dnit_data'] == 0)
        n_missing = missing_mask.sum()
        pct_missing = n_missing / len(df_out) * 100
        print(f"[{name}] Total de linhas a imputar: {n_missing:,} ({pct_missing:.1f}%)")
        
        for col in dnit_cols:
            s1 = df_out.set_index(['br', 'trecho_10km']).index.map(trecho_priors[col])
            s2 = df_out.set_index(['br', 'uf']).index.map(bruf_priors[col])
            s3 = df_out['uf'].map(uf_priors[col])
            
            val_col = np.where(df_out[col] >= 0, df_out[col], s1)
            val_col = np.where(pd.notna(val_col), val_col, s2)
            val_col = np.where(pd.notna(val_col), val_col, s3)
            val_col = np.where(pd.notna(val_col), val_col, global_priors[col])
            
            df_out[col] = np.round(val_col.astype(float), 2)
            
        df_out['has_dnit_data'] = 1
        return df_out

    df_train_imp = apply_imputation(df_train, "Treino 2020-2023")
    df_test_imp = apply_imputation(df_test, "Teste 2024-2025")
    return df_train_imp, df_test_imp

def run_experiment_v7():
    print("=" * 80)
    print("  EXPERIMENTO V7 — RANDOM FOREST COM DNIT 100% IMPUTADO & HIPERPARÂMETROS")
    print("=" * 80)
    
    data_path = find_dataset()
    print(f"Carregando base de: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    
    # 1. Partição Temporal Idêntica à V6 Oficial (Out-of-time)
    df_train_raw = df[df['ano'].isin([2020, 2021, 2022, 2023])].copy()
    df_test_raw = df[df['ano'].isin([2024, 2025])].copy()
    
    # 2. Imputação Fisiográfica Contrafactual
    df_train_imp, df_test_imp = impute_dnit_100_percent(df_train_raw, df_test_raw)
    
    target_col = 'quantidade_acidentes'
    y_train = df_train_imp[target_col].values
    y_test = df_test_imp[target_col].values
    
    cols_to_drop = [
        target_col, 'ano', 'mes', 'br', 'trecho_10km',
        'total_mortos', 'total_feridos_graves', 'total_veiculos', 'is_zero_real'
    ]
    
    cat_cols = ['estacao_ano', 'uf']
    df_train_encoded = pd.get_dummies(df_train_imp.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False)
    df_test_encoded = pd.get_dummies(df_test_imp.drop(columns=cols_to_drop), columns=cat_cols, drop_first=False)
    
    all_feature_cols = df_train_encoded.columns.tolist()
    df_test_encoded = df_test_encoded.reindex(columns=all_feature_cols, fill_value=0)
    
    X_train = df_train_encoded.astype(float).values
    X_test = df_test_encoded.astype(float).values
    
    print(f"\nMatriz de Treino: {X_train.shape[0]:,} linhas x {X_train.shape[1]} features")
    print(f"Matriz de Teste : {X_test.shape[0]:,} linhas x {X_test.shape[1]} features")
    
    print("\n" + "-" * 70)
    print(">>> Treinando Random Forest V7-Experimental (Poisson)...")
    print("    Parâmetros: n_estimators=200, criterion='poisson', max_depth=28,")
    print("                max_features=0.35, min_samples_leaf=4, max_samples=0.9")
    
    rf_v7 = RandomForestRegressor(
        n_estimators=200,
        criterion='poisson',
        max_depth=28,
        max_features=0.35,
        min_samples_leaf=4,
        max_samples=0.9,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    
    t0 = time.time()
    rf_v7.fit(X_train, y_train)
    t_v7 = time.time() - t0
    print(f"    Treinamento concluído em {t_v7:.1f}s.")
    
    y_pred_v7 = rf_v7.predict(X_test)
    r2_v7 = r2_score(y_test, y_pred_v7)
    mae_v7 = mean_absolute_error(y_test, y_pred_v7)
    mse_v7 = mean_squared_error(y_test, y_pred_v7)
    rmse_v7 = np.sqrt(mse_v7)
    d2_v7 = d2_tweedie_score(y_test, y_pred_v7, power=1)
    
    zero_mask = (y_test == 0)
    mae_zeros_v7 = mean_absolute_error(y_test[zero_mask], y_pred_v7[zero_mask])
    mae_sinistros_v7 = mean_absolute_error(y_test[~zero_mask], y_pred_v7[~zero_mask])
    r2_sinistros_v7 = r2_score(y_test[~zero_mask], y_pred_v7[~zero_mask])
    
    importances_v7 = rf_v7.feature_importances_
    df_imp_v7 = pd.Series(importances_v7, index=all_feature_cols).sort_values(ascending=False)
    
    dnit_feats = ['icp', 'icc', 'icm', 'has_dnit_data']
    dnit_v7_total = df_imp_v7[[c for c in dnit_feats if c in df_imp_v7.index]].sum() * 100
    
    r2_v6 = 0.8678
    mae_v6 = 0.6137
    rmse_v6 = 1.2786
    d2_v6 = 0.7964
    mae_zeros_v6 = 0.2889
    dnit_v6_total = 2.24
    
    print("\n" + "=" * 85)
    print("         TABELA COMPARATIVA: MODELO V6 (OFICIAL) vs MODELO V7 (EXPERIMENTAL)        ")
    print("=" * 85)
    print(f"{'Métrica / Característica':<32} | {'V6 Oficial (Base Real)':<22} | {'V7 Experimental (100% DNIT)':<22}")
    print("-" * 85)
    print(f"{'Cobertura do Pavimento DNIT':<32} | {'17.7% (sentinela -1)':<22} | {'100.0% (imputação contínua)':<22}")
    print(f"{'max_depth':<32} | {'None (ilimitado)':<22} | {'28 (regularizado)':<22}")
    print(f"{'max_features':<32} | {'sqrt (~15%)':<22} | {'0.35 (35% do subespaço)':<22}")
    print("-" * 85)
    print(f"{'R² Global (Teste 2024-25)':<32} | {r2_v6:.4f}{'':<16} | {r2_v7:.4f}{'':<16}")
    print(f"{'MAE Global':<32} | {mae_v6:.4f} ac/mês{'':<10} | {mae_v7:.4f} ac/mês{'':<10}")
    print(f"{'RMSE':<32} | {rmse_v6:.4f}{'':<16} | {rmse_v7:.4f}{'':<16}")
    print(f"{'Poisson D² Score':<32} | {d2_v6:.4f}{'':<16} | {d2_v7:.4f}{'':<16}")
    print(f"{'MAE em Trechos Seguros (Y=0)':<32} | {mae_zeros_v6:.4f} ac/mês{'':<10} | {mae_zeros_v7:.4f} ac/mês{'':<10}")
    print(f"{'MAE em Trechos com Sinistros (Y>=1)':<32} | {'~1.38 ac/mês':<22} | {mae_sinistros_v7:.4f} ac/mês{'':<10}")
    print(f"{'R² em Trechos com Sinistros (Y>=1)':<32} | {'~0.62':<22} | {r2_sinistros_v7:.4f}{'':<16}")
    print("-" * 85)
    print(f"{'Peso Relativo DNIT (MDI)':<32} | {dnit_v6_total:.2f}%{'':<17} | {dnit_v7_total:.2f}%{'':<17}")
    print("=" * 85)
    
    print("\nTop 10 Features por Importância MDI no Modelo V7:")
    for rank, (feat, imp) in enumerate(df_imp_v7.head(10).items(), 1):
        print(f"  {rank:02d}. {feat:<30}: {imp*100:6.2f}%")
        
    print("\nImportâncias Específicas do DNIT no V7:")
    for feat in dnit_feats:
        if feat in df_imp_v7:
            print(f"  * {feat:<20}: {df_imp_v7[feat]*100:.2f}%")
            
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_rep = os.path.join(base_dir, "relatorio_experimento_v7.txt")
    with open(out_rep, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO EXPERIMENTAL — ARQUITETURA V7 (ORACLE DNIT 100%)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"R² Global V7: {r2_v7:.4f} (vs V6: {r2_v6:.4f})\n")
        f.write(f"MAE Global V7: {mae_v7:.4f} (vs V6: {mae_v6:.4f})\n")
        f.write(f"RMSE V7: {rmse_v7:.4f} (vs V6: {rmse_v6:.4f})\n")
        f.write(f"Poisson D² V7: {d2_v7:.4f} (vs V6: {d2_v6:.4f})\n")
        f.write(f"Peso do DNIT no V7: {dnit_v7_total:.2f}% (vs V6: {dnit_v6_total:.2f}%)\n\n")
        f.write("Top 10 Features V7:\n")
        for f_name, imp in df_imp_v7.head(10).items():
            f.write(f"  - {f_name}: {imp*100:.2f}%\n")
    print(f"\n[OK] Relatório experimental salvo em: {out_rep}")
    print("Lembrete: O Modelo V6 Oficial e todos os arquivos do TCC continuam intactos.")

if __name__ == "__main__":
    run_experiment_v7()
