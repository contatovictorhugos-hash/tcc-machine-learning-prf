"""
model_training_v5.py
====================
Treinamento da Arquitetura V5 (Temporal Split - Modelo Oficial Selecionado) do TCC.

Especificações (Tabela 4 da Monografia):
- Divisão dos dados: Temporal Split (Treino: 2019–2023 | Teste: 2024–2025)
- Pré-processamento: Nenhum (Escala Original da Variável Alvo)
- n_estimators: 200 árvores
- max_features: 'sqrt' (Feature Subspace)
- criterion: squared_error (MSE)
- max_samples: 0.9 (Bagging Subsampling: 90% das amostras por árvore)
- Regularização: max_depth=None, min_samples_leaf=2
- Validação Adicional: OOB Score (Out-Of-Bag) e TimeSeriesSplit (5 folds)
- Métricas esperadas (Tabela 5): R²=0.9726, MAE=0.4181, MSE=0.8787
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def find_data_file(filename="df_model_ready_tcc.csv"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, "..", filename),
        os.path.join(base_dir, "..", "05_Pre_Modelagem", filename),
        os.path.join(base_dir, "..", "kdf_final", filename)
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[1]

def execute_training_v5():
    print("=" * 70)
    print("  TREINAMENTO MODELO V5 — PARTIÇÃO TEMPORAL OFICIAL (TCC 2026)")
    print("=" * 70)

    data_path = find_data_file()
    print(f"Carregando dataset de: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERRO: Dataset não encontrado em {data_path}.")
        return None

    df = pd.read_csv(data_path, low_memory=False)
    target_col = 'quantidade_acidentes'

    # Identificar coluna de ano (pode ser 'ano' ou 'ano_base')
    year_col = 'ano_base' if 'ano_base' in df.columns else ('ano' if 'ano' in df.columns else None)
    if not year_col:
        print("ERRO: Coluna de ano ('ano' ou 'ano_base') não encontrada no dataset.")
        return None

    # ==========================================================
    # 1. SPLIT TEMPORAL CRONOLÓGICO (2019-2023 vs 2024-2025)
    # ==========================================================
    print(f"\nAplicando Split Temporal usando a coluna temporal '{year_col}'...")
    df_train = df[df[year_col].isin([2019, 2020, 2021, 2022, 2023])].copy()
    df_test = df[df[year_col].isin([2024, 2025])].copy()

    # Ordenar por tempo
    sort_cols = [year_col, 'mes'] if 'mes' in df.columns else [year_col]
    df_train = df_train.sort_values(by=sort_cols)
    df_test = df_test.sort_values(by=sort_cols)

    cols_to_drop = [target_col, 'ano', 'mes', 'br', 'trecho_10km', 'ano_base']
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    X_train = df_train[feature_cols].fillna(0).astype(float)
    y_train = df_train[target_col]

    X_test = df_test[feature_cols].fillna(0).astype(float)
    y_test = df_test[target_col]

    print(f"Treino Temporal (2019-2023): {len(X_train):,} amostras")
    print(f"Teste Temporal  (2024-2025): {len(X_test):,} amostras")
    print(f"Features: {X_train.shape[1]}")

    # ==========================================================
    # 2. ARQUITETURA DE TREINAMENTO V5
    # ==========================================================
    print("\nInstanciando RandomForestRegressor V5:")
    print("  - n_estimators: 200")
    print("  - max_features: 'sqrt'")
    print("  - criterion: 'squared_error'")
    print("  - max_samples: 0.9 (Sub-bagging anti-overfitting)")
    print("  - min_samples_leaf: 2")
    print("  - max_depth: None")
    print("  - oob_score: True")

    model_v5 = RandomForestRegressor(
        n_estimators=200,
        max_features='sqrt',
        criterion='squared_error',
        max_samples=0.9,
        min_samples_leaf=2,
        max_depth=None,
        oob_score=True,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    t0 = time.time()
    print("\nTreinando Modelo V5...")
    model_v5.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"Treinamento concluído em {t_train:.1f}s.")

    # ==========================================================
    # 3. AVALIAÇÃO NO TESTE TEMPORAL (2024-2025)
    # ==========================================================
    y_pred = model_v5.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    oob = model_v5.oob_score_

    print("\n" + "=" * 50)
    print("RESULTADOS DE PERFORMANCE DO MODELO V5 (TEMPORAL 2024-2025):")
    print(f"  R² (Teste Temporal) : {r2:.4f}")
    print(f"  MAE (Erro Médio)    : {mae:.4f} acidentes/mês")
    print(f"  MSE (Erro Quadr.)   : {mse:.4f}")
    print(f"  OOB Score (Treino)  : {oob:.4f}")
    print(f"  Tempo de Treino     : ~{t_train:.0f}s")
    print("=" * 50)

    # ==========================================================
    # 4. VALIDAÇÃO CRUZADA TEMPORAL (TimeSeriesSplit)
    # ==========================================================
    print("\nExecutando Cross-Validation Temporal (TimeSeriesSplit n_splits=5)...")
    try:
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model_v5, X_train, y_train, cv=tscv, scoring='r2', n_jobs=-1)
        print("  R² por Fold CV:", [round(s, 4) for s in cv_scores])
        print(f"  Média CV R²   : {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    except Exception as e:
        print(f"  Aviso na validação cruzada: {e}")

    # ==========================================================
    # 5. EXPORTAÇÃO DO MODELO
    # ==========================================================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'random_forest_v5_tcc.pkl')
    joblib.dump(model_v5, model_path)
    print(f"\nModelo V5 serializado e salvo em: {model_path}")

    # Salva gráficos de diagnóstico se a pasta de gráficos existir
    try:
        graficos_dir = os.path.join(base_dir, "..", "07_Variaveis_Importantes")
        os.makedirs(graficos_dir, exist_ok=True)
        
        # Feature Importance
        imp_series = pd.Series(model_v5.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
        plt.figure(figsize=(10, 6))
        imp_series.plot(kind='barh', color='#2b5c8f')
        plt.title('Top 20 Variáveis por Importância — Random Forest V5 (Temporal)', fontsize=12, fontweight='bold')
        plt.xlabel('Importância (MDI)')
        plt.tight_layout()
        plt.savefig(os.path.join(graficos_dir, 'feature_importance_v5.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Aviso gerando diagnósticos visuais: {e}")

    return model_v5, {'r2': r2, 'mae': mae, 'mse': mse, 'oob': oob, 'time': t_train}

if __name__ == "__main__":
    execute_training_v5()
