"""
model_training_v1.py
====================
Treinamento da Arquitetura V1 (Baseline Histórico) do TCC.

Especificações (Tabela 4 da Monografia):
- Divisão dos dados: Random 80% treino / 20% teste
- Pré-processamento: Nenhum (escala original)
- n_estimators: 100 árvores
- max_features: 1.0 (todas as variáveis por nó)
- criterion: squared_error (MSE)
- max_samples: 1.0 (Bootstrap padrão)
- Regularização: max_depth=None
- Métricas esperadas (Tabela 5): R²=0.9892, MAE=0.2276, MSE=0.2985
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

def execute_training_v1():
    print("=" * 70)
    print("  TREINAMENTO MODELO V1 — BASELINE HISTÓRICO (TCC)")
    print("=" * 70)

    data_path = find_data_file()
    print(f"Carregando dataset de: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERRO: Dataset não encontrado em {data_path}.")
        return None

    df = pd.read_csv(data_path, low_memory=False)
    target_col = 'quantidade_acidentes'
    cols_to_drop = [target_col, 'ano', 'mes', 'br', 'trecho_10km', 'ano_base']
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    X = df[feature_cols].fillna(0).astype(float)
    y = df[target_col]
    print(f"Features: {X.shape[1]} | Registros: {X.shape[0]:,}")

    # Split 80/20 conforme Tabela 4 do TCC
    print("\nExecutando Split Aleatório 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Treino: {len(X_train):,} amostras | Teste: {len(X_test):,} amostras")

    # Arquitetura V1
    print("\nInstanciando RandomForestRegressor V1:")
    print("  - n_estimators: 100")
    print("  - max_features: 1.0 (todas)")
    print("  - criterion: 'squared_error'")
    print("  - max_depth: None")

    model = RandomForestRegressor(
        n_estimators=100,
        max_features=1.0,
        criterion='squared_error',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    t0 = time.time()
    print("\nTreinando Modelo V1...")
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"Treinamento concluído em {t_train:.1f}s.")

    # Avaliação
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("\n" + "-" * 50)
    print("RESULTADOS DE PERFORMANCE (MODELO V1):")
    print(f"  R²  : {r2:.4f}")
    print(f"  MAE : {mae:.4f} acidentes/mês")
    print(f"  MSE : {mse:.4f}")
    print(f"  Tempo de Treino: ~{t_train:.0f}s")
    print("-" * 50)

    # Salvar modelo
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'random_forest_v1_tcc.pkl')
    joblib.dump(model, model_path)
    print(f"Modelo V1 salvo em: {model_path}")

    return model, {'r2': r2, 'mae': mae, 'mse': mse, 'time': t_train}

if __name__ == "__main__":
    execute_training_v1()
