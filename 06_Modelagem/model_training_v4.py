"""
model_training_v4.py
====================
Treinamento da Arquitetura V4 (Log-Transformation + Subsampling) do TCC.

Especificações (Tabela 4 da Monografia):
- Divisão dos dados: Random 70% treino / 30% teste
- Pré-processamento: np.log1p(y) no treino, expm1(y_pred) na avaliação
- n_estimators: 300 árvores
- max_features: 'sqrt' (Feature Subspace)
- criterion: squared_error (MSE na escala logarítmica)
- max_samples: 0.8 (Bootstrap Subsampling: 80% das amostras por árvore)
- Regularização: max_depth=30, min_samples_leaf=3
- Métricas esperadas (Tabela 5): R²=0.9556, MAE=0.4107, MSE=1.2398
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

def execute_training_v4():
    print("=" * 70)
    print("  TREINAMENTO MODELO V4 — LOG-TRANSFORMATION + SUBSAMPLING (TCC)")
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

    # Split 70/30 conforme Tabela 4 do TCC
    print("\nExecutando Split Aleatório 70/30...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    print(f"Treino: {len(X_train):,} amostras | Teste: {len(X_test):,} amostras")

    # Transformação Logarítmica no Treino: np.log1p(y)
    print("\nAplicando transformação logarítmica np.log1p(y) no conjunto de treino...")
    y_train_log = np.log1p(y_train)

    # Arquitetura V4
    print("\nInstanciando RandomForestRegressor V4:")
    print("  - n_estimators: 300")
    print("  - criterion: 'squared_error' (na escala log)")
    print("  - max_features: 'sqrt'")
    print("  - max_depth: 30")
    print("  - min_samples_leaf: 3")
    print("  - max_samples: 0.8")

    model = RandomForestRegressor(
        n_estimators=300,
        criterion='squared_error',
        max_features='sqrt',
        max_depth=30,
        min_samples_leaf=3,
        max_samples=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    t0 = time.time()
    print("\nTreinando Modelo V4...")
    model.fit(X_train, y_train_log)
    t_train = time.time() - t0
    print(f"Treinamento concluído em {t_train:.1f}s.")

    # Avaliação: inverter log com expm1 na predição para avaliar na escala real
    print("\nInvertendo predições para escala real via np.expm1...")
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("\n" + "-" * 50)
    print("RESULTADOS DE PERFORMANCE (MODELO V4 - Escala Real):")
    print(f"  R²  : {r2:.4f}")
    print(f"  MAE : {mae:.4f} acidentes/mês")
    print(f"  MSE : {mse:.4f}")
    print(f"  Tempo de Treino: ~{t_train:.0f}s")
    print("-" * 50)

    # Salvar modelo
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'random_forest_v4_tcc.pkl')
    joblib.dump(model, model_path)
    print(f"Modelo V4 salvo em: {model_path}")

    return model, {'r2': r2, 'mae': mae, 'mse': mse, 'time': t_train}

if __name__ == "__main__":
    execute_training_v4()
