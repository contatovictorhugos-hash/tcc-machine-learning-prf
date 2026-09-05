"""
train_compare_all_models.py
===========================
Executador e Comparador das 5 Arquiteturas de Machine Learning do TCC.

Reproduz exatamente o estudo comparativo documentado no Capítulo 4 do TCC
e gera os relatórios Tabela 4 e Tabela 5 consolidados.

Modelos Avaliados:
- V1: Baseline Histórico (80/20, 100 árvores, max_features=1.0)
- V2: Otimizado (70/30, 200 árvores, max_features='sqrt')
- V3: Otimizado + Poda (70/30, Poisson, max_depth=25, min_samples_leaf=5)
- V4: Log-Transform (70/30, np.log1p, 300 árvores, max_samples=0.8)
- V5: Partição Temporal (2019-2023 Treino / 2024-2025 Teste, max_samples=0.9, min_samples_leaf=2)
"""

import os
import sys
import time
import pandas as pd
import numpy as np

def run_all():
    print("=" * 80)
    print("      COMPARAÇÃO COMPLETA DAS 5 ARQUITETURAS DE MACHINE LEARNING (V1 à V5)      ")
    print("=" * 80)

    from model_training_v1 import execute_training_v1
    from model_training_v2 import execute_training_v2
    from model_training_v3 import execute_training_v3
    from model_training_v4 import execute_training_v4
    from model_training_v5 import execute_training_v5

    results = {}

    print("\n>>> 1/5 Executando Modelo V1 (Baseline)...")
    res_v1 = execute_training_v1()
    if res_v1: results['V1'] = res_v1[1]

    print("\n>>> 2/5 Executando Modelo V2 (Otimizado)...")
    res_v2 = execute_training_v2()
    if res_v2: results['V2'] = res_v2[1]

    print("\n>>> 3/5 Executando Modelo V3 (Poisson + Poda)...")
    res_v3 = execute_training_v3()
    if res_v3: results['V3'] = res_v3[1]

    print("\n>>> 4/5 Executando Modelo V4 (Log-Transformation)...")
    res_v4 = execute_training_v4()
    if res_v4: results['V4'] = res_v4[1]

    print("\n>>> 5/5 Executando Modelo V5 (Temporal Oficial)...")
    res_v5 = execute_training_v5()
    if res_v5: results['V5'] = res_v5[1]

    print("\n" + "=" * 80)
    print("           TABELA CONSOLIDADA DE PERFORMANCE DAS ARQUITETURAS (TABELA 5)        ")
    print("=" * 80)
    print(f"{'Métrica':<15} | {'V1 Baseline':<12} | {'V2 Otimizado':<12} | {'V3 Poisson':<12} | {'V4 Log-Trans':<12} | {'V5 Temporal':<12}")
    print("-" * 80)

    for metric in ['r2', 'mae', 'mse', 'time']:
        row_str = f"{metric.upper():<15} | "
        for m in ['V1', 'V2', 'V3', 'V4', 'V5']:
            val = results.get(m, {}).get(metric, None)
            if val is not None:
                if metric == 'time':
                    row_str += f"~{val:.0f}s{'':<9} | "
                else:
                    row_str += f"{val:.4f}{'':<6} | "
            else:
                row_str += f"{'N/D':<12} | "
        print(row_str)
    print("-" * 80)

if __name__ == "__main__":
    run_all()
