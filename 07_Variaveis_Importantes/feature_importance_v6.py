"""
07_Variaveis_Importantes/feature_importance_v6.py
=================================================
Explicabilidade (XAI) e Importância das Variáveis para o Modelo V6.
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_feature_importance_v6():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "06_Modelagem", "random_forest_v6_tcc.pkl")
    feat_path = os.path.join(base_dir, "06_Modelagem", "v6_feature_names.pkl")
    out_path = os.path.join(base_dir, "07_Variaveis_Importantes", "feature_importance_v6.png")
    
    print("=" * 70)
    print("  EXPLICABILIDADE (XAI) — IMPORTÂNCIA DE FEATURES V6 (TCC 2026)")
    print("=" * 70)
    
    model = joblib.load(model_path)
    feature_names = joblib.load(feat_path)
    
    importances = model.feature_importances_
    df_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    print("\nTop 15 Features por Importância (MDI):")
    for feat, imp in df_imp.head(15).items():
        print(f"  * {feat:<28}: {imp*100:.2f}%")
        
    plt.figure(figsize=(10, 7))
    top15 = df_imp.head(15).iloc[::-1]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(top15)))
    bars = plt.barh(top15.index, top15.values * 100, color=colors, edgecolor='black', alpha=0.85)
    
    plt.xlabel('Importância Relativa (%) — Mean Decrease Impurity', fontsize=11, fontweight='bold')
    plt.title('Top 15 Variáveis Mais Importantes — Random Forest V6 (Oficial)', fontsize=13, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    for b in bars:
        w = b.get_width()
        plt.text(w + 0.3, b.get_y() + b.get_height()/2, f'{w:.1f}%', va='center', fontsize=9, fontweight='bold')
        
    plt.xlim(0, max(top15.values * 100) * 1.18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"\nGráfico oficial salvo em: {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_feature_importance_v6()
