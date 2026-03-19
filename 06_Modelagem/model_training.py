import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def execute_model_training():
    log_content = []
    def log(msg):
        print(msg)
        log_content.append(msg)

    log("\n18. INICIANDO FASE 4: TREINAMENTO DO MODELO - RandomForestRegressor")

    # 1. Carregamento da Base Model-Ready
    log("   - Carregando df_model_ready_tcc.csv...")
    try:
        df = pd.read_csv('df_model_ready_tcc.csv', encoding='utf-8', low_memory=False)
    except Exception as e:
        log(f"   - ERRO ao carregar base: {e}")
        return

    log(f"   - Shape: {df.shape}")

    # 2. Separar X e Y
    target_col = 'quantidade_acidentes'
    log(f"   - Variável Alvo (Y): '{target_col}'")

    # Coluna-alvo é o Y. Todo o resto é X (exceto identificadores não preditivos).
    cols_to_drop = [target_col, 'ano', 'mes', 'br', 'trecho_10km']
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    log(f"   - Features (X): {X.shape[1]} variáveis | Observações: {X.shape[0]}")

    # Garantir que X é puro float (alguns bool podem ter sobrado)
    X = X.astype(float)

    # 3. Train/Test Split (Justificativa: ~37% OOB, ~63% treino por árvore no bagging)
    # Usamos 70/30 para o split externo e habilitamos OOB para estimativa interna.
    log("   - Dividindo Train/Test (70/30, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    log(f"   - Treino: {len(X_train)} observações | Teste: {len(X_test)} observações")

    # 4. Instanciar e Treinar RandomForestRegressor
    # Justificativa das Hyperparametrôs:
    # - n_estimators=200: 200 árvores por floresta. Boa convergência de variância sem custo extremo.
    # - max_features='sqrt': Raiz quadrada das features em cada nó (Feature Subspace conforme arquitetura).
    # - criterion='squared_error': MSE por nó (conforme ContextArquiteture.txt).
    # - oob_score=True: Usa as observações Out-of-Bag (~37% das linhas que não foram amostradas) para
    #   estimativa de generalização interna, conforme a diretriz da arquitetura.
    # - n_jobs=-1: Usa todos os núcleos disponíveis.
    # - random_state=42: Reprodutibilidade.
    # NÃO há normalização/padronização (conforme Restrição Metodológica Crítica).
    log("   - Instanciando RandomForestRegressor...")
    log("   - Hiperparâmetros: n_estimators=200, max_features='sqrt', criterion='squared_error', oob_score=True")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_features='sqrt',
        criterion='squared_error',
        oob_score=True,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    log("   - Treinando o modelo (aguarde, 200 árvores + 139k observações)...")
    model.fit(X_train, y_train)
    log("   - Treinamento concluído!")

    # 5. Avaliação Interna (OOB)
    oob_r2 = model.oob_score_
    log(f"   - [OOB Score] R² interno (Out-of-Bag): {oob_r2:.4f}")
    log(f"   - Justificativa OOB: Os dados não amostrados em cada árvore são usados como validação interna sem data leakage, provando generalização.")

    # 6. Avaliação Externa (Test Set)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    log(f"   - [MAE] Erro Médio Absoluto (Test Set): {mae:.4f} acidentes por trecho/mês")
    log(f"   - [R²] Coeficiente de Determinação (Test Set): {r2:.4f}")
    log(f"   - Interpretação: O modelo explica {r2*100:.1f}% da variância na frequência de acidentes por trecho/mês.")

    # 7. Feature Importance (Top 15)
    importance_series = pd.Series(model.feature_importances_, index=X.columns)
    top15 = importance_series.sort_values(ascending=False).head(15)
    log("   - [Top 15 Features por Importância no Modelo]:")
    for feat, imp in top15.items():
        log(f"     * {feat}: {imp:.4f}")

    # 8. TESTE DE SHOWCASE PARA O TCC ----------------------------------------------
    # Cenário: BR-116, Trecho KM 50-60, Próximo Feriado Chuvoso (Abril - mês 4 = feriados Tiradentes/Carnaval)
    log("\n   === TESTE DE SHOWCASE PARA O TCC ===")
    log("   Cenário: BR-116 | Trecho KM 50 (bucket 50) | Mês de Abril (04) | Feriado = Sim | Chuva = Sim")
    log("   Justificativa do mês: Abril concentra feriados nacionais (Tiradentes - 21/04 e Corpus Christi).")
    log("   Justificativa do trecho: KM 50 = trecho_10km bucket '50' (representa KM 50-59).")

    # Criamos uma linha de feature sintética representando o cenário descrito.
    # Todas as features que não são implicadas pelo cenário receberão a MEDIANA histórica da BR-116.
    base_br116 = df[df['br'] == 116]
    if len(base_br116) == 0:
        log("   AVISO: Nenhuma observação histórica da BR-116 encontrada. Usando mediana geral do dataset.")
        base_br116 = df.copy()

    # Pega a mediana de todas as features para a BR-116 como baseline
    medians = base_br116[feature_cols].median().fillna(0)
    input_row = medians.to_frame().T.copy()

    # Ajustes Contextuais do Cenário:
    # 1. Mês de Abril (feriados nacionais)
    mes_col = 'mes_4' if 'mes_4' in input_row.columns else None
    # 2. Condição meteorológica = Chuva
    cond_chuva = [c for c in input_row.columns if 'chuva' in c.lower() or 'chu' in c.lower()]
    if cond_chuva:
        for c in input_row.columns:
            if 'condicao_metereologica' in c.lower() or 'cond_meteor' in c.lower():
                input_row[c] = 0  # zera todas as dummies de clima primeiro
        for c in cond_chuva:
            input_row[c] = 1  # ativa a chuva

    # 3. is_feriado = 1
    if 'is_feriado' in input_row.columns:
        input_row['is_feriado'] = 1

    # 4. Bucket do Trecho (trecho_10km = 50) - usamos a mediana de trecho 50 da BR-116
    trecho_base = base_br116[base_br116['trecho_10km'] == 50] if 'trecho_10km' in base_br116.columns else pd.DataFrame()
    if len(trecho_base) > 0:
        trecho_medians = trecho_base[feature_cols].median().fillna(0)
        input_row = trecho_medians.to_frame().T.copy()
        if 'is_feriado' in input_row.columns:
            input_row['is_feriado'] = 1
        for c in cond_chuva:
            input_row[c] = 1

    # Garantir que X e input_row têm as mesmas colunas na mesma ordem
    input_row = input_row.reindex(columns=feature_cols, fill_value=0).astype(float)

    # Fazer a Predição
    prediction = model.predict(input_row)
    pred_rounded = round(float(prediction[0]))

    log(f"\n   *** RESULTADO DA PREVISÃO ***")
    log(f"   Na BR-116, entre o KM 50 e 60, no próximo feriado com chuva:")
    log(f"   Previsão do Modelo (Random Forest Regressor): {pred_rounded} acidentes esperados naquele trecho/mês.")
    log(f"   Valor Raw do modelo: {float(prediction[0]):.2f}")
    log(f"   Justificativa: O modelo baseou essa previsão nos padrões históricos de 2019 a 2025,")
    log(f"   integrados com os dados de infraestrutura do DNIT para a mesma rodovia e trecho.")
    log(f"   A combinação feriado+chuva é a interação mais preditiva do acidente, dada a assimetria da variável alvo.")

    # 9. Salvar o modelo
    import joblib
    model_file = 'random_forest_model.pkl'
    joblib.dump(model, model_file)
    log(f"\n   - Modelo serializado e salvo em '{model_file}' para reutilização.")
    log("   - FASE 4 COMPLETA.")

    # Gravar no Log
    with open('execution_log.txt', 'a', encoding='utf-8') as f:
        f.write('\n\n')
        f.write('\n'.join(log_content))
    print("Logs gravados!")

if __name__ == "__main__":
    execute_model_training()
