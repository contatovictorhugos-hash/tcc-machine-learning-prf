import pandas as pd
import numpy as np

def execute_pre_modeling():
    log_content = []
    def log(msg):
        print(msg)
        log_content.append(msg)

    log("\n17. INICIANDO FASE 3: PRÉ-MODELAGEM E ENCODING")

    # 1. Carregar Base Mestre
    log("   - Carregando df_master_tcc.csv...")
    try:
        df = pd.read_csv('df_master_tcc.csv', sep=',', encoding='utf-8', low_memory=False)
    except Exception as e:
        log(f"   - Erro crítico ao ler base Master: {e}")
        return

    log(f"   - Shape inicial: {df.shape}")

    # 2. Criar Buckets Espaço-Temporais -----------------------------------------------
    log("   - Criando colunas de Bucketing: 'mes' e 'trecho_10km'...")
    # Justificativa: Agrupar por data exata ou KM exato causa o problema Zero-Inflated
    # (99% dos KMs/dias sem acidentes). Usar mês e trechos de 10km garante densidade estatística
    # suficiente para a regressão do Random Forest.

    df['data_inversa'] = pd.to_datetime(df['data_inversa'], errors='coerce')
    df['mes'] = df['data_inversa'].dt.month

    df['km'] = df['km'].astype(str).str.replace(',', '.').astype(float, errors='ignore')
    df['km'] = pd.to_numeric(df['km'], errors='coerce')

    df.dropna(subset=['ano', 'mes', 'br', 'km'], inplace=True)
    df['trecho_10km'] = (np.floor(df['km'] / 10) * 10).astype(int)

    # 3. Definir colunas por tipo de agregação ----------------------------------------
    log("   - Mapeando colunas numéricas e categóricas para estratégia de agregação...")
    
    group_keys = ['ano', 'mes', 'br', 'trecho_10km']

    # Cols numéricas contínuas do DNIT e da PRF: agregar com Media
    num_mean_cols = [c for c in ['pessoas', 'mortos', 'feridos_leves', 'feridos_graves',
                                  'feridos', 'veiculos', 'icc', 'icp', 'icm'] if c in df.columns]
    # Colunas booleanas/binárias (is_*): agregar com Soma (total de acidentes no subtrecho com essa condição)
    bool_sum_cols = [c for c in df.columns if c.startswith('is_')]
    # Colunas de estação do ano: tirar a moda via first (alta cardinalidade baixa, muito repetitiva)
    first_cols = [c for c in ['estacao_ano', 'uf', 'dia_semana', 'fase_dia',
                               'sentido_via', 'condicao_metereologica', 'tipo_pista',
                               'cond_pavimento', 'cond_conservacao', 'cond_pav_panela',
                               'cond_pav_remendo', 'cond_pav_trinca', 'cond_cons_rocada',
                               'cond_cons_drenagem', 'cond_cons_sinalizacao'] if c in df.columns]

    agg_dict = {}
    for c in num_mean_cols:
        agg_dict[c] = 'mean'
    for c in bool_sum_cols:
        agg_dict[c] = 'sum'
    for c in first_cols:
        agg_dict[c] = 'first'

    # 4. Criar Target Y: quantidade_acidentes -----------------------------------------
    log("   - Agrupando e criando Target Y = 'quantidade_acidentes' (Frequência Absoluta)...")
    # Justificativa: O Target é a contagem de acidentes por trecho de 10km em cada mês.
    # Isso resolve o Zero-Inflated pois agrupa eventos pontuais em segmentos com densidade.

    df_counts = df.groupby(group_keys).size().reset_index(name='quantidade_acidentes')
    df_grouped = df.groupby(group_keys).agg(agg_dict).reset_index()
    df_model = pd.merge(df_grouped, df_counts, on=group_keys, how='inner')

    log(f"   - Agrupamento OK. Base colapsou: {len(df)} eventos -> {len(df_model)} cenários preditivos.")
    log(f"   - Distribuição do Target 'quantidade_acidentes': min={df_model['quantidade_acidentes'].min()}, max={df_model['quantidade_acidentes'].max()}, media={df_model['quantidade_acidentes'].mean():.2f}")

    # 5. One-Hot Encoding (Scikit-Learn Compatibility) --------------------------------
    log("   - Aplicando One-Hot Encoding nas colunas categoricas restantes...")
    # Justificativa: RandomForestRegressor do Scikit-Learn não aceita strings.
    # pd.get_dummies transforma cada categoria única em uma coluna 0/1 sem escalonamento.

    cols_to_encode = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
    high_cardinality = [c for c in cols_to_encode if df_model[c].nunique() > 100]
    # Drop cols with very high cardinality to avoid RAM blowout
    df_model.drop(columns=high_cardinality, inplace=True, errors='ignore')
    cols_to_encode = [c for c in cols_to_encode if c not in high_cardinality]

    if cols_to_encode:
        df_model = pd.get_dummies(df_model, columns=cols_to_encode, dummy_na=False, drop_first=False)
        for col in df_model.columns:
            if df_model[col].dtype == bool:
                df_model[col] = df_model[col].astype(int)

    log(f"   - Encoding realizado. Dimensões finais: {df_model.shape}")
    if high_cardinality:
        log(f"   - AVISO: {len(high_cardinality)} coluna(s) de alta cardinalidade removidas antes do encoding para evitar explosão de RAM: {high_cardinality}")

    # 6. Exportar Base Final ----------------------------------------------------------
    out_file = "df_model_ready_tcc.csv"
    log(f"   - Exportando base de Regressão final: {out_file}...")
    df_model.to_csv(out_file, index=False, encoding='utf-8')
    log(f"   - Exportação concluída: {df_model.shape[0]} observações | {df_model.shape[1]} variáveis explicativas (X+Y).")
    log(f"   - FASE 3 COMPLETA. O dataset '{out_file}' está pronto para o RandomForestRegressor.")

    # 7. Gravar no Execution Log
    with open('execution_log.txt', 'a', encoding='utf-8') as f:
        f.write('\n\n')
        f.write('\n'.join(log_content))
    print("Logs gravados com sucesso!")

if __name__ == "__main__":
    execute_pre_modeling()
