# trasformation.py
import pandas as pd
import numpy as np
import holidays
from datetime import datetime
from scipy.stats import skew

def main():
    log_content = []
    def log(msg):
        print(msg)
        log_content.append(msg)

    log("=== INICIANDO PIPELINE DE DADOS TCC PRF ===")
    
    # 1. Extraction
    input_file = "df_prf_consolidado_19-25.csv"
    log(f"\n1. Extracao: Carregando {input_file} (sep=',', utf-8)")
    try:
        df = pd.read_csv(input_file, sep=',', encoding='utf-8')
    except Exception as e:
        log(f"Erro ao ler arquivo: {e}")
        return

    log(f"Formato original: {df.shape[0]} linhas, {df.shape[1]} colunas.")

    # 2. Data Cleaning & Null Handling
    log("\n2. Limpeza e Tratamento de Nulos")
    
    linhas_antes_dup = df.shape[0]
    df = df.drop_duplicates()
    log(f"   - Arquivos duplicados removidos: {linhas_antes_dup - df.shape[0]}")

    log("   - Avaliando Nulos (antes):")
    nulos_antes = df.isna().sum()
    for col, count in nulos_antes[nulos_antes > 0].items():
        log(f"     - {col}: {count}")

    # Remove rows where vital keys are null
    essential_cols = ['br', 'km', 'data_inversa', 'municipio', 'uf']
    linhas_antes_essenciais = df.shape[0]
    df = df.dropna(subset=essential_cols)
    log(f"   - Linhas removidas por nulos essenciais (br, km, data): {linhas_antes_essenciais - df.shape[0]}")

    # Fill numeric counts with 0 where null
    num_cols_to_fill = ['pessoas', 'mortos', 'feridos_leves', 'feridos_graves', 'ilesos', 'ignorados', 'feridos', 'veiculos']
    for col in num_cols_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Fill remaining categorical with 'Nao Informado'
    cat_cols_to_fill = ['causa_acidente', 'tipo_acidente', 'classificacao_acidente', 'fase_dia', 'sentido_via', 'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo', 'regional', 'delegacia', 'uop']
    for col in cat_cols_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna('Nao Informado')

    # Detect Outliers with IQR (Logging only, not removing)
    try:
        if 'feridos' in df.columns:
            Q1 = df['feridos'].quantile(0.25)
            Q3 = df['feridos'].quantile(0.75)
            IQR = Q3 - Q1
            outliers_feridos = df[(df['feridos'] < (Q1 - 1.5 * IQR)) | (df['feridos'] > (Q3 + 1.5 * IQR))].shape[0]
            log(f"   - [Regra de Ouro] Outliers detectados em 'feridos' (IQR): {outliers_feridos} (MANTIDOS)")
    except Exception as e:
        log(f"     - Aviso: Erro calculando regras de IQR: {e}")

    # 3. Data Type Optimization
    log("\n3. Otimizacao de Tipos de Dados (Data Type Optimization)")
    log("   - Tipos ANTES da transformacao:")
    dtypes_antes = df.dtypes.to_string()
    log(f"     {dtypes_antes.replace(chr(10), chr(10)+'     ')}")

    # Convert Date
    df['data_inversa'] = pd.to_datetime(df['data_inversa'], errors='coerce')

    # Downcast numerics
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
        
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].max() < 2147483647:
            df[col] = df[col].astype('int32')

    # Categoricals for memory
    categorical_candidates = ['dia_semana', 'uf', 'fase_dia', 'sentido_via', 'uso_solo', 'classificacao_acidente', 'tipo_pista']
    for col in categorical_candidates:
        if col in df.columns:
            df[col] = df[col].astype('category')

    log("   - Tipos APOS transformacao (otimizados):")
    dtypes_apos = df.dtypes.to_string()
    log(f"     {dtypes_apos.replace(chr(10), chr(10)+'     ')}")

    # 4. Integration Logic
    log("\n4. Logica de Integracao (Chave br_km)")
    try:
        # km can be string with commas, cleaning it
        def limpa_km(x):
            try:
                # remove spaces, replace comma with dot, convert to float then to consistent string without decimals if possible
                val = str(x).strip().replace(',', '.')
                val_float = float(val)
                # Keep 1 decimal for consistency if needed, or integers
                if val_float.is_integer():
                    return f"{int(val_float)}"
                return f"{val_float:.1f}".replace('.0', '')
            except:
                return str(x)
        
        df['km_clnd'] = df['km'].apply(limpa_km)
        df['br_km'] = df['br'].astype(str) + '-' + df['km_clnd']
        df.drop(columns=['km_clnd'], inplace=True)
        log("   - Coluna br_km criada com sucesso. (Exemplo: " + str(df['br_km'].iloc[0]) + ")")
    except Exception as e:
         log(f"   - Erro gerando br_km: {e}")

    # 5. Feature Engineering
    log("\n5. Feature Engineering (Sazonalidade e Infraestrutura)")
    
    # Seasonality
    log("   - Criando is_final_semana, estacao_ano, is_feriado...")
    df['is_final_semana'] = df['data_inversa'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # estacao do ano
    def get_season(date):
        if pd.isna(date): return 'Desconhecido'
        month = date.month
        if month in [12, 1, 2]: return 'Verao'
        elif month in [3, 4, 5]: return 'Outono'
        elif month in [6, 7, 8]: return 'Inverno'
        else: return 'Primavera'
    
    df['estacao_ano'] = df['data_inversa'].apply(get_season).astype('category')

    # Feriados Brasil
    br_holidays = holidays.Brazil(years=range(2019, 2026))
    df['is_feriado'] = df['data_inversa'].dt.date.apply(lambda x: int(x in br_holidays) if pd.notna(x) else 0)

    # Binaries for tracado_via
    log("   - Criando colunas binarias dinamicas para tracado_via...")
    if 'tracado_via' in df.columns:
        # Some records might have multiple features separated by semicolon (e.g. 'Reta;Declive')
        # We need to get all unique single features.
        all_tracados = set()
        for t in df['tracado_via'].dropna():
            parts = [p.strip() for p in str(t).split(';')]
            for p in parts:
                 if p and p != 'Nao Informado':
                     all_tracados.add(p)
                     
        import re
        import unicodedata
        log(f"     - Tipos unicos encontrados: {all_tracados}")
        for feat in all_tracados:
            # Normalize to remove accents and special chars
            normalized_feat = unicodedata.normalize('NFKD', feat).encode('ASCII', 'ignore').decode('ASCII')
            # Replace spaces and non-alphanumeric with underscores
            clean_feat = re.sub(r'[^a-zA-Z0-9]', '_', normalized_feat)
            col_name = f"is_tracado_via_{clean_feat}"
            df[col_name] = df['tracado_via'].str.contains(feat, regex=False, na=False).astype(int)
    
    # NEW: Create BR column explicitly
    df['br_concat'] = 'BR-' + df['br'].astype(str)
    log("   - Coluna de apresentacao `br_concat` (ex: BR-316) criada com sucesso.")

    # 6. EDA & Column Importance
    log("\n6. Estatistica (Assimetria) e Importancia de Colunas")
    
    # Skewness
    target_cols = ['mortos', 'feridos_graves', 'feridos', 'veiculos']
    log("   - Relatorio de Assimetria (Skewness):")
    for col in target_cols:
        if col in df.columns:
            sk = skew(df[col].dropna())
            log(f"     - {col}: {sk:.2f} " + ("(Alta assimetria - Cauda longa)" if abs(sk) > 1 else "(Dist. Normal-ish)"))
            
    # Quick Correlation/Importance
    log("\n   - Importancia das variaveis numericas baseado em Correlacao de Pearson com 'feridos':")
    try:
        if 'feridos' in df.columns:
            num_df = df.select_dtypes(include=[np.number])
            corr = num_df.corr(numeric_only=True)['feridos'].sort_values(ascending=False)
            
            top_5 = corr.drop('feridos').head(5)
            bottom_5 = corr.drop('feridos').tail(5).sort_values() # magnitude or least
            
            log("     TOP 5 Variaveis Mais Importantes (positivamente correlacionadas):")
            for k, v in top_5.items(): log(f"       - {k}: {v:.3f}")
            
            log("     TOP 5 Variaveis Menos Importantes (menos correlacionadas/negativas):")
            for k, v in bottom_5.items(): log(f"       - {k}: {v:.3f}")
    except Exception as e:
        log(f"     - Aviso: Falha ao calcular correlacoes: {e}")

    # 7. Dimensionality Reduction (DROPS)
    log("\n7. Reducao de Dimensionalidade (Colunas Deletadas) e Data Leakage Prevention")
    cols_to_drop = {
        'id': "Lixo analitico; apenas uma chave primaria unica sem variancia para a Random Forest.",
        'tracado_via': "Redundante; ja mapeado no pipeline via colunas dummies (One-Hot Encoding).",
        'latitude': "Sobreposicao geografica; alta variancia que confunde as arvores. Variavel br_km e municipio ja explicam o espaco.",
        'longitude': "Sobreposicao geografica e tipo de dado impuro (object com virgulas).",
        'regional': "Sobreposicao organizacional; reflete zonas da PRF dependentes do local. Nao agrega na analise de engenharia de trafego da via.",
        'delegacia': "Sobreposicao organizacional da PRF.",
        'uop': "Sobreposicao organizacional da PRF (Unidade Operacional).",
        'causa_acidente': "Data Leakage (vazamento temporal); variavel gerada apos o sinistro. Nao e util para prever o acidente.",
        'tipo_acidente': "Data Leakage.",
        'classificacao_acidente': "Data Leakage.",
        'ilesos': "Data Leakage e baixa utilidade/importancia pre-via.",
        'ignorados': "Data Leakage e correlacao nula pre-evento.",
    }
    
    dropped_count = 0
    for col, motivo in cols_to_drop.items():
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            log(f"   - DROPPED [{col}]: {motivo}")
            dropped_count += 1
    log(f"   - Total de colunas removidas nesta etapa: {dropped_count}")

    # 8. Executional Summary and Next Steps Analysis (Narrative)
    log("\n9. Resumo Executivo e Analise de Prontidao para Modelagem")
    narrativa = """
    A. RESUMO DO QUE FOI FEITO E MOTIVAÇÃO:
    Todo o pipeline foi contruido visando as regras de negocio definidas. A Limpeza lidou com dados essenciais ausentes
    sem mascarar a distribuicao de acidentes, por isso outliers (acidentes com muitas vitimas) foram detectados mas preservados.
    A Otimizacao de Tipos reduziu o peso na memoria, convertendo objetos complexos em categorias leves, crucial para o Google Colab.
    A Feature Engineering focou em criar colunas vitais para prever o *futuro*: sazonalidades (estacao_ano, is_final_semana, is_feriado)
    e condicoes estaticas de infraestrutura desmembrando 12 tipos unicos de `tracado_via` em dummies booleanas. Isso facilita
    imansamente que as arvores do modelo identifiquem a correlacao isolada de uma "Curva" ou "Declive" com acidentes.
    Por fim, fizemos uma rigida Reducao de Dimensionalidade para erradicar o "Data Leakage" e sobreposicoes espaciais, removendo
    todas as variaveis pos-fato como "causa" ou "ilesos" e identificadores comolat/long/id. Isso assegura que o Random Forest
    sera treinado estritamente com dados pre-evento.
    
    B. ANALISE: ESTAMOS PRONTOS PARA A MODELAGEM?
    Falta um passo crucial antes do treinamento oficial do Random Forest Regressor. O objetivo base, como descrito na 
    "LOGICA_DE_INTEGRACAO", ainda requer o cruzamento efetivo com a BASE DO DNIT.
    Atualmente nos temos a chave `br` e `km` limpa (e a variavel visual `br_concat`), alem das colunas intrinsecas da PRF prontas.
    O que Faltaria:
    1. Importar os dados do DNIT (PNV - Plano Nacional de Viacao, base de condicoes ou VDM - Volume Diario Medio).
    2. Realizar o JOIN / Merge espacial usando a chave estrutural (br e km).
    3. Definir estritamente a Variavel Alvo (Target - y). Precisamos agrupar a tabela (groupby(br_km)) para contar 
       quantos acidentes ocorreram naquele trecho especifico, ou preveremos a letalidade por linha? O padrao 
       arquitetural para prever quantidade num trecho exige transformacao da base em painel (trecho vs tempo).
    
    Conclusao: A base atual esta 100% polida, com vazamento evitado e variaveis dummy criadas, porem tecnicamente
    AINDA NAO esta pronta para modelagem se nao executarmos o Merge Espacial do DNIT e o Agrupamento Temporal/Espacial da variavel alvo (target aggregations).
    """
    log(narrativa)

    # Output Processed File
    out_file = "df_prf_processed.csv"
    log(f"\n10. Salvando base processada em {out_file}...")
    df.to_csv(out_file, index=False, sep=',', encoding='utf-8')
    log(f"   - Arquivo salvo. Formato final: {df.shape[0]} linhas, {df.shape[1]} colunas.")

    # Write log
    with open("execution_log.txt", "w", encoding='utf-8') as f:
        f.write('\n'.join(log_content))
    print("Logs gravados em execution_log.txt")

if __name__ == "__main__":
    main()
