import pandas as pd
import numpy as np

def execute_integration():
    log_content = []
    def log(msg):
        print(msg)
        log_content.append(msg)

    log("\n15. INICIANDO FASE 2: INTEGRAÇÃO PRF x DNIT (Interval Join)")
    
    # 1. Carregar Bases
    log("   - Carregando df_prf_processed.csv...")
    try:
        df_prf = pd.read_csv('df_prf_processed.csv', sep=',', encoding='utf-8')
    except Exception as e:
        log(f"   - Erro crítico ao ler base PRF: {e}")
        return
        
    log("   - Carregando dnit/df_dnit_processed.csv...")
    try:
        df_dnit = pd.read_csv('dnit/df_dnit_processed.csv', sep=',', encoding='utf-8')
    except Exception as e:
        log(f"   - Erro crítico ao ler base DNIT: {e}")
        return
    
    # 2. Preparar Chaves de Join e Tipagem
    log("   - Harmonizando chaves de join (ano e br)...")
    if 'ano_base' in df_prf.columns:
        df_prf.rename(columns={'ano_base': 'ano'}, inplace=True)
    
    # Garantir exatidão de tipagem (evitar float vs int no 'by' e mixed strings)
    df_prf['ano'] = pd.to_numeric(df_prf['ano'], errors='coerce').astype('Int64')
    df_prf['br'] = pd.to_numeric(df_prf['br'], errors='coerce').astype('Int64')
    df_dnit['ano'] = pd.to_numeric(df_dnit['ano'], errors='coerce').astype('Int64')
    df_dnit['br'] = pd.to_numeric(df_dnit['br'], errors='coerce').astype('Int64')
    
    # Converter o KM da PRF (que pode ter virgula) para float real
    df_prf['km_float'] = df_prf['km'].astype(str).str.replace(',', '.').astype(float)
    df_dnit['km_inicial'] = df_dnit['km_inicial'].astype(float)
    df_dnit['km_final'] = df_dnit['km_final'].astype(float)
    
    # Filtrar eventuais NAs que quebram o asof
    linhas_antes = len(df_prf)
    df_prf = df_prf.dropna(subset=['ano', 'br', 'km_float']).copy()
    if len(df_prf) < linhas_antes:
        log(f"   - Removidas {linhas_antes - len(df_prf)} linhas da PRF com chaves espaciais/temporais nulas/inválidas.")
        
    df_dnit = df_dnit.dropna(subset=['ano', 'br', 'km_inicial']).copy()
    
    # 3. Ordenação rigorosa (Exigência do pd.merge_asof)
    log("   - Realizando o Merge Intervalar ASOF (O(N))...")
    df_prf = df_prf.sort_values('km_float')
    df_dnit = df_dnit.sort_values('km_inicial')
    
    # ASOF Join: Para cada acidente na PRF localiza o trecho DNIT anterior/exato (backward search) 
    # desde que 'ano' e 'br' sejam iguais.
    df_master = pd.merge_asof(
        df_prf,
        df_dnit,
        left_on='km_float',
        right_on='km_inicial',
        by=['ano', 'br'],
        direction='backward'
    )
    
    # 4. Auditoria de Validade (Match Geográfico)
    # Como o backward pega o último trecho inicial, devemos conferir se o KM do acidente
    # realmente está contido ATÉ o fim daquele trecho. Caso ele seja maior que o 'km_final', ele caiu num buraco.
    fora_do_trecho_mask = df_master['km_float'] > df_master['km_final']
    
    # Colunas que vieram do DNIT (removeremos as de ancoragem depois)
    col_dnit = [c for c in df_dnit.columns if c not in ['ano', 'br', 'km_inicial']] + ['km_final']
    
    # Invalida os trechos falsamente pinçados pelo backward
    df_master.loc[fora_do_trecho_mask, col_dnit] = np.nan
    
    total_acidentes = len(df_master)
    # Conta quantos ficaram com km_final de fato (significa match ok)
    # Se por acaso km_final for DataFrame:
    if isinstance(df_master.get('km_final'), pd.DataFrame):
        matches_validos = int(df_master['km_final'].iloc[:, 0].notna().sum())
    else:
        matches_validos = int(df_master['km_final'].notna().sum())
        
    taxa_sucesso = (matches_validos / total_acidentes) * 100
    
    log(f"   - Match Geográfico Realizado. {matches_validos} acidentes ({taxa_sucesso:.2f}%) parearam com um trecho DNIT válido.")
    
    # 5. Tratamento de Ausentes conforme Regra de Ouro (Random Forest)
    log("   - Processando Post-Join nulos sob regra categóricas (sem normalização)...")
    cols_numericas_dnit = ['icc', 'icp', 'icm']
    
    # Para índices numéricos
    for c in cols_numericas_dnit:
        if c in df_master.columns:
            df_master[c] = df_master[c].fillna(-1.0)
            
    # Para flags textuais / visuais da infraestrutura
    cols_cat_dnit = [c for c in col_dnit if c not in cols_numericas_dnit and c != 'km_final']
    for c in cols_cat_dnit:
        if c in df_master.columns:
            # Transforma eventuais floats brutos NaN em strings tratáveis e preenche com regra nula
            df_master[c] = df_master[c].fillna("Nao_Coberto")
            
    # Drop das colunas residuais geo
    cols_limpeza = ['km_float', 'km_inicial', 'km_final']
    df_master.drop(columns=[c for c in cols_limpeza if c in df_master.columns], inplace=True)
    
    # 6. Salvar DataFrame Master
    out_file = "df_master_tcc.csv"
    log(f"   - Salvando a base Mestre definitiva em {out_file}...")
    df_master.to_csv(out_file, index=False, sep=',', encoding='utf-8')
    log(f"   - Exportação concluída. Linhas: {df_master.shape[0]} | Variáveis: {df_master.shape[1]}")
    
    # Escrever os log no final do execution_log
    with open('execution_log.txt', 'a', encoding='utf-8') as f:
        f.write('\n\n')
        f.write('\n'.join(log_content))
    print("Logs gravados!")

if __name__ == "__main__":
    execute_integration()
