import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_eda_plots():
    log_content = []
    def log(msg):
        print(msg)
        log_content.append(msg)

    # Cria pasta para gráficos exploratórios
    output_dir = 'graficos_eda_tcc'
    os.makedirs(output_dir, exist_ok=True)
    
    log("\n12. Gerando 10 Graficos Exploratorios (EDA) a partir da Base Bruta (df_prf_consolidado_19-25.csv)")
    
    # Load dataset
    log("   - Carregando df_prf_consolidado_19-25.csv...")
    # Lendo apenas colunas necessarias para economia de RAM, ou um sample se necessario. 
    # Considerando que sao 470k linhas, o Pandas aguenta se formos diretos.
    try:
        df = pd.read_csv('df_prf_consolidado_19-25.csv', sep=',', encoding='utf-8')
    except Exception as e:
        log(f"Erro: {e}")
        return
        
    # Converter data
    df['data_inversa'] = pd.to_datetime(df['data_inversa'], errors='coerce')
    df['ano_ocorrencia'] = df['data_inversa'].dt.year
    
    # Config visual
    sns.set_theme(style="whitegrid", context="paper")
    
    # G1: Mapa de Nulos (Missing Values)
    # Mostra a qualidade dos dados antes de qualquer limpeza
    plt.figure(figsize=(12, 6))
    missing_data = df.isnull().mean() * 100
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    if not missing_data.empty:
        sns.barplot(x=missing_data.values, y=missing_data.index, palette='Reds_r')
        plt.title('EDA 1: Porcentagem de Valores Nulos por Variável na Base Bruta')
        plt.xlabel('% de Nulos')
    else:
        plt.text(0.5, 0.5, 'Sem Nulos Significativos', ha='center', va='center')
        plt.title('EDA 1: Analise de Nulos (Base Original)')
    plt.savefig(f'{output_dir}/eda1_missing_values.png', bbox_inches='tight')
    plt.close()
    log("   - [EDA Grafico 1] eda1_missing_values.png: Fundamental para justificar as escolhas no Pipeline de ETL (Tratamento de Nulos).")

    # G2: Frequencia Bruta do Tipo de Acidente vs Letalidade
    plt.figure(figsize=(12, 8))
    top_10_tipos = df['tipo_acidente'].value_counts().head(10).index
    df_top_tipos = df[df['tipo_acidente'].isin(top_10_tipos)]
    sns.countplot(data=df_top_tipos, y='tipo_acidente', palette='magma', order=top_10_tipos)
    plt.title('EDA 2: Top 10 Tipos de Acidentes Relatados (Dado Bruto)')
    plt.xlabel('Frequência Absoluta')
    plt.ylabel('Tipo de Acidente')
    plt.savefig(f'{output_dir}/eda2_top_tipos_acidente.png', bbox_inches='tight')
    plt.close()
    log("   - [EDA Grafico 2] eda2_top_tipos_acidente.png: Uma visa original da rotulacao humana dos eventos, essencial para a introducao da exploracao de dados.")

    # G3: Causa de Acidente (Top 10)
    plt.figure(figsize=(12, 8))
    top_10_causas = df['causa_acidente'].value_counts().head(10).index
    df_top_causas = df[df['causa_acidente'].isin(top_10_causas)]
    sns.countplot(data=df_top_causas, y='causa_acidente', palette='viridis', order=top_10_causas)
    plt.title('EDA 3: Principais Causas Condicionantes dos Acidentes')
    plt.xlabel('Volume')
    plt.ylabel('Causa Registrada')
    plt.savefig(f'{output_dir}/eda3_causas_acidente.png', bbox_inches='tight')
    plt.close()
    log("   - [EDA Grafico 3] eda3_causas_acidente.png: Analise do comportamento causal bruto, ajudando a entender por que \"causas\" geram data leakage na prevencao espacial.")

    # G4: Distribuicao de Idade dos Veiculos (Ano do Acidente - Ano Fabricacao/Base)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='fase_dia', hue='sentido_via', palette='Set2')
    plt.title('EDA 4: Frequência de Acidentes por Fase do Dia e Sentido da Via')
    plt.savefig(f'{output_dir}/eda4_fase_sentido.png', bbox_inches='tight')
    plt.close()
    log("   - [EDA Grafico 4] eda4_fase_sentido.png: Avalia comportamentos da combinacao do horario de luminosidade com a direcionalidade do transito bruto.")

    # G5: Dispersao de Acidentes Geograficos (BR vs Mortos)
    # Selecionando as top 10 BRs mais violentas
    plt.figure(figsize=(12, 6))
    top_10_brs = df.groupby('br')['mortos'].sum().sort_values(ascending=False).head(10).index
    df_top_brs = df[df['br'].isin(top_10_brs)]
    sns.barplot(data=df_top_brs, x='br', y='mortos', estimator=sum, errorbar=None, palette='Reds_r', order=top_10_brs)
    plt.title('EDA 5: Top 10 Rodovias (BRs) com Maior Letalidade Acumulada')
    plt.xlabel('BR')
    plt.ylabel('Total de Mortos (Bruto)')
    plt.savefig(f'{output_dir}/eda5_top10_brs_letais.png', bbox_inches='tight')
    plt.close()
    log("   - [EDA Grafico 5] eda5_top10_brs_letais.png: Crucial para justificar geograficamente quais rodovias puxam a cauda longa de mortes.")

    # G6: Concentracao de Vitimas (Pessoas vs Veículos no Acidente)
    plt.figure(figsize=(10, 8))
    # Amostragem para scatter nao travar
    df_sample = df.sample(n=10000, random_state=42) if len(df) > 10000 else df
    sns.scatterplot(data=df_sample, x='veiculos', y='pessoas', alpha=0.3, hue='classificacao_acidente', palette='tab10')
    plt.title('EDA 6: Dispersão Bruta - Ocupantes vs Veículos Envolvidos (Sample 10k)')
    plt.savefig(f'{output_dir}/eda6_dispersao_veiculos_pessoas.png', bbox_inches='tight')
    plt.close()
    log("   - [EDA Grafico 6] eda6_dispersao_veiculos_pessoas.png: Revela anomalias brutas, como acidentes envolvendo onibus (1 veiculo = dezenas de pessoas).")

    # G7: Estados (UF) Mais Atingidos
    plt.figure(figsize=(12, 6))
    ordem_ufs = df['uf'].value_counts().index
    sns.countplot(data=df, x='uf', palette='crest', order=ordem_ufs)
    plt.title('EDA 7: Volume Total de Ocorrências por Estado Federativo (UF)')
    plt.savefig(f'{output_dir}/eda7_volume_por_uf.png', bbox_inches='tight')
    plt.close()
    log("   - [EDA Grafico 7] eda7_volume_por_uf.png: Explora o desbalanceamento espacial natural dos acidentes em territorio nacional.")

    # G8: Composicao da Estrutura da Via Bruta (Tracado_Via sujo)
    plt.figure(figsize=(10, 10))
    # Para nao quebrar o grafico de torta com as 1200 categorias, pegamos o top 5 puro e agrupamos resto.
    tracados = df['tracado_via'].value_counts()
    top_tracados = tracados.head(5)
    outros = pd.Series([tracados.iloc[5:].sum()], index=['Múltiplos Tracados/Outros'])
    dados_tracado = pd.concat([top_tracados, outros])
    
    plt.pie(dados_tracado, labels=dados_tracado.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=140)
    plt.title('EDA 8: Composição do Traçado da Via Original')
    plt.savefig(f'{output_dir}/eda8_pie_tracado_original.png', bbox_inches='tight')
    plt.close()
    log("   - [EDA Grafico 8] eda8_pie_tracado_original.png: Justifica visualmente a complexidade do texto 'tracado_via' antes do tratamento realizado no ETL (onde separamos 'Reta;Declive', etc).")

    # G9: Classificacao Acidente (Pizza Letalidade)
    plt.figure(figsize=(8, 8))
    class_counts = df['classificacao_acidente'].value_counts()
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=['#4C72B0', '#55A868', '#C44E52'])
    plt.title('EDA 9: Classificação Gravimétrica dos Acidentes (Target Distribution)')
    plt.savefig(f'{output_dir}/eda9_classificacao_pizza.png', bbox_inches='tight')
    plt.close()
    log("   - [EDA Grafico 9] eda9_classificacao_pizza.png: Ilustra a variavel gravimetrica original na classe bruta antes do pareamento numérico.")

    # G10: Evolucao de Feridos Ilsesos e Ignorados
    plt.figure(figsize=(10, 6))
    agrupado_ano_condicoes = df.groupby('ano_ocorrencia')[['ilesos', 'ignorados', 'feridos']].sum().reset_index()
    sns.lineplot(data=agrupado_ano_condicoes, x='ano_ocorrencia', y='ilesos', label='Ilesos', marker='^')
    sns.lineplot(data=agrupado_ano_condicoes, x='ano_ocorrencia', y='feridos', label='Feridos Totais', marker='o')
    sns.lineplot(data=agrupado_ano_condicoes, x='ano_ocorrencia', y='ignorados', label='Estado Ignorado', marker='s')
    
    plt.title('EDA 10: Evolução Bruta das Condições Pós-Evento (Ilesos vs Feridos)')
    plt.ylabel('Contagem Total de Pessoas')
    plt.savefig(f'{output_dir}/eda10_condicoes_pessoas_anos.png', bbox_inches='tight')
    plt.close()
    log("   - [EDA Grafico 10] eda10_condicoes_pessoas_anos.png: Explica as proporcoes agregadas as vitimas documentadas na pista e evidencia a utilidade dos cortes realizados no treinamento do modelo.")

    # Append to execution_log
    with open('execution_log.txt', 'a', encoding='utf-8') as f:
        f.write('\n\n')
        f.write('\n'.join(log_content))
    
    print(f"\nGraficos gerados com sucesso na pasta '{output_dir}' e registrados no execution_log.txt")

if __name__ == "__main__":
    generate_eda_plots()
