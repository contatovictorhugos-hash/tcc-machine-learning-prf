SUMÁRIO
1 INTRODUÇÃO        7
1.1 Motivação        8
1.2 Objetivo Geral        9
1.3 Objetivo Específico        9
1.4 Metodologia        9
1.5 Organização do Texto        10
2 REVISÃO DA LITERATURA        11
2.1 Panorama da segurança viária no Brasil        11
2.2 De análises descritivas à modelagem preditiva        15
2.2.1 O processo de ETL (Extract, Transform, Load)        16
2.2.2 Pipeline de Ciência de Dados e metodologia CRISP-DM        17
2.2.3 Random Forest: fundamentos teóricos e aplicações        19
2.2.4 Fundamentos matemáticos e estatísticos        20
2.2.5 Random Forest no contexto de segurança viária        20
2.3 Business Intelligence: transformando dados em decisões        22
2.3.1 Power BI como ferramenta de Business Intelligence        23
2.4 Conceitos de engenharia de software        24
2.4.1 Requisitos funcionais        25
2.4.2 Requisitos não funcionais        26
2.4.3 Metodologias de desenvolvimento        26
2.4.4 Estratégias de teste        28
2.4.5 Arquitetura de sistemas        29
3 MATERIAIS E MÉTODOS        32
3.1 Materiais        32
3.1.1 Linguagem Python        32
3.1.2 Google Colab        33
3.1.3 Bibliotecas        34
3.1.4 Random Forest        37
3.1.5 PowerBI        42
3.2 Métodos        43
3.2.1 Entendimento de negócio        43
3.2.1.1 Identificação dos stakeholders        44
3.2.1.2 Requisitos Funcionais e Não Funcionais        46
3.2.1.3 Estratégia de Versionamento        48
3.2.2 Entendimento dos Dados        48
3.2.3 Preparação do Dados        57
3.2.3.1 Transformação e Otimização Estrutural        57
3.2.3.2 Engenharia de Atributos        58
3.2.3.3 Tratamento e Integração da Base do DNIT        62
3.2.3.4 Junção Intervalar e Resolução da Armadilha Zero-Inflated        62
3.2.4 Modelagem        63
3.2.4.1 Estratégia de Divisão dos Dados e Correção do Data Leakage        63
3.2.4.2 Especificação Técnica das Arquiteturas Avaliadas        64
3.2.4.3 Validação interna        67
4 RESULTADOS        69
4.1 Avaliação Comparativa das Arquiteturas e Seleção do Modelo V5        69
4.2 Diagnóstico do Modelo V5        71
4.3 Análise de Importância das Variáveis        72
4.4 Avaliação do Impacto da Infraestrutura Rodoviária (DNIT)        75
4.4.1 Diagnóstico do Sentinela -1.0 (Vias Não Monitoradas pelo DNIT)        75
4.4.2 Análise SHAP e Teste A/B Ceteris Paribus (BR-381 KM 340-350 / MG)        76
4.4.3 Não Linearidade no PDP e Retorno sobre Investimento        78
4.4.4 Showcase de Otimização Global Argmax (BR-40 KM 110-120)        80
4.5 Simulações de Cenários Táticos e Estudo de Caso Operacional (PRF)        81
4.5.1 Metodologia por Mediana e Avaliação dos Cenários Táticos        82
4.5.2 Estudo de Caso Real: BR-101 (KM 70)        85
5 DISCUSSÃO        87
5.1 Da Reatividade à Antecipação        87
5.2 A Ilusão das Métricas e a Assimetria Real        88
5.3 Economia de Infraestrutura e Diretrizes para o DNIT        89
5.4 A Hierarquia entre Geometria Física e Dinâmica Temporal        90


