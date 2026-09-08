4 RESULTADOS
O capítulo abaixo apresenta um compilado dos resultados empíricos obtidos através da modelagem preditiva que foi desenvolvida para estimar a sinistralidade nas rodovias federais brasileiras. A partir do pipeline metodológico consolidado no capítulo anterior, as evidências neste capítulo são discutidas sob a ótica da acurácia estatística e da utilidade prática para a administração pública.
A estrutura do capítulo organiza-se em etapas progressivas. Primeiro inicia-se com a avaliação comparativa das cinco arquiteturas avaliadas e a seleção do Modelo V5 com validação temporal. Em seguida, examinam-se o diagnóstico de resíduos e a hierarquia de importância de variáveis. A influência da infraestrutura viária do DNIT é isolada por meio de técnicas de interpretabilidade e testes contrafactuais. Por fim, exploram-se as simulações de cenários táticos para a PRF e a disponibilização dos resultados em painéis analíticos no Power BI.


4.1 Avaliação Comparativa das Arquiteturas e Seleção do Modelo V5
Conforme determinado no RNF-5, foram treinadas e avaliadas cinco configurações do algoritmo Random Forest Regressor, descritas no capítulo 3.2.4.2. O processo de evolução das métricas reflete os ajustes progressivos nos hiperparâmetros e na estratégia de amostragem. A Tabela 5 consolida os resultados de desempenho obtidos no conjunto de testes designados.
Tabela 5 — Resultados de performance das arquiteturas no conjunto de teste
Métrica
	V1
	V2
	V3
	V4
	V5
	R²
	0,9892
	0,9770
	0,9678
	0,9556
	0,9726
	MAE
	0,2276
	0,3450
	0,4011
	0,4107
	0,4181
	MSE
	0,2985
	0,6418
	0,8980
	1,2398
	0,8787
	Tempo de treino
	~15 s
	~8 s
	~5 s
	~6 s
	~7 s
	Fonte: Elaborado pelo autor (2026).
A interpretação desses resultados exige uma reflexão sobre a estratégia de partição dos dados. As versões V1 a V4 utilizaram amostragem aleatória convencional, train_test_split. Embora essas versões apresentem métricas aparentemente superiores, como o R² de 0,9892 na V1, essa abordagem introduz uma imprecisão metodológica conhecida como data leakage temporal em dados de séries históricas. A partição aleatória permite que o modelo treine com observações cronologicamente futuras em relação a partes do conjunto de teste, inflando as métricas de desempenho.
Em oposição a isso, o Modelo V5 foi treinado exclusivamente com a partição temporal cronológica restrita de dados de 2019 a 2023 para treino e dados de 2024 e 2025 para teste. Essa estratégia reflete com fidelidade o cenário operacional real em produção, no qual o modelo preditivo deve estimar períodos futuros com base única em dados históricos. 
A análise comparativa entre as cinco arquiteturas conduz à seleção definitiva do Modelo V5 como a solução final do trabalho. Essa decisão fundamenta-se em três pilares estruturais que são:
   * Rigor Metodológico: O Modelo V5 é a única arquitetura cujas métricas foram obtidas sem data leakage temporal. O R² de 0,9726 e o MAE de 0,4181 confirmam a capacidade preditiva do sistema para períodos futuros, indicando um erro médio inferior a meio acidente por trecho/mês e superando as metas do RNF-4 (R² > 0,90 e MAE < 1,0).
   * Aderência aos Requisitos Não Funcionais: Ao operar na escala original dos dados, sem transformações de normalização ou padronização na variável alvo, o algoritmo atende ao RNF-1 de forma nativa. Suas previsões são consumidas diretamente nos dashboards em Power BI. Além disso, a manutenção dos valores extremos, outliers, em conformidade com o RNF-2, garante que os trechos de alta severidade não sejam atenuados.
   * Riqueza Analítica: Além do desempenho numérico, a configuração do V5, max_features='sqrt', min_samples_leaf=2, max_samples=0.9, reduz a correlação entre as árvores e fornece um mapa de importância de atributos que orienta intervenções preventivas de infraestrutura e fiscalização.


