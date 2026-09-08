# 4 RESULTADOS


O capítulo abaixo apresenta um compilado dos resultados empíricos obtidos através da modelagem preditiva que foi desenvolvida para estimar a sinistralidade nas rodovias federais brasileiras. A partir do pipeline metodológico consolidado no capítulo anterior, as evidências neste capítulo são discutidas sob a ótica da acurácia estatística e da utilidade prática para a administração pública.
A estrutura do capítulo organiza-se em etapas progressivas. Primeiro inicia-se com a avaliação comparativa das seis arquiteturas avaliadas e a seleção do Modelo V6 com validação temporal. Em seguida, examinam-se o diagnóstico de resíduos e a hierarquia de importância de variáveis. A influência da infraestrutura viária do DNIT é isolada por meio de técnicas de interpretabilidade e testes contrafactuais. Por fim, exploram-se as simulações de cenários táticos para a PRF e a disponibilização dos resultados em painéis analíticos no Power BI.


## 4.1 Avaliação Comparativa das Arquiteturas e Seleção do Modelo V6 Oficial
Conforme determinado no RNF-5, foram treinadas e avaliadas seis configurações do algoritmo Random Forest Regressor, descritas no capítulo 3.2.4.2. O processo de evolução das métricas reflete os ajustes progressivos nos hiperparâmetros e na estratégia de amostragem. A Tabela 5 consolida os resultados de desempenho obtidos no conjunto de testes designados.
Tabela 5: Resultados de performance das arquiteturas no conjunto de teste

| Métrica | V1 Baseline | V2 Otimizado | V3 Otim. + Poda | V4 Log-Transform | V5 Temporal | V6 Oficial (Poisson) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **R²** | 0,9892 | 0,9770 | 0,9678 | 0,9556 | 0,9726 | 0,8678 |
| **MAE** | 0,2276 | 0,3450 | 0,4011 | 0,4107 | 0,4181 | 0,6137 |
| **MSE** | 0,2985 | 0,6418 | 0,8980 | 1,2398 | 0,8787 | 1,6348 |
| **RMSE** | 0,5463 | 0,8011 | 0,9476 | 1,1135 | 0,9374 | 1,2786 |
| **Tempo de treino** | ~15 s | ~8 s | ~5 s | ~6 s | ~7 s | ~206 s |

Fonte: Elaborado pelo autor (2026).
A interpretação desses resultados exige uma reflexão sobre a estratégia de partição dos dados. As versões V1 a V4 utilizaram amostragem aleatória convencional, train_test_split. Embora essas versões apresentem métricas aparentemente superiores, como o R² de 0,9892 na V1, essa abordagem introduz uma imprecisão metodológica conhecida como data leakage temporal em dados de séries históricas. A partição aleatória permite que o modelo treine com observações cronologicamente futuras em relação a partes do conjunto de teste, inflando as métricas de desempenho.
Em oposição a isso, o Modelo V5 estabeleceu a partição temporal cronológica restrita para simular o cenário operacional real em produção, no qual o modelo preditivo deve estimar períodos futuros com base única em dados históricos. Contudo, as versões V1 a V5 operavam exclusivamente sobre registros em que houve sinistros (139.590 observações com Y >= 1), ignorando os períodos em que nenhum acidente ocorreu. Essa limitação foi solucionada na arquitetura V6, que adotou a grade cartesiana com 70,61% de zeros reais e a função de perda baseada em Poisson Deviance, atingindo um R² de 0,8678 em teste fora do tempo sem vazamento de dados.
A análise comparativa entre as arquiteturas conduz à seleção definitiva do Modelo V6 Oficial como a solução final do trabalho. Essa decisão fundamenta-se em três pilares estruturais:
* Rigor Metodológico: O Modelo V6 Oficial é a única arquitetura cujas métricas foram obtidas sem data leakage temporal e com a eliminação do viés de sobrevivência. O R² de 0,8678 e o MAE de 0,6137 confirmam a capacidade preditiva do sistema para períodos futuros, superando a meta definida no RNF-4 (R² > 0,70 e MAE < 1,0 acidente por trecho/mês), com o erro médio caindo para apenas 0,2889 acidentes por mês nos trechos pacificados.
* Aderência aos Requisitos Não Funcionais: Ao operar na escala original dos dados, sem transformações de normalização ou padronização na variável alvo, o algoritmo atende ao RNF-1 de forma nativa. Suas previsões são consumidas diretamente nos dashboards em Power BI. Além disso, a manutenção dos valores extremos, outliers, em conformidade com o RNF-2, garante que os trechos de alta severidade não sejam atenuados.
* Riqueza Analítica: Além do desempenho numérico, a configuração da Arquitetura V6 (critério de Poisson Deviance, max_features='sqrt', min_samples_leaf=5 e max_samples=0.9) reduz a correlação entre as árvores e equilibra o peso dos atributos defasados de tráfego com os índices de infraestrutura do DNIT, orientando intervenções preventivas de engenharia e fiscalização.
