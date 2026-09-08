## 3.2.4 Modelagem


Após a preparação da base de dados para atuação do modelo nos capítulos 3.2.2 e 3.2.3, inicia-se uma etapa determinante para o sucesso do projeto, a quarta fase do framework CRISP-DM, a modelagem preditiva. Essa etapa traz à tona o objetivo central do projeto, com o treinamento e comparação de versões do modelo Random Forest, sendo escolhido o modelo mais adequado ao contexto do projeto e aos requisitos funcionais e não funcionais, como RNF-1, que proíbe qualquer transformação da variável alvo, ao RNF-4, que exige R² superior a 0,70 e MAE inferior a 1,0 acidente por trecho/mês, e ao RNF-5, que determina a comparação de no mínimo três arquiteturas distintas.
### 3.2.4.1 Estratégia de Divisão dos Dados e Correção do Data Leakage
Primeiramente, uma das principais decisões tomadas nessa etapa foi o abandono da divisão aleatória de dados, utilizada nas versões V1 a V4, em prol de uma partição temporal apenas. Essa decisão sacrificou um pouco da pontuação do modelo, entretanto ganhou-se muito no quesito de estar adequado a realidade e ao objetivo do projeto de estimar datas futuras, dado que em dados de série temporal, a divisão aleatória por meio da função train_test_split representa uma violação metodológica famosa, conhecida como data leakage temporal, na qual o modelo é treinado com observações futuras em relação a parte do conjunto de teste, inflando artificialmente as métricas de desempenho, pois o modelo apenas aprende a relação das variáveis e não a estimar dados que nunca viu. Conforme mostrado por Hastie, Tibshirani e Friedman (2009), em problemas de forecasting, a avaliação deve sempre respeitar a ordem cronológica dos dados, pois garante que o modelo seja testado exclusivamente com dados que não participaram do treinamento.
Seguindo essa diretriz, a base final foi ordenada cronologicamente e dividida em etapas bem definidas. O ano de 2019 foi reservado para a inicialização dos atributos defasados, período de warm-up temporal, necessário para calcular os lags de até doze meses sem gerar lacunas na série. A partir dessa janela, o conjunto de treino foi formado pelos registros de 2020 a 2023, correspondendo a 271.440 observações (cerca de 67% da base modelada), enquanto o conjunto de teste reuniu os dados de 2024 e 2025, totalizando 135.720 observações (os 33% restantes). Essa estratégia tem como objetivo simular com fidelidade o cenário operacional real em produção, no qual as previsões para períodos futuros são geradas com base exclusivamente no histórico disponível até o momento da análise, evitando que o modelo receba informações posteriores durante o treinamento.


### 3.2.4.2 Especificação Técnica das Arquiteturas Avaliadas


Em conformidade com o RNF-5, que estabelece a necessidade de comparar diferentes configurações do algoritmo, foram avaliadas seis arquiteturas do Random Forest Regressor, partindo da versão inicial de referência, V1 Baseline, até a formulação definitiva, V6 Oficial. O objetivo desse processo iterativo foi examinar como a escolha dos hiperparâmetros de amostragem, profundidade, critérios de divisão e descorrelação de atributos afeta a capacidade de generalização do modelo, prevenindo o sobreajuste, overfitting. A transição entre as versões acompanhou também o amadurecimento no tratamento dos dados, culminando na incorporação da grade cartesiana de zeros reais e na função de perda por Poisson Deviance na arquitetura final.


As especificações técnicas de cada versão estão consolidadas na Tabela 4.


Tabela 4 - Especificações técnicas das arquiteturas avaliadas

| Parâmetro | V1 Baseline | V2 Otimizado | V3 Otim. + Poda | V4 Log-Transform | V5 Temporal | V6 Oficial (Poisson) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Estratégia de Divisão** | Aleatória 80/20 | Aleatória 70/30 | Aleatória 70/30 | Aleatória 70/30 | Temporal 70/30 | Temporal 67/33 (Warm-up 2019) |
| **Universo Amostral** | Apenas $Y \ge 1$ (139k) | Apenas $Y \ge 1$ (139k) | Apenas $Y \ge 1$ (139k) | Apenas $Y \ge 1$ (139k) | Apenas $Y \ge 1$ (139k) | Grade Cartesiana (N=475.020, 70,6% zeros) |
| **Pré-processamento** | Nenhum | Nenhum | Nenhum | np.log1p(y) | Nenhum | Zeros Reais + Lags (sem vazamento) |
| **n_estimators** | 100 | 200 | 200 | 300 | 200 | 200 |
| **max_features** | 1,0 (todas) | sqrt | sqrt | sqrt | sqrt | sqrt |
| **criterion** | squared_error | squared_error | poisson | squared_error | squared_error | poisson |
| **max_samples** | 1,0 | 1,0 | 1,0 | 0,8 | 0,9 | 0,9 |
| **Regularização** | depth=None | depth=None | depth=25, leaf=5 | depth=30, leaf=3 | depth=None, leaf=2 | depth=None, leaf=5 |

Fonte: Elaborado pelo autor (2026).


A concepção de cada arquitetura respondeu a hipóteses metodológicas bem definidas para atingir o objetivo do projeto:


Modelo V1: A arquitetura V1 funcionou como o ponto de partida inicial do projeto. O modelo foi configurado utilizando a totalidade dos atributos a cada nó (`max_features=1.0`), amostragem integral e sem restrição de profundidade (`max_depth=None`), estabelecendo uma referência inicial no espaço de dados original. Contudo, ao utilizar todas as variáveis em cada divisão binária, a V1 expôs o modelo à dominância de poucas variáveis altamente frequentes, como o traçado em reta, gerando árvores correlacionadas entre si e elevando a variância do comitê.


Modelo V2: A segunda arquitetura introduziu a restrição `max_features='sqrt'`, instruindo o algoritmo a selecionar aleatoriamente uma subamostra de variáveis a cada tentativa de partição nos nós. Com base na formulação de Breiman (2001), essa modificação quebra a correlação entre as árvores da floresta. Ao restringir temporariamente o acesso à variável dominante em cada partição, o algoritmo passa a explorar padrões secundários, como a interação entre a condição do pavimento do DNIT, feriados e tipo de pista.


Modelo V3: A versão V3 testou duas hipóteses teóricas, a alteração da função de perda e a regularização por poda. Substituiu-se o erro quadrático médio pelo critério de Poisson, hipótese baseada no fato de que acidentes de trânsito constituem dados de contagem discretos e assimétricos. Simultaneamente, aplicou-se poda com profundidade máxima de 25 níveis (`max_depth=25`) e mínimo de 5 observações nas folhas terminais (`min_samples_leaf=5`), avaliando se o controle explícito de crescimento conteria o sobreajuste em ruídos locais da base.


Modelo V4: A V4 investigou o efeito da redução no tamanho das amostras, reduzindo a amostragem de cada árvore para 80% do conjunto de treino (`max_samples=0.8`), combinada a uma profundidade máxima de 30 níveis (`max_depth=30`), regularização de folhas em 3 observações (`min_samples_leaf=3`) e transformação logarítmica da variável alvo (`np.log1p`). A premissa testada foi verificar se subconjuntos menores aumentariam a diversidade do comitê, reduzindo a sensibilidade a valores extremos individuais.


Modelo V5: A versão V5 representou a correção metodológica do vazamento temporal. Constatou-se que as versões V1 a V4, por utilizarem partição aleatória, sofriam de vazamento temporal, situação em que observações de períodos posteriores entravam no treinamento de períodos anteriores, inflando as métricas. Para solucionar essa fragilidade, a V5 estabeleceu a divisão cronológica rígida entre treino e teste. Nos hiperparâmetros, mantiveram-se `max_features='sqrt'`, profundidade livre (`max_depth=None`) para não podar a cauda longa, folhas em 2 (`min_samples_leaf=2`) e amostragem em 90% (`max_samples=0.9`), com `criterion='squared_error'`.


Modelo V6: Embora a V5 tenha corrigido o vazamento temporal, diagnósticos posteriores revelaram duas limitações estruturais, a persistência do viés de sobrevivência, por treinar apenas em trechos com acidentes registrados, e a dependência de variáveis de contexto pós-evento. A arquitetura V6 foi desenvolvida para resolver esses pontos de forma definitiva:
* Universo Amostral Completo: O modelo passou a ser treinado sobre a grade cartesiana espaço-temporal de 475.020 observações (5.655 trechos físicos monitorados ao longo de 84 meses), incorporando formalmente 70,61% de zeros reais.
* Eliminação de Variáveis Pós-Evento: Todas as variáveis posteriores ao sinistro foram descartadas. A matriz de preditores consolidou-se em quatro atributos defasados (`acidentes_lag_1m`, `acidentes_lag_2m`, `acidentes_media_movel_3m` e `acidentes_lag_12m`), variáveis de calendário (`dias_no_mes`, `dias_fim_semana`, `dias_feriado`, `estacao_ano`), índices do DNIT (`icp`, `icc`, `icm`, `has_dnit_data`) e a codificação da Unidade Federativa (`uf`).
* Critério de Perda por Poisson Deviance: Como a variável alvo é uma contagem inteira e assimétrica com severa inflação de zeros, adotou-se oficialmente `criterion='poisson'`. Essa função ajusta as estimativas de forma proporcional à magnitude esperada, evitando previsões negativas e reduzindo o erro médio absoluto em trechos seguros para 0,28 acidentes por mês.
* Hiperparâmetros Finais: Fixaram-se 200 árvores (`n_estimators=200`), subespaço aleatório (`max_features='sqrt'`), profundidade livre (`max_depth=None`), regularização de folhas em 5 observações (`min_samples_leaf=5`) e amostragem de 90% (`max_samples=0.9`). Como benchmark comparativo para o Capítulo 4, foram treinadas também as variantes V6-MSE (mesma base com erro quadrático) e V6-Ablation (modelo sem os atributos defasados, isolando o impacto líquido da infraestrutura viária).


### 3.2.4.3 Validação interna


De forma complementar à avaliação no conjunto de teste fora do tempo, o modelo foi verificado internamente por dois mecanismos independentes, conforme os fundamentos teóricos apresentados no Capítulo 2.2.3. O primeiro deles foi o Out-of-Bag Score (OOB), habilitado pelo parâmetro `oob_score=True`. Esse recurso utiliza a propriedade do Bagging em que cada árvore é avaliada na fração de aproximadamente 37% dos dados de treino que não participaram da sua amostra bootstrap, fornecendo uma estimativa direta de generalização sem custo computacional adicional. A convergência entre o OOB Score e as métricas obtidas no conjunto de teste confirma que o algoritmo aprendeu padrões estruturais da série, sem incorrer em sobreajuste.
O segundo mecanismo consistiu na validação cruzada temporal por meio do `Time Series Split` com cinco divisões. Essa técnica assegura que, em cada divisão, o conjunto de validação corresponda estritamente a um período cronologicamente posterior ao bloco de treino. O procedimento foi executado exclusivamente sobre os dados de treinamento (2020 a 2023), permitindo acompanhar a estabilidade do modelo ao longo dos anos e detectar eventuais sensibilidades a variações sazonais antes da avaliação final. A regularidade dos coeficientes de determinação entre os cinco blocos confirmou a consistência temporal do modelo, indicando que o desempenho apurado no teste cego de 2024 e 2025 decorre da capacidade do algoritmo de assimilar a dinâmica histórica de sinistralidade.
Com o pipeline de dados estruturado, as especificações técnicas das arquiteturas definidas e os critérios de validação temporal estabelecidos, encerra-se a fase de métodos. O Capítulo 4 apresenta a avaliação comparativa de desempenho dos modelos, o diagnóstico dos resíduos e as simulações de cenários operacionais desenvolvidas para subsidiar a atuação da PRF e do DNIT.
