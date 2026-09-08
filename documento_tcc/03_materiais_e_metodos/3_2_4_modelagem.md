3.2.4 Modelagem


        Após a preparação da base de dados para atuação do modelo nos capítulos 3.2.2 e 3.3.3, inicia-se uma etapa determinante para o sucesso do projeto, a quarta fase do framework CRISP-DM, a modelagem preditiva. Essa etapa trás a tona o objetivo central do projeto, com o treinamento e comparação de versões do modelo Random Forest, sendo escolhido o modelo mais adequado ao contexto do projeto e aos requisito funcionais e não funcionais, como RNF-1, que proíbe qualquer transformação da variável alvo, ao RNF-4, que exige R² superior a 0,90 e MAE inferior a 1,0 acidente por trecho/mês, e ao RNF-5, que determina a comparação de no mínimo três arquiteturas distintas.
3.2.4.1 Estratégia de Divisão dos Dados e Correção do Data Leakage
        Primeiramente, uma das principais decisões tomadas nessa etapa foi o abandono da divisão aleatória de dados, utilizada nas versões V1 a V4, em prol de uma partição temporal apenas. Essa decisão sacrificou um pouco da pontuação do modelo, entretanto ganhou-se muito no quesito de estar adequado a realidade e ao objetivo do projeto de estimar data futuras, dado que em dados de série temporal, a divisão aleatória por meio da função train_test_split representa uma violação metodológica famosa, conhecida como data leakage temporal, na qual o modelo é treinado com observações futuras em relação a parte do conjunto de teste, inflando artificialmente as métricas de desempenho, pois o modelo apenas aprende a relação das variáveis e não a estimar dados que nunca viu. Conforme mostrado por Hastie, Tibshirani e Friedman (2009), em problemas de forecasting, a avaliação deve sempre respeitar a ordem cronológica dos dados, pois garante que o modelo seja testado exclusivamente com dados que não participaram do treinamento.
        Seguindo essa estrita diretriz, a base final foi ordenada baseada na ordem cronológica e dividida de modo que o conjunto de treino englobasse os registros do período de 2019 a 2023, correspondendo no final da divisão a aproximadamente 70% da base, enquanto o conjunto de teste foi formado pelos dados de 2024 e 2025, representando os 30% restantes. Essa estratégia tem como objetivo simular com máxima fidelidade o cenário operacional real do projeto em produção, no qual previsões são geradas para períodos futuros com base exclusivamente em dados históricos disponíveis até o momento da análise, caso contrário o modelo se torna capaz de ter informações do futuro em alguns cenários de treino.


3.2.4.2 Especificação Técnica das Arquiteturas Avaliadas


Em conformidade com o RNF-5, que estabelece a necessidade de comparar diferentes configurações do algoritmo, foram projetadas cinco arquiteturas do Random Forest Regressor, sendo V1 a primeira e V5 a quinta e última. O objetivo desse processo iterativo foi avaliar como o ajuste dos hiperparâmetros de amostragem, profundidade e descorrelação de atributos influencia a capacidade do modelo de generalizar padrões de sinistralidade sem incorrer em sobreajuste,  conceito de overfitting, quando o modelo decora os dados.


As especificações técnicas de cada versão estão consolidadas na Tabela 4.


Tabela 4 - Especificações técnicas das arquiteturas avaliadas
Parâmetro
	V1 Baseline
	V2 Otimizado
	V3 Otim. + Poda
	V4 Log-Transform
	V5 Temporal
	Divisão
	Random 80/20
	Random 70/30
	Random 70/30
	Random 70/30
	Temporal 70/20
	Pré-processamento
	Nenhum
	Nenhum
	Nenhum
	np.log1p(y)
	Nenhum
	n_estimators
	100
	200
	200
	300
	200
	max_features
	1.0 (todas)
	sqrt
	sqrt
	sqrt
	sqrt
	criterion
	squared_error
	squared_error
	poisson
	squared_error
	squared_error
	max_samples
	1,0
	1,0
	1,0
	0,8
	0,9
	Regularização
	depth=None
	depth=None
	depth=25, leaf=5
	depth=30, leaf=3
	depth=None, leaf=2
	Fonte: Elaborado pelo autor (2026).


A concepção de cada arquitetura respondeu a hipóteses metodológicas bem definidas para atingir o objetivo do projeto:


Modelo V1: A arquitetura V1 funcionou como o baseline inicial do projeto, o modelo foi configurado utilizando a totalidade dos atributos a cada nó, max_features=1.0, amostragem integral e sem restrição de profundidade, max_depth=None. O objetivo dessa versão foi estabelecer um teto de referência no espaço de dados original. Contudo, ao utilizar todas as variáveis em cada divisão binária, a V1 expôs o modelo à dominância de poucas variáveis altamente preditivas, como o traçado reto, gerando árvores fortemente correlacionadas entre si e elevando a variância do estimador conjunto.


Modelo V2: A segunda arquitetura introduziu a restrição max_features='sqrt', instruindo o algoritmo a selecionar aleatoriamente uma sub amostra de apenas sqrt(p) variáveis a cada tentativa de partição nos nós. Fundamentada na tese de Breiman (2001), essa modificação quebra a correlação entre as árvores da floresta, ao desativar temporariamente a árvore para a variável dominante, como o traçado reto, força-se o modelo a explorar padrões latentes secundários, tais como a interação entre a condição do pavimento do DNIT, feriados e tipo de pista.


Modelo V3: A tentativa V3 do modelo testou duas hipóteses teóricas adicionais, a alteração da função de perda e a regularização por poda, pruning. Substituiu-se o erro quadrático médio pelo critério de Poisson, hipótese estatística baseada no fato de que acidentes de trânsito representam dados de contagem discretos e assimétricos. Simultaneamente, aplicou-se uma poda severa fixando a profundidade máxima em 25 níveis, max_depth=25, e exigindo no mínimo 5 observações nas folhas terminais, min_samples_leaf=5. O objetivo foi testar se o controle explícito do crescimento impediria o sobreajuste em ruídos da base.


Modelo V4: A V4 investigou o efeito da redução no tamanho das amostras bootstrap para isso reduziu-se a amostragem de cada árvore para 80% do conjunto de treino, max_samples=0.8, combinada a uma profundidade máxima de 30 níveis, max_depth=30, e regularização de folhas em 3 observações, min_samples_leaf=3. A premissa testada foi que forçar cada árvore a enxergar subconjuntos menores e mais distintos do dataset aumentaria a diversidade do comitê, ensemble, reduzindo a sensibilidade a outliers individuais.


Modelo V5: A última versão do modelo representou a grande virada metodológica do trabalho. Constatou-se que as versões V1 a V4, por utilizarem partição aleatória através do random split, sofriam do fenômeno de data leakage temporal, em que observações do futuro vazavam para o treino do passado, inflando artificialmente as métricas de desempenho, tornando o modelo melhor mas infiel aos critérios estatísticos.


Para corrigir essa fragilidade e simular o ambiente operacional real, a V5 estabeleceu uma Barreira Temporal Estrita:


* Conjunto de Treinamento: Registros cronológicos de 2019 a 2023, sendo ~70% da base histórica.
* Conjunto de Teste: Registros cronológicos exclusivamente de 2024 e 2025, equivalente a ~30% da base histórica.

Além da barreira temporal, a V5 refinou os hiperparâmetros combinando os aprendizados das versões anteriores:

   * Mantiveram-se o Feature Subspace, max_features='sqrt', e a função de perda quadrática, squared_error.
   * Abandonou-se a poda rígida de profundidade, max_depth=None, pois os testes da V3 e V4 demonstraram que limitar a profundidade introduz viés excessivo em dados com cauda longa, impedindo a floresta de representar eventos de alta severidade.
   * Fixou-se uma regularização suave de folhas, min_samples_leaf=2, para suprimir ruídos isolados sem podar o topo da distribuição.
   * Elevou-se o subsampling do Bagging para 90%, max_samples=0.9, preservando representatividade enquanto gera diversidade entre as árvores.
   * Habilitou-se o Out-of-Bag Score, oob_score=True, e a validação cruzada por TimeSeriesSplit de 5 folds sobre os dados de treino, garantindo o diagnóstico de generalização temporal sem contaminação do conjunto de teste.


3.2.4.3 Validação interna


De forma complementar, a avaliação no conjunto de teste, o Modelo V5 foi validado internamente por dois mecanismos independentes, já destrinchados no capítulo 2.2.3. O Out-of-Bag Score, habilitado pelo parâmetro oob_score=True, no qual explorou o mecanismo nativo do Bagging em que cada árvore é avaliada nos 37% de registros de treino que não participaram de sua construção, fornecendo uma estimativa de generalização sem custo computacional adicional, melhorando os resultados sem comprometer o tempo de treino. A convergência entre o OOB Score e o R² obtido no conjunto de teste temporal nos dá prova de que o modelo não possui overfitting.
Ademais, foi aplicada também validação cruzada temporal por meio do TimeSeriesSplit com cinco divisões, garantindo que em cada fold o conjunto de validação sempre correspondesse a um período cronologicamente posterior ao conjunto de treino. Esse procedimento, que foi executado exclusivamente sobre os dados de treinamento (2019-2023), permitiu estimar a estabilidade do desempenho do modelo ao longo do tempo, identificando possíveis instabilidades sazonais antes do teste final. A consistência dos R² entre os cinco folds atestou a robustez temporal do modelo, confirmando que o desempenho observado no teste 2024-2025 não é resultado de um grupo favorável para a assertividade do modelo, mas sim de um aprendizado generalizado dos padrões históricos de sinistralidade.
Com o pipeline de pré-processamento consolidado, as especificações técnicas das cinco arquiteturas definidas e os mecanismos de validação temporal estabelecidos, encerra-se a fase de métodos. O Capítulo 4 apresenta a avaliação empírica de desempenho das arquiteturas, o diagnóstico de resíduos e as simulações de cenários operacionais.
________________
