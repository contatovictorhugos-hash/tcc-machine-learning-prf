3.2.3 Preparação dos Dados


Concluída a fase de entendimento dos dados, iniciou-se a preparação da base com o objetivo de estruturar os registros para o treinamento do Random Forest. Nessa etapa, as ocorrências deixam o formato transacional de registros pontuais e passam a compor uma base tabular organizada para que o modelo aprenda os padrões preditivos da sinistralidade nas rodovias federais.


3.2.3.1 Transformação e Otimização Estrutural


O primeiro desafio enfrentado foi computacional, decorrente do volume dos dados. A série histórica da PRF (2019 a 2025) reúne quase meio milhão de ocorrências que, sem tratamento prévio, sobrecarregariam a memória em ambientes com restrições como o Google Colab. Para atender ao RNF-3, realizou-se a conversão dos tipos de dados: variáveis numéricas em float64 foram convertidas para float32, e variáveis de texto com categorias repetidas foram convertidas para o tipo category do Pandas, reduzindo o consumo de memória.

Em seguida, foram tratados os valores ausentes de localização geográfica. Registros sem informação de rodovia (br) ou quilômetro (km) foram descartados, pois a ausência da coordenada espacial impede vincular o acidente ao trecho físico e inviabiliza a posterior integração com os dados do DNIT. Com essa filtragem, a base consolidou-se em 473.734 registros válidos, mantendo a integridade da série histórica. Variáveis de uso puramente administrativo e sem valor preditivo, a exemplo de delegacia e uop, também foram removidas para simplificar a estrutura.

Por fim, realizou-se a transformação estrutural dos dados: a conversão dos acidentes pontuais em trechos espaciais padronizados de 10 km (calculados como trecho_10km = int(km // 10) * 10). Essa discretização consolidou a chave espaço-temporal composta por ano, mes, br e trecho_10km, padronizando a unidade de observação do estudo e permitindo associar diretamente as contagens de sinistros aos segmentos de rodovia avaliados pelo DNIT.


3.2.3.2 Engenharia de Atributos


Com a base limpa, iniciou-se a etapa de engenharia de atributos, processo pelo qual variáveis brutas são materializadas em colunas mensuráveis, ampliando a capacidade explicativa do modelo. Essa etapa é diretamente sustentada pelos insights colhidos na análise exploratória.
O primeiro conjunto de atributos criados foi de natureza temporal e sazonal. Conforme evidenciado pelo gráfico da Figura 14, o volume de acidentes apresenta sazonalidade mensal, com dezembro registrando o maior pico (aproximadamente 44.000 ocorrências) e fevereiro o menor vale, padrão explicável pelo maior fluxo de viagens durante o período de festas e férias escolares. 


Figura 14 - G2: Sazonalidade - Volume de Acidentes por Mês
  

Fonte: Elaborado pelo autor (2026).
Complementarmente, o gráfico da Figura 15 revelou que sexta-feira, sábado e domingo concentram sistematicamente os maiores volumes de sinistros, superando 73.000 ocorrências cada, ao passo que terça-feira apresenta o menor índice semanal. A partir dessas evidências, foram criadas as flags binárias is_final_semana e is_feriado, esta última calculada programaticamente com o auxílio da biblioteca Holidays, que identificou de forma automatizada todos os feriados nacionais, estaduais e municipais do período, incluindo datas móveis como o Carnaval e a Sexta-Feira Santa. Adicionalmente, a variável estacao_ano foi extraída para capturar a sazonalidade climática de longo prazo.
Figura 15 - G5: Frequência de Acidentes por Dia da Semana
  

Fonte: Elaborado pelo autor (2026).
Além do mais, a relevância da variável is_feriado para o projeto, é confirmada quantitativamente pelo gráfico da Figura 16, que demonstra uma média de feridos superior nos dias feriados (1,21) em comparação com os dias comuns (1,14), evidência que, embora aparente ser pequena em magnitude absoluta, é estatisticamente consistente ao longo de toda a série histórica de seis anos.
Figura 16 - G9: Média de Feridos: Dias Comuns (0) vc Feriado (1)
  

Fonte: Elaborado pelo autor (2026).
Com isso, a criação das variáveis temporais e sazonais é essencialmente importante, quando o modelo entrar em produção para a simulação de cenários e para o uso no dia a dia de avaliação de datas.
        Além disso, o segundo conjunto de atributos derivou da variável categórica tracado_via. Foi possível analisar que os três traçados de maior representatividade absoluta na base são a Reta (aproximadamente 330.000 registros), a Curva (aproximadamente 88.000) e a Interseção, com volume consideravelmente menor. Para que o modelo pudesse processar essa informação de forma nativa, aplicou-se um processo de One-Hot Encoding manual, desmembrando a coluna original em variáveis binárias independentes para cada categoria. Essa estratégia transforma uma variável nominal em um conjunto de sinais binários que o algoritmo de árvore pode utilizar como pontos de ramificação booleanos.
Por fim, foram eliminadas 12 colunas do dataset que representavam risco de vazamento de dados (data leakage) ou sobreposição. Variáveis como causa_acidente, tipo_acidente e ilesos foram descartadas por descreverem consequências ou circunstâncias do sinistro que, em um cenário real de previsão, não estariam disponíveis antes do evento ocorrer, o que tornaria o modelo artificialmente perfeito no treino e completamente inútil na prática operacional.
Figura 17 - G10: Mapa de Calor (Correlação de Pearson)
  

Fonte: Elaborado pelo autor (2026).
Após a criação das variáveis é necessário avaliar, o Mapa de Calor de Correlação de Pearson, apresentado na Figura 17, em que consolida matematicamente as relações entre as variáveis numéricas do dataset pós-processado, funcionando como um raio-x das interdependências estruturais da base, tal gráfico ajuda a validar e a explicar diversas decisões tomadas ao longo do pipeline realizado.
Analisando é possível perceber, que a maior correlação observada é entre feridos_leves e feridos (0,86), o que por sua vez, é esperado e matematicamente faz sentido, dado que feridos é uma variável composta que agrega feridos_leves e feridos_graves. Esse excesso de variáveis redundantes confirmou a necessidade de eliminação de colunas na etapa de redução de dimensionalidade, pois manter ambas as variáveis introduziria ruído desnecessário ao modelo, podendo gerar perdas de desempenho.
Adicionalmente, um segundo agrupamento relevante é o par pessoas e feridos (0,54), e pessoas e feridos_leves (0,46), sinalizando que o volume de ocupantes envolvidos no sinistro possui correlação com o número de vítimas, o que faz muito sentido e está de acordo com a realidade, concluímos que mais pessoas no evento implica maior probabilidade de ferimentos.
Um achado que apoia a criação das variáveis sazonais e temporais e analiticamente mais relevante, entretanto, reside justamente nas correlações baixíssimas das variáveis is_feriado e is_final_semana com as demais. Ambas apresentam valores próximos a zero com todas as outras variáveis numéricas, o que à primeira vista poderia sugerir irrelevância preditiva. Contudo, essa interpretação seria um erro metodológico. Correlação de Pearson mede exclusivamente relações lineares, e o impacto de um feriado ou final de semana sobre a sinistralidade não é linear, mas sim condicional e não-linear, a combinação de feriado com chuva em uma curva de pista simples produz um risco desproporcional que nenhuma correlação bivariada capturaria. É exatamente para esse tipo de interação que o Random Forest, através do Feature Subspace, foi projetado, capturando sinergias que a matriz de Pearson é estruturalmente incapaz de revelar. Portanto, o gráfico da Figura 17 não invalida essas variáveis, mas pelo contrário, ele justifica a escolha de um algoritmo não-linear para explorar seu potencial preditivo real.
Por fim, a ausência de correlações espúrias ou inesperadamente altas entre variáveis independentes distintas atesta a qualidade do pré-processamento realizado, confirmando que o dataset final está livre de vazamento de dados e apto para a próxima fase, de modelagem.


3.2.3.3 Tratamento e Integração da Base do DNIT


Primeiramente, a integração com os dados de infraestrutura do DNIT representa um grande diferencial deste projeto, além de ser um grande desafio técnico. A base do DNIT, que contém índices críticos como o ICC (Índice de Condição de Capa) e o ICP (Índice de Condição do Pavimento), possui cobertura temporal iniciada em 2021, gerando uma lacuna de dois anos em relação ao início da série histórica da PRF (2019). Para contornar esse problema sem introduzir imputação pela média, que por sua vez iria contaminar a variância e distorceria os dados, optou-se por aplicar a técnica de um Backfill Estratégico, ou retro-preenchimento temporal, que copia as medições de 2021 para os anos anteriores de 2019 e 2020. A justificativa é baseada na premissa estabelecida que as condições estruturais de um pavimento variam de forma lenta e gradual, sendo normal e razoável assumir que o estado de uma via em 2021 representa uma aproximação válida para a sua condição nos anos imediatamente anteriores.
Para os registros em que a integração do DNIT com a PRF não resultou em correspondência, optou-se pela imputação com a constante sentinela -1.0 nos índices contínuos ausentes, em vez de recorrer à média ou mediana. Essa decisão é a melhor metodologicamente para o Random Forest, por operar através de partições binárias, o modelo processa o valor -1.0 como uma categoria diferente, aprendendo que a ausência de monitoramento sistemático do DNIT é, por si só, um indicador de risco.


3.2.3.4 Junção Intervalar e Resolução da Armadilha Zero-Inflated


Passando agora para efetivamente a fusão entre as duas bases, foi realizada através de um Interval Join do tipo ASOF, utilizando como chaves de junção o ano e o código da rodovia, a respectiva BR, alocando o quilômetro exato do sinistro dentro do intervalo contínuo [km_inicial, km_final] correspondente no cadastro do DNIT. Essa estratégia, executada pelo Pandas, resultou em correspondência geográfica para 63,95% da base histórica, enriquecendo os registros com os índices de condição do pavimento sem recorrer à explosão combinatória que um merge convencional por quilômetro decimal exato produziria, o que com um grande nível de certeza esgotaria a memória disponível para continuidade do projeto.
Com isso, foi resolvida a integração, e ao mesmo tempo emergiu o principal risco estatístico encontrado no projeto, a Armadilha Zero-Inflated. A base de 473.734 eventos individuais, por sua natureza granular, concentrava uma proporção enorme de registros com zero ou uma ocorrência por trecho em determinado dia, um cenário de inflação de zeros que comprometeria qualquer modelo de regressão treinado diretamente sobre os eventos brutos. A solução adotada foi o Bucketing de espaço temporal, a base foi agregada em trechos de 10 km mensais, criando 139.590 cenários preditivos únicos com uma variável alvo contínua e densa, quantidade_acidentes. Essa transformação é o que permite ao modelo estimar o volume de acidentes em um trecho específico para um determinado mês, convertendo eventos discretos e esparsos em inteligência estratégica, esses passos realizados representam a materialização do objetivo central do projeto, com a conclusão da base para treinamento.
