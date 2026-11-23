3.2.2 Entendimento dos Dados


        A segunda fase do procedimento CRISP-DM, consiste na famosa análise exploratória do dataset, momento em que é feito análises estatísticas a fim de entender estruturalmente os dados e a descoberta de características que muitas vezes não são vistas à primeira vista. 
Ademais, tais informações descobertas nessa etapa são utilizadas por todo o projeto, apoiando decisões técnicas críticas, algumas delas já foram citadas ao longo deste trabalho, como a escolha do algoritmo Random Forest Regressor citada no capítulo 3.1.5.
Primeiramente, o processo se iniciou com a realização de análises a fim de compreender o complexo cenário dos acidentes nas rodovias federais brasileiras. Antes de qualquer exploração estatística, contudo, foi necessário diagnosticar a qualidade da base bruta da PRF, identificando o percentual de valores nulos por variável, conforme apresentado na Figura 7.








Figura 7 – EDA 1: Porcentagem de Valores Nulos por Variável na Base Bruta.
  

Fonte: Elaborado pelo autor (2026).
A análise da Figura 7 revelou que as variáveis uop e delegacia concentram os maiores índices de ausência, com aproximadamente 8,2% e 4,3% de valores nulos, respectivamente. Em contrapartida, variáveis críticas para o modelo preditivo, como tipo_acidente e classificacao_acidente, apresentaram taxas de nulidade inferiores a 0,5%, o que confirma a integridade das informações centrais da base. Essa etapa fundamentou as decisões de tratamento de dados da fase de pré-processamento, onde as colunas com alto percentual de nulos e baixa relevância analítica foram descartadas, enquanto as variáveis com nulidade residual foram tratadas por remoção pontual de registros. Com a qualidade da base diagnosticada, a análise avançou para a compreensão do fenômeno em si. A Figura 8 apresenta os dez tipos de acidentes mais frequentes registrados na base histórica.










Figura 8 – EDA 2: Top 10 Tipos de Acidentes Relatados (Dado Bruto).
  

Fonte: Elaborado pelo autor (2026).
Com isso, percebe-se que a colisão traseira destaca-se como o tipo mais recorrente, ultrapassando 85.000 ocorrências, seguida pela saída de leito carroçável com aproximadamente 72.000 registros e pela colisão transversal com cerca de 60.000 casos. Esses três tipos, juntos, respondem por uma parcela expressiva do volume total de sinistros. Tal distribuição evidencia que a maioria dos acidentes está associada ao comportamento do condutor em situações de tráfego convencional, onde a distância de segurança, a atenção e o controle direcional são determinantes. 
Foi observado que os acidentes em sua maior parte são causados por fatores humanos, mas são potencializados por fatores estruturais da via, fatores estruturais da via, visto que a precariedade desta eleva a taxa de sinistros.
Aprofundando a análise sobre o comportamento humano como um dos fatores condicionantes, a Figura 9 detalha as principais causas registradas pelos agentes da PRF.




Figura 9 – EDA 3: Principais Causas Condicionantes dos Acidentes.
  

Fonte: Elaborado pelo autor (2026).
Com o entendimento da Figura 9 percebe-se que a falta de atenção à condução, a reação tardia ou ineficiente do condutor e a ausência de reação do condutor lideram o ranking de causas primárias, cada uma superando 40.000 ocorrências na série histórica analisada. Juntas, essas três categorias evidenciam que o fator humano é o principal vetor de risco nas rodovias federais brasileiras, respondendo por uma proporção majoritária dos sinistros. Velocidade incompatível e o acesso à via sem observar outros veículos completam o grupo das cinco causas mais frequentes, reforçando a predominância de comportamentos inadequados na condução. 
Além disso, a ingestão de álcool pelo condutor, embora em menor volume absoluto, aparece em sétimo lugar com aproximadamente 20.000 registros, sendo um fator de particular gravidade dado seu impacto na severidade dos acidentes. Essa distribuição de causas é um insumo estratégico fundamental para os stakeholders operacionais, pois orienta as campanhas educativas e as operações de fiscalização da PRF para os comportamentos de maior impacto estatístico.
A identificação das rodovias mais críticas em termos de letalidade acumulada é apresentada na Figura 10.




Figura 10 – EDA 4: Top 10 Rodovias (BRs) com Maior Letalidade Acumulada.


  

Fonte: Elaborado pelo autor (2026).
Analisando o gráfico, percebe-se que as rodovias BR-116 e BR-101 destacam-se com folga como as mais letais do país no período analisado, acumulando aproximadamente 5.000 e 4.700 mortes, respectivamente. Essa concentração é esperada dado que ambas são as principais rodovias longitudinais do Brasil, atravessando múltiplos estados e concentrando enormes volumes de tráfego de cargas e passageiros. As demais rodovias do ranking, como BR-153, BR-163, BR-381 e BR-040, apresentam letalidade acumulada entre 1.300 e 1.800 mortes, configurando um segundo grupo de vias prioritárias para intervenção. Essa hierarquização das rodovias por letalidade é um dos principais outputs estratégicos do projeto, pois permite que órgãos como a PRF e o DNIT priorizem geograficamente os investimentos em duplicação, manutenção e sinalização com base em evidências quantitativas.








Figura 11 – EDA 5: Dispersão Bruta – Ocupantes vs. Veículos Envolvidos (Amostra 10k).
  

Fonte: Elaborado pelo autor (2026).


O gráfico de dispersão foi gerado a partir de uma amostra de 10.000 registros aleatórios da base, evidenciando que a grande maioria dos acidentes envolve entre 1 e 5 veículos e até 10 pessoas, o que caracteriza o perfil típico do sinistro rotineiro nas rodovias federais. Os pontos classificados como "Com Vítimas Fatais" (verde) aparecem distribuídos ao longo de toda a faixa de veículos envolvidos, mas com concentração em acidentes de menor número de veículos, reforçando que um único veículo com muitos ocupantes, como ônibus ou vans, pode gerar elevada letalidade. Os outliers extremos, com mais de 40 pessoas envolvidas, correspondem exatamente ao fenômeno de cauda longa discutido estatisticamente no capítulo 3.1.4, confirmando que esses eventos, embora raros, são estruturais e não podem ser descartados da base.
A distribuição gravimétrica dos acidentes, consolidada na Figura 12, fornece uma visão da proporção entre os diferentes níveis de severidade.


Figura 12 – EDA 6: Classificação Gravimétrica dos Acidentes (Distribuição por Severidade).
  

Fonte: Elaborado pelo autor (2026).
A análise do gráfico de pizza revela que 75,9% dos acidentes registrados resultaram em vítimas feridas, enquanto 17,0% foram classificados como sem vítimas e 7,1% como com vítimas fatais. Essa distribuição é um insumo direto para a priorização de políticas públicas: enquanto a maioria dos sinistros gera demanda sobre o sistema de saúde com feridos, a fração fatal, embora minoritária em volume, representa o impacto humano mais grave e irreversível. Do ponto de vista do modelo preditivo, essa distribuição reforça o cenário de classes desbalanceadas, onde a concentração em "Com Vítimas Feridas" poderia enviesar um classificador ingênuo, justificando a abordagem de regressão adotada neste projeto em vez de uma classificação direta.
A composição do traçado das vias onde os acidentes ocorreram, apresentada na Figura 13, adiciona a dimensão de infraestrutura à análise exploratória.
Figura 13 – EDA 7: Composição do Traçado da Via Original.
  

Fonte: Elaborado pelo autor (2026).
Pelo gráfico de pizza fica claro que trechos em pista reta concentram 58,4% dos acidentes registrados, seguidos por "Múltiplos Traçados/Outros" com 20,6% e curvas com 13,5%. O elevado percentual de acidentes em retas pode parecer contraditório à primeira vista, mas é explicado pela maior proporção de extensão em reta da malha rodoviária brasileira e pelo comportamento de excesso de velocidade que esses trechos tendem a induzir. As curvas ao contrário do senso comum representam apenas 13,5% dos casos, entretanto, concentram acidentes de maior gravidade, pois a perda de controle em curvas frequentemente resulta em saídas de pista e colisões frontais. Interseções de vias, declives e a combinação reta-declive, embora de menor representatividade percentual, são tipologias que exigem atenção específica na modelagem, pois concentram riscos desproporcionais em relação à sua extensão na malha.
Portanto, percebe-se que as rodovias federais brasileiras possuem um cenário de sinistralidade estrutural e multifatorial, onde fica claro que nenhum fator isolado explica a totalidade dos acidentes, mas sim a convergência de vulnerabilidades humanas, operacionais e de infraestrutura. A análise exploratória revelou que o fator humano é o principal vetor de risco, com falta de atenção, reação tardia e velocidade incompatível respondendo pela maioria das causas registradas, enquanto a infraestrutura precária atua como amplificador dessa fragilidade, especialmente em pista simples e em trechos sem duplicação. A concentração geográfica de ocorrências nos estados do Sul e Sudeste, liderados por Minas Gerais, Santa Catarina e Paraná, reflete a maior densidade de tráfego nessas regiões, ao passo que a letalidade desproporcional das BR-116 e BR-101 evidencia que volume de tráfego aliado à ausência de investimento em duplicação forma uma combinação historicamente fatal. 
Ademais, a evolução temporal da série histórica evidenciou não se tratar de um problema em vias de solução, o número de feridos cresceu consistentemente entre 2019 e 2024, reforçando com evidência uma das premissas do projeto, de que políticas públicas reativas têm se mostrado insuficientes diante da magnitude do desafio. Diante desse complexo retrato, fica evidente que a transformação desses dados em inteligência preditiva, proposta por este trabalho, não é apenas uma contribuição técnica, mas uma resposta concreta a um problema de saúde pública que ceifa milhares de vidas e consome dezenas de bilhões de reais anualmente.




________________


