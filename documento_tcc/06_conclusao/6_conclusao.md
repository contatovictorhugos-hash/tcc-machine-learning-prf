6 CONCLUSÃO


A segurança viária em rodovias federais permanece como um dos nós mais dramáticos da administração pública brasileira, pois vidas são ceifadas diariamente no asfalto brasileiro. Por trás dos números consolidados nos boletins da Polícia Rodoviária Federal, persistem tragédias familiares e prejuízos econômicos que drenam dezenas de bilhões de reais da sociedade ano após ano. O poder público, historicamente condicionado a agir apenas após a colisão, opera na maior parte do tempo em regime reativo, mobilizando viaturas e equipes de resgate quando o dano já se consumou. Este Trabalho de Conclusão de Curso desafiou essa postura de inércia, a investigação buscou demonstrar como a integração de Ciência de Dados, Aprendizado de Máquina e inteligência geoespacial pode erguer uma ponte sólida entre repositórios abertos de dados governamentais e a tomada de decisão preventiva no Estado.
Ao conectar os registros operacionais de sinistros da PRF (2019 a 2025) aos cadastros técnicos de infraestrutura viária do DNIT, o projeto estruturou uma esteira analítica orientada pelo framework CRISP-DM. Longe de tratar os algoritmos preditivos como caixas-pretas de oráculos, o esforço investigativo priorizou a transparência metodológica, a interpretabilidade dos fatores de risco e a formulação de diretrizes acionáveis para as políticas públicas de transporte. O percurso revelou tanto o vigor dos modelos de árvores de decisão para prever volumes de sinistralidade quanto os desafios matemáticos que emergem quando se tenta modelar a realidade complexa e assimétrica das rodovias brasileiras.


6.1 Síntese do Cumprimento dos Objetivos


O objetivo geral do trabalho foi plenamente atingido ao se desenvolver, testar e homologar uma arquitetura preditiva robusta capaz de estimar o volume mensal de acidentes em segmentos da malha federal. Para isso, o modelo Random Forest V5 consolidou-se como o resultado central da pesquisa. Diferentemente de abordagens puramente teóricas que repartem conjuntos de treino e teste de modo aleatório caindo no vício de data leakage temporal, o Modelo V5 submeteu-se a um particionamento cronológico estrito: foi calibrado com 5 anos de 2019 a 2023 e testado frente aos anos de 2024 e 2025. Ao confrontar o futuro desconhecido, o modelo alcançou um coeficiente de determinação R2 de 0,9726 e um Erro Médio Absoluto (MAE) de 0,4181 acidentes por trecho de 10 km ao mês, superando com folga os parâmetros estabelecidos no RNF-4 (R2 > 0,90 e MAE<1,0).
Os objetivos específicos traçados logo no início do planejamento da pesquisa foram concretizados ao longo das etapas do pipeline analítico:
Ingestão e Harmonização Multi-fonte: O pipeline de ETL processou mais de 450 mil boletins de ocorrência da PRF e os integrou aos levantamentos de condição estrutural do DNIT por meio de junções intervalares temporais e espaciais (ASOF join). Para contornar a esparsidade extrema da escala métrica pontual, a malha rodoviária foi agregada em buckets espaço-temporais homogêneos de 10 quilômetros por mês, assegurando consistência tabular para o treinamento algorítmico.
Interpretabilidade e Avaliação Causal da Infraestrutura: A aplicação combinada de importância por impureza (MDI), valores SHAP (Shapley Additive Explanations), Gráficos de Dependência Parcial (PDP) e testes A/B Ceteris Paribus decifrou a mecânica interna do Random Forest. Os experimentos comprovaram a relação não linear entre a qualidade do pavimento e a sinistralidade. A otimização global via algoritmo Argmax identificou na BR-40 (KM 110 ao 120) o trecho prioritário nacional, no qual a recuperação asfáltica (ICP de 5 para 70) aponta uma redução potencial de -17,04% na ocorrência de acidentes.
Simulações de Cenários Operacionais e Validação Empírica: Foram simulados cinco cenários táticos balizados pela mediana das estimativas, neutralizando o efeito de caudas longas. As simulações comprovaram que o risco de sinistros salta +3,1% nos finais de semana, cresce +7,3% em trechos sinuosos e escala +23,1% em interseções viárias, culminando em uma alta de +19,5% no cenário de `tempestade perfeita`. No estudo de caso real do KM 70 da BR-101 (trecho com viaduto e rotatória), o modelo estimou 48,7 acidentes no mês frente aos 54 efetivamente registrados, demonstrando grande proximidade com a realidade operacional.
Disponibilização em Plataforma de Business Intelligence: Atendendo integralmente aos requisitos RF-7 e RNF-1, as estimativas foram preservadas na escala original das contagens para poderem ser exportadas para painéis executivos no Microsoft Power BI. A ferramenta entrega à gestão pública mapas de calor de criticidade viária, histórico descritivo, simulador interativo de cenários What-If e ranking automatizado de intervenção em pavimentos para o DNIT.


6.2 Contribuições para a Gestão Pública


A principal virtude desta pesquisa reside em fornecer ferramentas concretas para a transformação cultural da segurança viária no Brasil. Conforme advertem Provost e Fawcett (2013), a maturidade analítica de uma organização só atinge sua plenitude quando os dados deixam de servir a relatórios de autópsia e passam a balizar escolhas estratégicas prévias. O setor público brasileiro não carece de registros, carece de capacidade preditiva articulada. O presente trabalho oferece uma contribuição palpável para preencher essa lacuna em duas instituições chaves para a malha rodoviária: a Polícia Rodoviária Federal e o Departamento Nacional de Infraestrutura de Transportes.
Para a Polícia Rodoviária Federal, o modelo disponibiliza uma régua quantitativa de antecipação. Em vez de definir rondas a partir do hábito ou de impressões empíricas dispersas, o comando operacional pode alocar viaturas e posicionar radares móveis guiando-se pela probabilidade espacial de risco. Como revelou a análise dos cenários táticos, cruzamentos e curvas sinuosas combinados com fluxo de finais de semana demandam policiamento ostensivo prioritário. Saber onde e quando a probabilidade de colisões se eleva autoriza o remanejamento cirúrgico do efetivo policial, otimizando o gasto com combustíveis e ampliando a presença dissuasória da autoridade nos pontos de maior perigo.
Para o DNIT, as contribuições recaem diretamente sobre a eficiência na aplicação dos escassos recursos do orçamento federal. A análise por Dependência Parcial (PDP) desfez uma premissa intuitiva: a curva em cotovelo identificada na relação entre o Índice de Condição do Pavimento (ICP) e os sinistros prova que recuperar um trecho em estado Péssimo para Regular (ICP<50) traz um retorno marginal substancialmente superior em vidas salvas (-0,120 acidentes/mês por km) do que recapear vias que já operam em condições aceitáveis (-0,080 acidentes/mês por km). Esse achado estabelece um critério objetivo de Retorno Social do Investimento (ROMI) para a engenharia viária. O algoritmo Argmax testado no estudo de caso da BR-40 exemplifica como a inteligência computacional pode criar uma fila técnica de prioridades para obras asfálticas, baseado inteiramente na ciência de dados afastando de cena outras pressões para priorizar trechos onde o asfalto novo possui maior poder de preservar vidas.
Há ainda uma contribuição institucional intangível, porém indispensável para a análise, a integração técnica entre PRF e DNIT. Historicamente, ministérios e autarquias federais operam seus dados em compartimento, ou silos. O policial preenche o boletim na pista, o engenheiro mede o asfalto meses depois em outra planilha, mas raramente esses dados se cruzam e são analisados como um só. Este estudo provou a viabilidade técnica e a riqueza analítica de fundir esses dois universos em uma camada de dados única, inaugurando um precedente metodológico para políticas integradas de mobilidade e segurança de transportes no governo federal.


6.3 Limitações da Pesquisa


O desenvolvimento de modelos estatísticos sobre dados abertos da administração pública impõe um dos maiores desafios na ciência de dados. Durante a construção do Modelo V5, a busca pela superação de barreiras estruturais da base revelou uma dinâmica clássica da Ciência de Dados, a tentativa de contornar uma limitação matemática frequentemente introduz efeitos colaterais na etapa seguinte do pipeline.
A distribuição bruta dos boletins de ocorrência da PRF apresenta esparsidade extrema quando analisada em escala métrica linear contínua. Em um determinado dia e quilômetro isolado de rodovia, a probabilidade pontual de colisão é virtualmente nula, gerando uma inflação massiva de zeros conhecido como zero-inflation, conceito já explicado anteriormente, superior a 99% das observações. Para viabilizar a convergência de algoritmos baseados em árvores de decisão, estruturou-se a agregação da malha em buckets espaço-temporais de 10 quilômetros por mês. A abordagem tomada atingiu com sucesso o objetivo imediato de densificar a variável resposta. Contudo, essa consolidação acarretou dois desdobramentos metodológicos críticos, não previstos.
O primeiro desdobramento consiste no viés de sobrevivência decorrente da perda dos trechos seguros. O agrupamento temporal foi executado aplicando operações de sumarização (groupby) sobre a tabela histórica de sinistros registrados pela PRF. Consequentemente, apenas trechos e meses que registraram ao menos um acidente integraram a matriz final de treinamento, pois no dataset da PRF contém apenas ocorrências de sinistros e não os meses que os trechos não tiveram acidente algum. Dessa forma, variável alvo assumiu valor mínimo unitário, excluindo da modelagem os milhares de segmentos federais que operaram com total segurança ao longo do ano. O modelo não estima a probabilidade incondicional de acidentes na malha rodoviária federal, ele opera estritamente condicionado à preexistência de sinistralidade. O sistema perde, portanto, a sensibilidade matemática para discriminar ativamente os fatores que mantiveram um segmento preservado sem nenhuma ocorrência ao longo de doze meses consecutivos.
O segundo desdobramento diz respeito a uma decomposição contábil das variáveis geométricas que atuou como facilitador estatístico nas métricas de desempenho. Na fase de engenharia de atributos, as variáveis categóricas de traçado da via e de calendário foram computadas no nível do evento e consolidadas no bucket mensal por meio de somatórios de ocorrências. Ao receber essas variáveis somadas como preditores, o Random Forest identificou uma quase identidade contábil, a soma das colisões em retas, curvas e cruzamentos reconstitui perfeitamente o próprio volume total de acidentes daquele mês, dessa forma o modelo identificou um atalho para o atingimento da meta.
Esse mecanismo esclarece a razão pela qual o Modelo V5 alcançou um coeficiente de determinação aparentemente excepcional mesmo sob uma partição temporal rigorosa (treino 2019–2023 / teste 2024–2025). As colunas explicativas do conjunto de teste já continham a contagem agregada de sinistros daquele mês. O modelo operou com maestria como um reconciliador de perfis físicos de sinistros ocorridos.
Outro ponto de desafio, foi a integração das medições de engenharia viária do DNIT aos registros da PRF, que revelou assimetrias severas na infraestrutura de dados governamentais.
Lacuna Amostral de 82,3%: Apenas 17,7% dos buckets georreferenciados possuíam avaliação estrutural contemporânea do pavimento cadastrada pelo DNIT. A esmagadora maioria da malha federal não dispunha de séries contínuas para o Índice de Condição da Superfície (ICP) e o Índice de Gravidade Global (IGG).
Com esse fato, foi tomada a decisão para não distorcer a variância populacional por meio de imputações artificiais de média ou mediana, os trechos desprovidos de monitoramento receberam a constante sentinela -1.0. A topologia de divisão das árvores do Random Forest isolou esse valor em ramos próprios. Entretanto, esse artifício transformou a feature contínua em um separador binário de governança: o modelo aprendeu a distinguir trechos monitorados pelo DNIT daqueles negligenciados, atenuando a detecção suave da curva de degradação física do asfalto.
O ofuscamento de Atributos, a presença de somatórios contábeis de acidentes nas colunas de traçado fez com que o algoritmo concentrasse cortes de impureza (MDI) na geometria viária, que respondeu por mais de 60% do ganho de informação, devido ao atalho que ele identificou. Variáveis contínuas de pavimento (icp, icc, icm) receberam participações residuais entre 1% e 2%, limitando a sensibilidade pura do asfalto nos testes preditivos.
Por fim, identificaram-se restrições na sensibilidade das simulações de cenários (Ceteris Paribus). Ao realizar as simulações de cenários e fixar um atributo de cenário como is_tracado_via_Curva = 1, o modelo treinado sob somas absolutas e fortemente impactado pelo atalho, interpretou a entrada como a ocorrência de apenas uma colisão naquela condição, comprimindo as previsões simuladas em uma faixa estreita entre 1,36 e 1,60 acidentes por trecho/mês. 


6.4 Trabalhos Futuros
As fronteiras encontradas nesta pesquisa não encerram a discussão; abrem, pelo contrário, um roteiro claro para o amadurecimento científico da modelagem preditiva em segurança viária. Propõem-se quatro frentes complementares para investigações posteriores:
A primeira iniciativa consiste na construção de uma matriz cartesiana exaustiva com presença de zeros reais. O pipeline deve abandonar a estruturação restrita aos sinistros ocorridos e adotar uma grade espaço-temporal completa. Essa grade será formada pelo produto cartesiano de todos os trechos homogêneos da malha rodoviária federal multiplicados pelos doze meses do ano civil. Os segmentos que não registrarem ocorrências em determinado período receberão explicitamente a contagem 
Y=0
Y=0. Esse ajuste capacitará o algoritmo a discriminar os fatores determinantes da segurança viária e da imunidade a sinistros, superando o viés de sobrevivência.
A segunda frente requer o desacoplamento categórico dos atributos cadastrais. As características físicas do leito viário devem emanar de dados cadastrais estáticos e perenes do DNIT, jamais de somatórios de sinistros no bucket. Recomenda-se substituir as variáveis dicotômicas somadas por atributos geométricos contínuos reais:
Grau e raio médio de curvatura do segmento (medido em metros);
Rampa longitudinal e inclinação média de declive e aclive (em percentual);
Largura útil de pista de rolamento e presença de acostamento pavimentado;
Contagem de acessos comerciais e interseções em nível por quilômetro.
Dessa forma, os preditores representarão a topografia real da rodovia de forma imutável e exógena, existindo antes e independentemente de qualquer sinistro acontecer.
A terceira diretriz direciona-se à experimentação com arquiteturas estatísticas nativas para dados de contagem com excesso de zeros. Em vez de forçar o Random Forest padrão sobre distribuições truncadas, sugere-se investigar:
Modelos Hurdle e Zero-Inflated: Abordagens como Zero-Inflated Poisson (ZIP) e Zero-Inflated Negative Binomial (ZINB) decompõem o problema em dois momentos estatísticos distintos. Primeiro, um regressor logístico estima a probabilidade de um trecho sofrer ao menos um acidente (
P(Y>0)
P(Y>0) versus 
P(Y=0)
P(Y=0)); em seguida, um modelo de contagem truncada calcula o volume de sinistros condicionado à quebra do zero.
Funções de Perda Tweedie em Algoritmos de Boosting: Implementação de LightGBM ou XGBoost com distribuição Tweedie, cuja formulação matemática acomoda naturalmente distribuições com massa pontual concentrada em zero combinada a caudas contínuas estritamente positivas.
A quarta frente reside na integração sensorial contínua com sensores climáticos e contadores de tráfego. Propõe-se conectar os buckets georreferenciados às séries horárias das estações automáticas do Instituto Nacional de Meteorologia (INMET), alimentando o modelo com grandezas físicas reais de precipitação acumulada (milímetros por hora), rajadas de vento e índice de visibilidade horizontal. Da mesma forma, torna-se essencial incorporar o Volume Diário Médio (VDM) aferido por praças de pedágio e radares do DNIT. A inclusão da contagem volumétrica de tráfego permitirá converter previsões de volume absoluto em taxas normalizadas de exposição ao risco por milhão de veículos-quilômetro percorridos, elevando a precisão das decisões operacionais do Estado a um patamar inédito.




















































































