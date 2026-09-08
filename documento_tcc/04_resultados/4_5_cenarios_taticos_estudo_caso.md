4.5 Simulações de Cenários Táticos e Estudo de Caso Operacional (PRF)


Enquanto a análise de infraestrutura do DNIT volta-se a intervenções estruturais de médio a longo prazo, buscando responder perguntas no que tange às maiores prioridades de melhorias com base no retorno sobre o capital investido, o planejamento da Polícia Rodoviária Federal (PRF) exige respostas dinâmicas e de curto prazo. A gestão policial opera sobre variáveis contextuais: escalonamento de plantões, fiscalização com etilômetro aos finais de semana e posicionamento de radares móveis em condições de baixa visibilidade. Nesse cenário complexo, estimar a quantidade de acidentes em determinados períodos torna-se fundamental para maximizar a cobertura policial. 
Para subsidiar essas decisões táticas, o Modelo V5 foi submetido a simulações de cenários operacionais sobre a totalidade da malha federal, avaliando o comportamento do risco sob diferentes combinações de calendário, traçado e clima.


4.5.1 Metodologia por Mediana e Avaliação dos Cenários Táticos


Nas simulações de cenários em escala nacional, adotou-se a mediana das predições como métrica primária de avaliação 1,8978 acidentes/mês no baseline global, mantendo a média aritmética 3,4182 como referência secundária. A escolha decorre da severa assimetria positiva da base histórica. A média é artificialmente inflada por uma minoria de pontos críticos com mais de 50 acidentes mensais. A mediana reflete o comportamento do trecho rodoviário típico brasileiro, assegurando que o planejamento tático não seja distorcido por valores extremos.
A Tabela 6 consolida os resultados dos cinco cenários táticos simulados no Modelo V5.
Tabela 6 – Impacto das Variáveis Exógenas no Risco de Acidentes


Situação Testada
	Condição Normal (Base)
	Condição Alterada (Teste)
	Variação de Acidentes
	Impacto no Risco
	O que isso significa para a PRF?
	Fim de Semana
	Dia Útil (1,83 ac/mês)
	Final de Semana (1,89 ac/mês)
	+0,06
	+3,1%
	Aumentar fiscalização de sexta a domingo.
	Traçado da Pista
	Trecho em Reta (1,06 ac/mês)
	Trecho em Curva (1,14 ac/mês)
	+0,08
	+7,3%
	Fiscalizar velocidade antes de curvas.
	Clima
	Céu Claro (1,90 ac/mês)
	Pista com Chuva (1,87 ac/mês)
	-0,03
	-1,5%
	O fluxo reduz a velocidade por cautela.
	Turno do Dia
	Pleno Dia (1,90 ac/mês)
	Plena Noite (1,87 ac/mês)
	-0,03
	-1,3%
	Menor fluxo, mas maior severidade.
	Efeito Extremo
	Cenário Ideal (1,08 ac/mês)
	Tempestade Perfeita (1,29 ac/mês)
	+0,21
	+19,5%
	O risco dispara em 20% em condições extremas.
	Fonte: Elaborado pelo autor (2026).


A Tabela 6 sintetiza os resultados do modelo quando alteramos as variáveis exógenas, que no caso são parâmetros que podem ser ajustados de um dia para o outro no ambiente de produção. O objetivo dessas simulações foi responder a uma pergunta prática da gestão pública: qual é o impacto isolado de cada fator sobre a ocorrência de acidentes?
Ao isolar os fatores um a um (Cenários 1 a 4), as oscilações no volume de acidentes variam em faixas moderadas, situando-se entre 1% e 7%. No entanto, quando múltiplos fatores de risco ocorrem simultaneamente, como observado no Cenário 5, que combina pista reta, final de semana, chuva, noite e feriado, o modelo estima um salto expressivo de +19,5% na quantidade de sinistros em relação ao cenário ideal. Isso comprova que o risco rodoviário não cresce de forma isolada, mas se multiplica quando fatores climáticos, temporais e de geometria coincidem. A Figura 24 ilustra graficamente o comparativo entre a condição normal e a condição de risco para cada uma dessas simulações.


Figura 24 — Comparativo geral dos cinco cenários táticos simulados pela mediana das predições.
  
Fonte: Elaborado pelo autor (2026).












A leitura analítica dos cinco cenários revela diretrizes operacionais diretas:


   * Cenário 1 (Final de Semana X Dia Útil): A sinistralidade mediana eleva-se em +3,1% aos sábados e domingos, de 1,8307 para 1,8881. O aumento reflete a intensificação de fluxos recreativos, consumo de álcool e viagens de longa distância, justificando a concentração de efetivo e operações de alcoolemia nesses dias.
   * Cenário 2 (Traçado da Via): Pistas em curva registram mediana +7,3% superior às retas, de 1,0600 para 1,1437, enquanto interseções de vias disparam para 1,3048 (+23,1%). O risco concentrado em cruzamentos e curvas acentuadas exige fiscalização eletrônica pontual de velocidade nesses pontos.
   * Cenário 3 (Meteorologia) e Cenário 4 (Fase do Dia): Em dias chuvosos ou turnos noturnos, observa-se uma sutil redução na mediana da contagem de sinistros (-1,5% e -1,3%), associada à redução voluntária da velocidade média e à diminuição do fluxo total de veículos de passeio. Contudo, relatórios da PRF indicam que os sinistros noturnos e sob pista molhada apresentam maior índice de letalidade e severidade física.
   * Cenário 5 (Tempestade Perfeita vs. Cenário Ideal): A combinação simultânea de todos os fatores de risco (reta, final de semana, chuva, plena noite e feriado) eleva a mediana estimada para, de 1,2865, contra 1,0761 no cenário ideal (curva, dia útil, céu claro e pleno dia). Trata-se de um salto de +19,5% no risco operacional, momento crítico em que protocolos de contingência e presença máxima de patrulhas devem ser ativados.


Portanto, fica claro a capacidade do modelo ao se adaptar às mudanças frequentes que podem ocorrer em trechos na malha rodoviária brasileira. Transmitindo aos órgãos públicos uma estimativa de vítimas em cada cenário, dessa forma o poder público consegue se planejar de forma assertiva e eficiente. O'Que antes era reativo e baseado única e exclusivamente na experiência dos agentes, passa a ser ancorado em dados e análises.
4.5.2 Estudo de Caso Real: BR-101 (KM 70)
Para validar a capacidade do modelo em reproduzir a dinâmica de um trecho de altíssima complexidade no mundo real, selecionou-se um KM aleatório para a demonstração, o escolhido foi KM 70 da BR-101. O segmento concentra uma interseção rodoviária de grande porte com rotatória e viaduto, acumulando um histórico de 2.965 acidentes na base da PRF.
Na simulação desse trecho, adotou-se uma metodologia estrita, a geometria real da via foi 100% preservada, a fim de simular com o máximo de exatidão um cenário do dia a dia da aplicação, com isso alterou-se exclusivamente às variáveis contextuais de tempo e clima. A Figura 25 apresenta o painel do estudo de caso, comparando as predições do Modelo V5 com as ocorrências reais observadas no pico histórico de Dezembro de 2023.


Figura 25 — Painel do estudo de caso na BR-101 (KM 70): histórico mensal, predições por cenário e comparação com dados reais.
  

Fonte: Elaborado pelo autor (2026).
Em análise do painel, é possível observar que em Dezembro de 2023, o trecho registrou 54 acidentes reais. Alimentado com os dados originais do período, o Modelo V5 estimou 48,7 acidentes/mês, demonstrando aderência ao volume efetivamente observado em um dos pontos mais críticos do país.
Em outro cenário no mesmo trecho, ao simular variações puramente contextuais, a predição oscilou de 32,0 acidentes no Cenário Ideal para 32,3 acidentes na Tempestade Perfeita.
Com isso, essa proximidade observada se materializa em uma evidência para a premissa central desta pesquisa, o volume de sinistros em rodovias federais não é um fenômeno aleatório, mas uma dinâmica estruturada e matematicamente previsível. Em um gargalo complexo que entrelaça o fluxo rodoviário de longa distância com o tráfego urbano local, o modelo foi capaz de capturar a sobrecarga da via sem atenuar sua periculosidade. Para a Polícia Rodoviária Federal, essa capacidade preditiva transforma a atuação institucional, substituindo o atendimento reativo a ocorrências pelo dimensionamento antecipado de viaturas, bafômetros e equipes de resgate médico nos meses onde a população está mais exposta a sinistros fatais.


________________
