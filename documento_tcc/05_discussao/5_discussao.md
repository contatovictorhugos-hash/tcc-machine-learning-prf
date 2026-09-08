5 DISCUSSÃO 


Primeiramente é sabido que, reduzir acidentes exige mais do que apenas registrar tragédias após o encerramento do exercício fiscal. Os achados quantitativos revelados pela modelagem preditiva comprovam que a sinistralidade nas rodovias federais obedece a padrões determinísticos que podem e devem ser antecipados pelo Estado. 
A apresentação numérica dos dados consolidados no capítulo anterior só ganha sentido prático quando é confrontado com o desafio real da segurança viária brasileira. Mais do que uma sequência de métricas de acurácia ou coeficientes estatísticos, os resultados obtidos pelo Modelo V5 representam uma resposta direta às dores estruturais diagnosticadas na literatura. O objetivo deste capítulo é dissecar o significado dessas evidências empíricas, avaliando como o rigor metodológico adotado, o isolamento dos efeitos da infraestrutura do DNIT e as simulações táticas da PRF transformam dados espalhados de ocorrências em diretrizes concretas de governança orientada a dados.


5.1 Da Reatividade à Antecipação


Primeiramente fica claro que, os resultados obtidos demonstrados no último capítulo, validam a hipótese central levantada no início do projeto, a Introdução, o uso de Ciência de Dados rompe o status quo atual da gestão de tráfego, que é a inércia reativa que historicamente caracteriza a gestão viária no Brasil. Durante anos, a administração pública atuou apenas após a consolidação da tragédia, cenário que o Laboratório de Inovação e Dados (ÍRIS, 2022) identifica como o principal entrave para que os governos atinjam a maturidade analítica e superem a cultura burocrática, cenário comumente conhecido como “apagar incêndio”. O Anuário da PRF (2025), ao registrar 73.156 sinistros e 6.160 mortes em 2024, atesta a falência desse modelo puramente retrospectivo.
Com isso, a revolução para o modelo orientado a dados,  atende aos princípios teóricos de Provost e Fawcett (2013) e Wegman et al. (2015), segundo os quais o dado bruto só se torna ativo estratégico quando reduz a incerteza de eventos futuros. Ao atingir um R2 de 0,9726 com erro médio inferior a meio acidente por trecho/mês, MAE=0,418, o pipeline comprova que a sinistralidade não é um evento puramente caótico ou imprevisível. Existe um padrão determinismo probabilístico imposto pela geometria da via, pelo calendário e pela qualidade do pavimento, no qual um algoritmo consegue estimar com um alto nível de acerto. 
Portanto, para o Estado brasileiro, que acumula perdas anuais estimadas em R$ 50 bilhões pelo IPEA(2020) e analisando apenas rodovias federais é estimado em R$ 12,9 bilhões segundo a CNT (2023), ter a possibilidade de antecipar a probabilidade de risco significa alocar viaturas e ambulâncias no local exato antes da colisão, transformando gastos emergenciais em investimento preventivo, ocasionando em um saving de vidas e orçamentário para o poder público.


5.2 A Ilusão das Métricas e a Assimetria Real


A comparação entre as cinco arquiteturas testadas traz uma discussão fundamental sobre a aplicação de Machine Learning em séries temporais governamentais. Se a avaliação do projeto tivesse se guiado exclusivamente pela busca do maior coeficiente numérico, a versão V1 teria sido a escolhida devido ao seu R2 de 0,9892. No entanto, essa escolha configuraria uma falha metodológica grave, o data leakage temporal introduzido pela amostragem aleatória convencional (train_test_split).
Em problemas de segurança pública, o tempo é um vetor estritamente cronológico. Permitir que observações de 2024 componham o treinamento para prever 2021 equivale a alimentar o algoritmo com o futuro que ele deveria inferir. A redução controlada das métricas no Modelo V5 (R2 de 0,9726 treinado entre 2019-2023 e testado em 2024-2025) reflete o único desempenho fidedigno para um ambiente real de produção. O modelo comprovou sua capacidade de generalizar diante do desconhecido, respaldando a teoria de ensemble learning de Breiman (2001) e James et al. (2023) sobre a contenção de variância sem sobreajuste conhecido como overfitting mundialmente.
De maneira complementar, a constatação da assimetria positiva severa, através do skewness no dataset da PRF justificou o abandono da média aritmética em favor da mediana como métrica primária de simulação (Mediana Baseline = 1,8978 vs. Média Baseline = 3,4182). Como a maioria dos trechos registra poucos sinistros rotineiros enquanto raros trechos concentram dezenas de ocorrências, utilizar a média como balizador operacional induziria a PRF a superdimensionar efetivos em vias calmas e subdimensionar em pontos de cauda longa. Outro ponto importante foi o tratamento via Bucketing espaço-temporal de 10 km que resolveu a armadilha zero-inflated, gerando uma variável densa e utilizável para regressão sem distorcer a realidade física da via.


5.3 Economia de Infraestrutura e Diretrizes para o DNIT


A análise de interpretabilidade via SHAP e Gráficos de Dependência Parcial (PDP) permitiu isolar o impacto da engenharia viária sobre o risco de acidentes, contrapondo-se ao senso comum que responsabiliza unicamente o fator humano. O experimento Ceteris Paribus na BR-381 (KM 340-350) demonstrou quantitativamente que a degradação física da via eleva a sinistralidade em 7,2%, mesmo quando variáveis de tráfego, clima e calendário permanecem congeladas. A pista deteriorada reduz a margem de manobra do condutor, transformando pequenas falhas em colisões severas, ou seja, a falta de manutenção no pavimento agrava erros do condutor, levando consideravelmente a sinistralidade.
O achado de maior relevância na pesquisa para o planejamento orçamentário do DNIT reside no formato em cotovelo identificado na curva PDP. A recuperação de um trecho em estado crítico para o nível regular (ICP 0 para 50) ou asfaltar um trecho reduz a sinistralidade em 0,120 acidentes/mês por 10 km, enquanto o salto de regular para excelente (ICP 50 para 100) traz um ganho marginal de apenas 0,080 acidentes.
Esse comportamento dita uma diretriz econômica clara para a gestão de infraestrutura, o Retorno sobre o Investimento em Manutenção (ROMI) para preservação de vidas é exponencialmente mais alto quando o orçamento público é direcionado para eliminar defeitos graves (buracos, trincamentos profundos e drenagens colapsadas) em trechos degradados, em vez de financiar recapeamentos de alto padrão em vias que já operam em condições regulares. O showcase de otimização Argmax na BR-40 (KM 110-120) comprova essa premissa, elevar o pavimento da nota 5 para 70 reduz os sinistros em 17,04%, poupando cerca de 9 acidentes graves por ano em apenas dez quilômetros de rodovia.


5.4 A Hierarquia entre Geometria Física e Dinâmica Temporal


As simulações de cenários operacionais revelam uma hierarquia de causalidade indispensável para o planejamento tático da Polícia Rodoviária Federal. A dominância das pistas retas na importância preditiva (MDI de aproximadamente 35%) decorre de sua predominância na extensão da malha e do excesso de velocidade que essas vias permitem e induzem por monotonia. No entanto, em contraponto ao analisar a densidade de risco por quilômetro, as interseções de vias e curvas acentuadas elevam o risco mediano em +23,1% e +7,3%, respectivamente.
O estudo de caso real do KM 70 da BR-101 sintetiza a relação entre estrutura física e contexto temporal. Em um trecho de geometria complexa (viaduto com rotatória) que acumulou 54 sinistros em um único mês, o Modelo V5 previu 48,7 acidentes. Ao simular variações puramente contextuais (tempo bom vs. chuva, feriado ou dia útil) com a geometria física fixada, a predição oscilou na faixa estreita de 32,0 a 32,3 acidentes/mês.
Essa evidência demonstra que a geometria física e a presença de acessos urbanos determinam o risco base do local, dessa forma esses fatores determinantes prevalecem sobre fatores humanos e temporais, com issoariações de chuva ou turno noturno atuam como moduladores secundários. Para a PRF, a lição prática é simples, blitzes sazonais e reforços temporários de patrulhamento são paliativos necessários no curto prazo, mas a erradicação de nós de acidentes históricos exige intervenções definitivas de engenharia viária para alterar a geometria da via, que devem conduzidas em cooperação com o DNIT.


















