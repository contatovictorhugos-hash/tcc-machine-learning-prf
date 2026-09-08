## 3.2.3 Preparação dos Dados


Concluída a fase de entendimento dos dados, iniciou-se a preparação da base com o objetivo de estruturar os registros para o treinamento do Random Forest. Nessa etapa, as ocorrências deixam o formato transacional de registros pontuais e passam a compor uma base tabular organizada para que o modelo aprenda os padrões preditivos da sinistralidade nas rodovias federais.
### 3.2.3.1 Transformação e Otimização Estrutural


O primeiro desafio enfrentado foi computacional, decorrente do volume enorme dos dados. A série histórica da PRF (2019 a 2025) reúne quase meio milhão de ocorrências que, sem tratamento prévio, sobrecarregariam a memória em ambientes com restrições como o Google Colab. Para atender ao RNF-3, realizou-se a conversão dos tipos de dados: variáveis numéricas em float64 foram convertidas para float32, e variáveis de texto com categorias repetidas foram convertidas para o tipo category do Pandas, reduzindo o consumo de memória.
Em seguida, foram tratados os valores ausentes de localização geográfica. Registros sem informação de rodovia (br) ou quilômetro (km) foram descartados, pois a ausência da coordenada espacial impede vincular o acidente ao trecho físico e inviabiliza a posterior integração com os dados do DNIT. Com essa filtragem, a base consolidou-se em 473.734 registros válidos, mantendo a integridade da série histórica. Variáveis de uso puramente administrativo e sem valor preditivo, a exemplo de delegacia e uop, também foram removidas para simplificar a estrutura.
Por fim, realizou-se a transformação estrutural dos dados: a conversão dos acidentes pontuais em trechos espaciais padronizados de 10 km (calculados como trecho_10km = int(km // 10) * 10). Esse agrupamento consolidou a chave espaço-temporal composta por ano, mes, br e trecho_10km, padronizando a unidade de observação do estudo e permitindo associar diretamente as contagens de sinistros aos segmentos de rodovia avaliados pelo DNIT.


### 3.2.3.2 Engenharia de Atributos


Com a base limpa, iniciou-se a etapa de engenharia de atributos, processo pelo qual variáveis brutas são materializadas em colunas mensuráveis, ampliando a capacidade explicativa do modelo. Essa etapa é diretamente sustentada pelos insights colhidos na análise exploratória.
O primeiro conjunto de atributos criados foi de natureza temporal e sazonal. Conforme evidenciado pelo gráfico da Figura 14, o volume de acidentes apresenta sazonalidade mensal, com dezembro registrando o maior pico (aproximadamente 44.000 ocorrências) e fevereiro o menor vale, padrão explicável pelo maior fluxo de viagens durante o período de festas e férias escolares.


Figura 14 - G2: Sazonalidade - Volume de Acidentes por Mês


Fonte: Elaborado pelo autor (2026).
Complementarmente, a análise da Figura 15 revelou que sexta-feira, sábado e domingo concentram sistematicamente os maiores volumes de sinistros, superando 73.000 ocorrências cada, enquanto a terça-feira registra o menor índice semanal. Como a unidade de observação do modelo foi estruturada em escala mensal, esse efeito foi capturado por meio da variável dias_fim_semana, que quantifica o total exato de sábados e domingos contidos em cada mês e ano avaliados (variando de 8 a 10 dias), além do atributo dias_no_mes (de 28 a 31 dias), que padroniza o tempo de exposição temporal do trecho.
Figura 15 - G5: Frequência de Acidentes por Dia da Semana


Fonte: Elaborado pelo autor (2026).
A influência de períodos atípicos sobre o tráfego é sustentada pela Figura 16, que indica média de feridos superior nos dias de feriado (1,21) em relação aos dias comuns (1,14). Embora a diferença aparente seja de magnitude modesta em valores absolutos, ela é consistente ao longo de toda a série histórica. Para incorporar esse fator à matriz mensal, criou-se o preditor dias_feriado, calculado programaticamente a partir do calendário oficial de feriados nacionais fixos e datas móveis (como Carnaval, Sexta-Feira Santa e Corpus Christi). Adicionalmente, extraiu-se a variável categórica estacao_ano (Verão, Outono, Inverno e Primavera), permitindo ao modelo considerar variações de clima e fluxos de férias ao longo do ano.


Figura 16 - G9: Média de Feridos: Dias Comuns (0) X Feriado (1)

![Figura 16 - G9: Média de Feridos: Dias Comuns (0) vs Feriado (1)](../figuras/figura16_media_feridos_feriados.png)


Fonte: Elaborado pelo autor (2026).
Com isso, essas variáveis exógenas de calendário fornecem ao algoritmo os condicionantes temporais necessários para isolar variações sazonais da malha e viabilizar a posterior simulação de cenários operacionais para a PRF.
O segundo conjunto de atributos, determinante para a capacidade preditiva do modelo, é composto pelos preditores temporais autorregressivos. No contexto da malha rodoviária federal brasileira, nem a PRF nem o DNIT disponibilizam medições contínuas de Volume Diário Médio (VDM) de veículos para a totalidade dos 5.655 trechos monitorados. Para suprir a ausência desse indicador fundamental de exposição ao risco, o histórico recente de sinistralidade de cada segmento foi empregado como um proxy empírico do fluxo de tráfego, da densidade urbana e da geometria fixa da via.
Para cada trecho espacial (br, trecho_10km), a série histórica foi ordenada cronologicamente mês a mês para a computação estrita de quatro variáveis defasadas (lags) sobre o dataset temporal completo detalhado em 3.2.3.4:
* acidentes_lag_1m: quantidade de sinistros ocorridos no mês imediatamente anterior.
* acidentes_lag_2m: quantidade de sinistros registrados há dois meses.
* acidentes_media_movel_3m: média móvel dos três meses anteriores, responsável por suavizar oscilações atípicas de curto prazo.
* acidentes_lag_12m: quantidade de sinistros observados no mesmo mês do ano anterior, atributo concebido para capturar a sazonalidade anual própria daquele trecho específico.
Um ponto importante é que o ano de 2019 foi utilizado para montar as métricas iniciais dessa forma, esse período não possui as variáveis defasadas.
Para assegurar a integridade temporal do pipeline e evitar qualquer forma de vazamento de dados (data leakage), o cálculo dos atributos defasados utilizou exclusivamente observações com defasagem, sem jamais incorporar informações do próprio mês de referência. Essa estrutura espelha a rotina operacional real da PRF, na qual o planejamento e a alocação de efetivo para o mês subsequente utilizam os dados consolidados do mês que acaba de ser encerrado.
Paralelamente, aplicou-se uma regra estrita de eliminação de variáveis pós-evento para garantir o princípio de Zero Target Leakage. Variáveis que descrevem a dinâmica interna ou os desdobramentos do sinistro, tais como `causa_acidente`, `tipo_acidente`, `ilesos`, `total_mortos`, `total_feridos_graves` e `total_veiculos`, foram descartadas da matriz preditiva. Em um cenário real de planejamento operacional, essas informações são posteriores ao evento e não estariam disponíveis antes da sua ocorrência. Permitir que entrassem no conjunto de preditores faria o modelo aprender relações contábeis artificiais, inflando as métricas em laboratório, mas tornando a ferramenta inútil na prática policial preventiva.
Figura 17 - G10: Mapa de Calor (Correlação de Pearson)

![Figura 17 - G10: Mapa de Calor da Correlação de Pearson entre Variáveis Numéricas](../figuras/figura17_matriz_correlacao_pearson.png)


Fonte: Elaborado pelo autor (2026).
O Mapa de Calor de Correlação de Pearson (Figura 17) serviu como ferramenta de diagnóstico para validar essas decisões de seleção e compreender a estrutura de dependência linear da base. A análise confirma fortes correlações esperadas entre grandezas compostas e suas partes, a exemplo de `feridos_leves` e `feridos` (0,86), assim como entre `pessoas` e `feridos` (0,54). Essas redundâncias reiteraram a necessidade de descartar métricas concorrentes de severidade na matriz de entrada, prevenindo ruído estatístico desnecessário.
Mais relevante metodologicamente é o comportamento das variáveis de calendário, cujas correlações lineares bivariadas com o volume de acidentes aproximam-se de zero. Esse resultado não indica ausência de relevância preditiva, mas sim a limitação inerente da correlação linear de Pearson para capturar fenômenos complexos de tráfego. O impacto de feriados ou fins de semana não ocorre de forma linear aditiva, mas de maneira condicional e sinérgica, o acréscimo de risco manifesta-se quando o maior volume de veículos em um feriado interage com curvas perigosas, pista simples ou pavimentos degradados.
Essa característica justifica a opção pelo Random Forest com a técnica de Feature Subspace, uma vez que algoritmos baseados em árvores particionam o espaço amostral para identificar interações não lineares que a matriz de Pearson é estruturalmente incapaz de registrar. Assegurada a ausência de variáveis falsas e com os preditores devidamente estruturados, a base consolidou-se livre de vazamentos e apta para as etapas seguintes de integração e modelagem.


### 3.2.3.3 Tratamento e Integração da Base do DNIT


A integração com os dados de infraestrutura do DNIT adiciona a dimensão da engenharia viária à modelagem, representando um dos principais diferenciais técnicos deste trabalho. A base do DNIT disponibiliza índices avaliativos sobre o estado físico da malha federal: o Índice de Condição do Pavimento (ICP), o Índice de Condição da Capa (ICC) e o Índice de Condição da Manutenção (ICM).
Durante a auditoria da base bruta do DNIT, identificou-se uma inconsistência no arquivo original, 38.526 registros de campo estavam com a coluna de ano em branco, o que causava o descarte dessas linhas em rotinas convencionais de limpeza. A conferência da sequência dos trechos e dos contratos de medição confirmou que o lote correspondia ao levantamento de 2023. O pipeline de extração foi ajustado para rotular e reinserir esses registros no ano correto, restabelecendo as avaliações de pavimento desse período na malha federal.
Como os levantamentos do DNIT são estruturados em segmentos lineares de extensão variável, delimitados por km_inicial e km_final, foi necessário agregá-los espacialmente para compatibilizá-los com o agrupamento de 10 km adotado no projeto. Para cada trecho [t, t+10) de uma determinada rodovia em determinado ano, calculou-se a média aritmética dos índices dos segmentos viários sobrepostos ao intervalo.
Para lidar com a ausência de cobertura em trechos ou períodos sem medição oficial, evitou-se deliberadamente a imputação por média ou mediana, bem como qualquer tipo de retro-preenchimento sintético, conhecido como backfill, práticas que distorceriam a variância real e introduziriam dados fictícios no treinamento. Em vez disso, adotou-se a estratégia do valor sentinela -1.0 para os índices contínuos ausentes (ICP, ICC e ICM), acompanhada da criação da variável binária indicadora has_dnit_data (igual a 1 para trechos com levantamento no ano e 0 para trechos sem cobertura).
Essa abordagem é adequada à arquitetura do Random Forest. Por operar por meio de partições binárias, o algoritmo utiliza o sentinela negativo e a flag booleana para isolar a malha monitorada da não monitorada em ramos específicos das árvores de decisão. Dessa forma, o modelo aprende regras distintas para cada cenário, tratando a falta de monitoramento sistemático como um padrão estrutural próprio da infraestrutura rodoviária, sem comprometer as estimativas dos trechos com avaliação física comprovada.


### 3.2.3.4 Junção Intervalar e Resolução da Armadilha Zero-Inflated


A etapa seguinte consistiu na fusão efetiva entre as informações de ocorrências da PRF e os dados de infraestrutura do DNIT. Como os levantamentos de pavimento do DNIT são delimitados por intervalos lineares contínuos [km_inicial, km_final] de extensões variáveis, a integração não poderia ser realizada por uma junção relacional comum de chave pontual exata, sob risco de inconsistência espacial ou perda substancial de correspondência. A compatibilização foi executada por meio de uma junção por sobreposição intervalar: para cada rodovia e ano, identificaram-se os segmentos do DNIT cujos limites intersectavam o intervalo do trecho padronizado [trecho_10km, trecho_10km + 10), consolidando os índices médios de condição da via para aquele segmento físico.
Com a malha rodoviária federal delimitada em 5.655 trechos físicos de 10 km a partir do universo conjunto de rodovias monitoradas pela PRF e pelo DNIT, emergiu o principal desafio estatístico do projeto: a Armadilha Zero-Inflated decorrente do viés de sobrevivência (survivorship bias). Se o agrupamento fosse realizado exclusivamente sobre os registros em que houve acidentes, o conjunto conteria apenas 139.590 cenários observados com Y >= 1, ignorando os trechos e meses em que nenhum sinistro ocorreu. Um modelo treinado sobre uma base truncada superestimaria sistematicamente a taxa de sinistralidade da malha, sofrendo compressão artificial de variância.
Para solucionar essa armadilha e modelar fidedignamente o cenário inflacionado de zeros (zero-inflated), construiu-se uma grade cartesiana espaço-temporal completa. Todos os 5.655 trechos físicos de 10 km foram cruzados com os 84 meses da série histórica (2019 a 2025), totalizando exatamente 475.020 cenários espaço-temporais únicos. Nos meses e trechos onde ocorreram acidentes, atribuiu-se a contagem observada (139.590 registros, representando 29,39% da base), nos períodos em que nenhum sinistro foi registrado, atribuiu-se formalmente o valor zero (335.430 registros, correspondendo a 70,61% da base).
A inclusão desses zeros reais eliminou o viés de sobrevivência da modelagem e transformou a variável alvo em uma distribuição de contagem realista, permitindo que o Random Forest aprenda a discriminar trechos pacificados de pontos críticos de sinistralidade viária.
