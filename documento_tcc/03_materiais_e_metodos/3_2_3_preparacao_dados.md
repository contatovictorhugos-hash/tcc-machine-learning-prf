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
Complementarmente, a análise da Figura 15 revelou que sexta-feira, sábado e domingo concentram sistematicamente os maiores volumes de sinistros, superando 73.000 ocorrências cada, enquanto a terça-feira registra o menor índice semanal. Como a unidade de observação do modelo foi estruturada em escala mensal, esse efeito foi capturado por meio da variável dias_fim_semana, que quantifica o total exato de sábados e domingos contidos em cada mês e ano avaliados (variando de 8 a 10 dias), além do atributo dias_no_mes (de 28 a 31 dias), que padroniza o tempo de exposição temporal do trecho.

Figura 15 - G5: Frequência de Acidentes por Dia da Semana

Fonte: Elaborado pelo autor (2026).

A influência de períodos atípicos sobre o tráfego é sustentada pela Figura 16, que indica média de feridos superior nos dias de feriado (1,21) em relação aos dias comuns (1,14). Embora a diferença aparente magnitude modesta em valores absolutos, ela é consistente ao longo de toda a série histórica. Para incorporar esse fator à matriz mensal, criou-se o preditor dias_feriado, calculado programaticamente a partir do calendário oficial de feriados nacionais fixos e datas móveis (como Carnaval, Sexta-Feira Santa e Corpus Christi). Adicionalmente, extraiu-se a variável categórica estacao_ano (Verão, Outono, Inverno e Primavera), permitindo ao modelo considerar variações de clima e fluxos de férias ao longo do ano.

Figura 16 - G9: Média de Feridos: Dias Comuns (0) vs Feriado (1)

Fonte: Elaborado pelo autor (2026).

O segundo conjunto de atributos, determinante para a capacidade preditiva do modelo, é composto pelos preditores temporais autorregressivos. No contexto da malha rodoviária federal brasileira, nem a PRF nem o DNIT disponibilizam medições contínuas de Volume Diário Médio (VDM) de veículos para a totalidade dos 5.655 trechos monitorados. Para suprir a ausência desse indicador fundamental de exposição ao risco, o histórico recente de sinistralidade de cada segmento foi empregado como um proxy empírico do fluxo de tráfego, da densidade urbana e da geometria fixa da via.

Para cada trecho espacial (br, trecho_10km), a série histórica foi ordenada cronologicamente mês a mês para a computação estrita de quatro variáveis defasadas (lags):
* acidentes_lag_1m: quantidade de sinistros ocorridos no mês imediatamente anterior (t-1).
* acidentes_lag_2m: quantidade de sinistros registrados há dois meses (t-2).
* acidentes_media_movel_3m: média móvel dos três meses anteriores (t-1, t-2 e t-3), responsável por amortecer oscilações atípicas de curto prazo.
* acidentes_lag_12m: quantidade de sinistros observados no mesmo mês do ano anterior (t-12), atributo concebido para capturar a sazonalidade anual própria daquele trecho específico.

Para assegurar a integridade temporal do pipeline e evitar qualquer forma de vazamento de dados (data leakage), o cálculo dos atributos defasados utilizou exclusivamente observações com defasagem t-k (k >= 1), sem jamais incorporar informações do próprio mês de referência t. Essa estrutura espelha a rotina operacional real da PRF, na qual o planejamento e a alocação de efetivo para o mês subsequente utilizam os dados consolidados do mês que acaba de ser encerrado.

Paralelamente, aplicou-se uma regra estrita de eliminação de variáveis pós-evento para garantir o princípio de Zero Target Leakage. Variáveis que descrevem a dinâmica interna ou os desdobramentos do sinistro — tais como causa_acidente, tipo_acidente, ilesos, total_mortos, total_feridos_graves e total_veiculos — foram descartadas da matriz preditiva. Em um cenário real de planejamento operacional, essas informações são posteriores ao evento e não estariam disponíveis antes da sua ocorrência. Permitir que entrassem no conjunto de preditores faria o modelo aprender relações contábeis artificiais, inflando as métricas em laboratório, mas tornando a ferramenta inútil na prática policial preventiva.

Figura 17 - G10: Mapa de Calor da Correlação de Pearson entre Variáveis Numéricas

Fonte: Elaborado pelo autor (2026).

O Mapa de Calor de Correlação de Pearson (Figura 17) serviu como ferramenta de diagnóstico para validar essas decisões de seleção e compreender a estrutura de dependência linear da base. A análise confirma fortes correlações esperadas entre grandezas compostas e suas partes, a exemplo de feridos_leves e feridos (0,86), assim como entre pessoas e feridos (0,54). Essas redundâncias reiteraram a necessidade de descartar métricas concorrentes de severidade na matriz de entrada, prevenindo ruído estatístico desnecessário.

Mais relevante metodologicamente é o comportamento das variáveis de calendário, cujas correlações lineares bivariadas com o volume de acidentes aproximam-se de zero. Esse resultado não indica ausência de relevância preditiva, mas sim a limitação inerente da correlação linear de Pearson para capturar fenômenos complexos de tráfego. O impacto de feriados ou fins de semana não ocorre de forma linear aditiva, mas de maneira condicional e sinérgica: o acréscimo de risco manifesta-se quando o maior volume de veículos em um feriado interage com curvas perigosas, pista simples ou pavimentos degradados. 

Essa característica justifica a opção pelo Random Forest com a técnica de Feature Subspace, uma vez que algoritmos baseados em árvores particionam o espaço amostral para identificar interações não lineares que a matriz de Pearson é estruturalmente incapaz de registrar. Assegurada a ausência de variáveis espúrias e com os preditores devidamente estruturados, a base consolidou-se livre de vazamentos e apta para as etapas seguintes de integração e modelagem.


3.2.3.3 Tratamento e Integração da Base do DNIT


Primeiramente, a integração com os dados de infraestrutura do DNIT representa um grande diferencial deste projeto, além de ser um grande desafio técnico. A base do DNIT, que contém índices críticos como o ICC (Índice de Condição de Capa) e o ICP (Índice de Condição do Pavimento), possui cobertura temporal iniciada em 2021, gerando uma lacuna de dois anos em relação ao início da série histórica da PRF (2019). Para contornar esse problema sem introduzir imputação pela média, que por sua vez iria contaminar a variância e distorceria os dados, optou-se por aplicar a técnica de um Backfill Estratégico, ou retro-preenchimento temporal, que copia as medições de 2021 para os anos anteriores de 2019 e 2020. A justificativa é baseada na premissa estabelecida que as condições estruturais de um pavimento variam de forma lenta e gradual, sendo normal e razoável assumir que o estado de uma via em 2021 representa uma aproximação válida para a sua condição nos anos imediatamente anteriores.
Para os registros em que a integração do DNIT com a PRF não resultou em correspondência, optou-se pela imputação com a constante sentinela -1.0 nos índices contínuos ausentes, em vez de recorrer à média ou mediana. Essa decisão é a melhor metodologicamente para o Random Forest, por operar através de partições binárias, o modelo processa o valor -1.0 como uma categoria diferente, aprendendo que a ausência de monitoramento sistemático do DNIT é, por si só, um indicador de risco.


3.2.3.4 Junção Intervalar e Resolução da Armadilha Zero-Inflated


Passando agora para efetivamente a fusão entre as duas bases, foi realizada através de um Interval Join do tipo ASOF, utilizando como chaves de junção o ano e o código da rodovia, a respectiva BR, alocando o quilômetro exato do sinistro dentro do intervalo contínuo [km_inicial, km_final] correspondente no cadastro do DNIT. Essa estratégia, executada pelo Pandas, resultou em correspondência geográfica para 63,95% da base histórica, enriquecendo os registros com os índices de condição do pavimento sem recorrer à explosão combinatória que um merge convencional por quilômetro decimal exato produziria, o que com um grande nível de certeza esgotaria a memória disponível para continuidade do projeto.
Com isso, foi resolvida a integração, e ao mesmo tempo emergiu o principal risco estatístico encontrado no projeto, a Armadilha Zero-Inflated. A base de 473.734 eventos individuais, por sua natureza granular, concentrava uma proporção enorme de registros com zero ou uma ocorrência por trecho em determinado dia, um cenário de inflação de zeros que comprometeria qualquer modelo de regressão treinado diretamente sobre os eventos brutos. A solução adotada foi o Bucketing de espaço temporal, a base foi agregada em trechos de 10 km mensais, criando 139.590 cenários preditivos únicos com uma variável alvo contínua e densa, quantidade_acidentes. Essa transformação é o que permite ao modelo estimar o volume de acidentes em um trecho específico para um determinado mês, convertendo eventos discretos e esparsos em inteligência estratégica, esses passos realizados representam a materialização do objetivo central do projeto, com a conclusão da base para treinamento.
