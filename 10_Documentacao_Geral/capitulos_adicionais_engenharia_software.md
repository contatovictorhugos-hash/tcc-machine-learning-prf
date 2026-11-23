# Capítulos complementares de engenharia de software

## Capítulo 12. Engenharia de integração espaço-temporal e modelagem de dados

### 12.1. Referenciamento linear e cruzamento de bases heterogêneas

O cruzamento entre os boletins de acidentes da Polícia Rodoviária Federal e as medições de infraestrutura viária do Departamento Nacional de Infraestrutura de Transportes exigiu a aplicação do conceito de referenciamento linear (*Linear Referencing System*). Na base da PRF, cada ocorrência possui uma localização espacial pontual expressa por três coordenadas principais: a identificação da rodovia federal (código BR), a coordenada quilométrica contínua (quilômetro e fração decimal) e o par de latitude e longitude obtido pelos aparelhos de GPS das equipes policiais.

Por outro lado, o cadastro do Sistema Nacional de Viação mantido pelo DNIT organiza as informações da malha rodoviária federal em segmentos homogêneos de inspeção. Cada trecho cadastrado no DNIT possui uma rodovia de referência, um quilômetro inicial e um quilômetro final, aos quais estão associados índices técnicos de engenharia civil, como o Índice de Condição do Pavimento (ICP), o Índice de Gravidade Global (IGG) e o tipo de superfície do rolamento.

A integração dos dados foi construída por meio de uma operação de junção por intervalo (*interval join*) em Python. A rotina processa as bases aplicando uma chave composta formada pelo código numérico da rodovia federal e pelo enquadramento da quilometragem do sinistro dentro do intervalo fechado entre o quilômetro inicial e o quilômetro final do segmento de inspeção correspondente. Nos casos de divergência geométrica decorrentes de desvios locais ou contornos urbanos sem correspondência cadastral direta, o algoritmo calcula a distância euclidiana entre as coordenadas geográficas da ocorrência e o centroide do trecho viário mais próximo, fixando uma tolerância máxima de quinhentos metros para validar o vínculo espacial.

### 12.2. Discretização e agregação em blocos espaço-temporais

A análise espacial direta dos sinistros revelou uma taxa de esparsidade superior a 99% quando os eventos são observados no nível do quilômetro individual e da data exata. Em um dia típico de operação, a grande maioria dos pontos quilométricos de uma rodovia federal registra zero acidentes, o que gera uma distribuição estatística dominada por excesso de zeros (*zero-inflation*). Essa concentração impede o treinamento estável de estimadores de regressão supervisionada, que tendem a convergir para estimativas nulas ou a colapsar a variância prevista.

Para contornar esse obstáculo matemático, o pipeline estruturou a malha rodoviária em blocos espaço-temporais regulares. A malha federal pavimentada foi fatiada em segmentos contíguos de dez quilômetros de extensão, agregados na granularidade de mês civil. O período histórico de janeiro de 2019 a dezembro de 2025 foi distribuído ao longo de 84 meses consecutivos. A discretização resultou em 5.655 segmentos rodoviários monitorados, totalizando uma matriz analítica de 475.020 observações mensais.

A agregação dos dados de infraestrutura dentro de cada bloco de dez quilômetros empregou a média ponderada das extensões inspecionadas pelo DNIT. As características topográficas da via, tais como a presença de traçado em curva ou perfil em rampa, foram mapeadas como proporções físicas do trecho. A quantidade total de sinistros verificados em cada combinação de trecho e mês assumiu o papel de variável dependente contínua (Y) para o treinamento algorítmico.

### 12.3. Engenharia de variáveis temporais defasadas

Rodovias federais não possuem sensores contínuos de contagem veicular instalados de forma homogênea em todo o território nacional. Diante da ausência de dados sistemáticos de volume diário médio de tráfego para a totalidade dos trechos, o pipeline adotou a criação de variáveis temporais defasadas (*lags*) como estimadores indiretos da inércia de circulação.

A rotina de preparação calculou quatro variáveis defasadas para cada trecho rodoviário:

1. Defasagem de um mês (*lag 1m*): contagem de ocorrências observada no mês imediatamente anterior.
2. Defasagem de dois meses (*lag 2m*): contagem observada com dois meses de antecedência.
3. Defasagem de três meses (*lag 3m*): contagem observada com três meses de antecedência.
4. Defasagem de doze meses (*lag 12m*): contagem registrada no mesmo mês do ano civil anterior, projetada para absorver efeitos sazonais recorrentes, como períodos de férias escolares e feriados prolongados.

Essas variáveis funcionam como filtros de inércia temporal. Trechos densamente povoados ou localizados em regiões metropolitanas sustentam médias históricas estáveis de ocorrências devido ao alto volume diário de veículos. Já rodovias do interior com tráfego reduzido mantêm contagens residuais próximas a zero ao longo de vários ciclos mensais. A introdução dos *lags* permitiu que o modelo separasse o efeito do volume habitual de circulação das variações induzidas por deterioração física do pavimento ou por oscilações climáticas sazonais. O ano de 2019 foi reservado como janela de aquecimento (*warm-up*), servindo exclusivamente para preencher os valores das defasagens temporais sem integrar o cálculo de métricas de avaliação do modelo preditivo.

### 12.4. Persistência intermediária e modelagem dimensional

O processamento conjunto das tabelas de sinistros e de infraestrutura produz o conjunto consolidado `df_model_ready_v6.csv`, que armazena as quarenta e duas variáveis preditoras selecionadas para o algoritmo Random Forest. Para atender aos requisitos de apresentação analítica no Power BI, a camada de persistência foi estruturada sob os princípios de modelagem dimensional em esquema estrela (*star schema*).

A arquitetura organiza os dados em torno de uma tabela fato central, denominada `Fato_Sinistros_Predicoes`, cujas linhas representam a observação de um segmento rodoviário de dez quilômetros em um mês específico. Essa tabela fato armazena a contagem real de sinistros observados, a quantidade predita pelo modelo serializado, os resíduos absolutos do algoritmo e o somatório de vítimas feridas e fatais.

Conectadas à tabela fato, quatro tabelas dimensionais fornecem o contexto analítico:

1. `Dim_Tempo`: detalha o ano, mês, trimestre, semestre e indicadores booleanos de feriados nacionais e períodos de férias.
2. `Dim_Geografia`: descreve a Unidade da Federação, o código numérico da BR, o intervalo quilométrico inicial e final do bloco, a superintendência regional e a delegacia responsável da PRF.
3. `Dim_Infraestrutura`: categoriza os trechos de acordo com as faixas de qualidade do pavimento estabelecidas pelo DNIT (Ótimo, Bom, Regular, Ruim e Péssimo), além de registrar o tipo de pista (simples, dupla ou múltipla).
4. `Dim_Gravidade`: segmenta as classificações operacionais de severidade das ocorrências (sem vítimas, com feridos leves, com feridos graves e com óbitos confirmados).

Essa separação dimensional reduz a redundância de dados textuais e acelera as consultas relacionais no mecanismo de análise em memória do Power BI, permitindo filtragens dinâmicas por região e por faixa de pavimento sem perdas de desempenho.

## Capítulo 13. Governança de dados, conformidade e qualidade cadastral

### 13.1. Ingestão de dados abertos e conformidade com a LGPD

O pipeline consome dados disponibilizados publicamente por dois órgãos federais: os microdados de acidentes de trânsito da Polícia Rodoviária Federal, publicados no portal de dados abertos do governo federal, e as planilhas de condição da malha viária mantidas pelo DNIT.

Embora se trate de bases de acesso público regidas pela Lei de Acesso à Informação (Lei nº 12.527/2011), o tratamento desses registros deve observar as exigências da Lei Geral de Proteção de Dados Pessoais (Lei nº 13.709/2018). Os boletins de acidentes da PRF podem conter campos que identificam condutores e passageiros envolvidos em sinistros, tais como números de documentos de habilitação, placas de veículos e descrições detalhadas do histórico médico dos feridos.

A etapa de ingestão do software aplica uma rotina de descarte preventivo que elimina esses campos antes de qualquer operação de gravação ou cruzamento. Apenas variáveis de natureza estritamente estatística, física e espacial são preservadas pelo processo de ETL. Identificadores cadastrais de pessoas físicas e placas de veículos são removidos no momento da leitura do arquivo bruto. O sistema mantém apenas o código sequencial anônimo da ocorrência para fins de integridade referencial interna.

### 13.2. Padronização de esquemas e tratamento de divergências históricas

A extração de séries temporais estendidas entre 2019 e 2025 encontrou variações frequentes na formatação dos arquivos publicados pelos órgãos governamentais. Ao longo desse período de sete anos, a PRF alterou por duas vezes a nomenclatura padronizada de colunas em seus repositórios anuais. Por exemplo, a variável que identifica a data do acidente transitou de `data_inversa` para `data_ocorrencia` em determinadas publicações, enquanto os identificadores de delegacias regionais sofreram reestruturações em suas siglas institucionais.

Além das divergências de nomes, os arquivos brutos apresentavam inconsistências de codificação de caracteres (*encodings*) e de delimitadores textuais. Documentos dos anos de 2019 e 2020 utilizavam a codificação ISO-8859-1 (Latin-1) e ponto e vírgula como separador de campos, enquanto arquivos a partir de 2023 adotaram a codificação UTF-8 e vírgula como delimitador padrão.

Para assegurar a repetibilidade do pipeline, o módulo de extração implementa um dicionário centralizado de mapeamento de esquemas. Antes de integrar qualquer novo arquivo anual à base consolidada, o script realiza as seguintes operações:

1. Verificação automática da codificação do arquivo por meio da análise dos primeiros bytes do documento.
2. Leitura com o delimitador correto detectado por amostragem das linhas iniciais.
3. Renomeação padronizada das colunas de acordo com o esquema mestre da arquitetura do projeto.
4. Coerção estrita dos tipos de dados para ponto flutuante, inteiro ou texto padronizado em letras minúsculas sem acentuação.
5. Validação de formato para as datas e horas de registro das ocorrências.

### 13.3. Qualidade cadastral e valores sentinela nas medições do DNIT

A auditoria sobre as bases do DNIT revelou uma limitação estrutural na cobertura das avaliações de pavimento. No período analisado, apenas 17,7% dos blocos espaço-temporais monitorados continham relatórios de inspeção contemporâneos de Índice de Condição do Pavimento (ICP) e Índice de Gravidade Global (IGG). Os restantes 82,3% dos trechos da malha federal não dispunham de vistorias registradas no período ou não foram contemplados nos contratos vigentes de monitoramento do órgão.

Em projetos convencionais de engenharia de dados, valores ausentes costumam ser tratados por técnicas de imputação estatística, tais como o preenchimento pela média, pela mediana ou pelo valor mais recente observado. No caso da infraestrutura viária federal, essa prática introduziria distorções graves. Atribuir a condição média do asfalto de uma rodovia duplicada a uma rodovia vicinal não monitorada criaria uma falsa sensação de regularidade e alteraria a variância natural da distribuição física dos pavimentos.

A decisão de engenharia adotada consistiu no emprego do valor sentinela `-1.0` para todas as medições de ICP e IGG em trechos sem cobertura cadastral. Como o algoritmo de modelagem selecionado é o Random Forest, baseado em particionamento recursivo por árvores de decisão, a escolha do valor sentinela oferece vantagens técnicas diretas:

1. O valor `-1.0` situa-se fora da escala teórica natural do índice de pavimento do DNIT, que varia exclusivamente entre 0.0 (pavimento totalmente destruído) e 100.0 (pavimento excelente).
2. O algoritmo consegue criar pontos de corte numéricos nos nós das árvores que isolam perfeitamente os segmentos não monitorados em ramos específicos da estrutura de decisão.
3. A ausência de inspeção cadastral passa a ser tratada como um estado físico explícito de governança, sem inventar valores artificiais e sem reduzir o tamanho da amostra histórica disponível para treinamento.

## Capítulo 14. Dívida técnica e auditoria de vazamento de dados (*target leakage*)

### 14.1. Diagnóstico de dívida técnica nas versões preliminares

O desenvolvimento do pipeline de modelagem preditiva passou por seis versões iterativas de arquitetura, identificadas de V1 a V6. As versões iniciais buscaram estimar a frequência de acidentes com base em dados desagregados de boletins de ocorrência. Conforme documentado no relatório técnico de auditoria do projeto, os modelos V4 e V5 alcançaram métricas de desempenho aparentemente excepcionais, com coeficiente de determinação $R^2$ de 0,9726 e Erro Médio Absoluto (MAE) de 0,4181 em testes temporais.

Na literatura de engenharia de software para aprendizado de máquina, resultados excessivamente altos em problemas com forte componente aleatório constituem um sinal de alerta de falha metodológica. Acidentes rodoviários possuem variabilidade estocástica decorrente de fatores comportamentais dos condutores, falhas mecânicas repentinas e eventos climáticos imprevistos. Modelos legítimos reportados na literatura internacional de engenharia de tráfego raramente superam valores de $R^2$ entre 0,30 e 0,55 quando projetados para períodos futuros.

A análise aprofundada do código-fonte confirmou a presença de dívida técnica oculta, conforme classificada por Sculley et al. (2015). A pressa em alcançar índices elevados de ajuste ocultou dois vícios matemáticos de construção: o vazamento de variáveis dependentes (*target leakage*) e o viés de sobrevivência na composição da base amostral.

### 14.2. O mecanismo contábil de vazamento de dados no Modelo V5

A investigação forense do script de pré-modelagem (`pre_modeling.py`) identificou a origem exata do vazamento de dados presente na versão V5. Na etapa de engenharia de atributos, as características de traçado geométrico da via registradas pela PRF em cada boletim de ocorrência foram convertidas em variáveis binárias (por exemplo: `is_tracado_via_Reta`, `is_tracado_via_Curva` e `is_tracado_via_Intersecao`).

Ao agrupar os acidentes nos blocos espaço-temporais de dez quilômetros por mês, a rotina de agregação aplicou a função de soma para todas as colunas que iniciavam com o prefixo `is_`:

```python
elif col.startswith('is_') or col.startswith('is'):
    agg_dict[col] = 'sum'
```

Como cada linha da base bruta da PRF representa um sinistro ocorrido, a operação gerou a seguinte relação matemática:

`sum(is_tracado_via_Reta)` = contagem de acidentes em linha reta naquele mês.

`sum(is_tracado_via_Curva)` = contagem de acidentes em curva naquele mês.

`sum(is_tracado_via_Intersecao)` = contagem de acidentes em interseção naquele mês.

Somando essas categorias de traçado contidas na própria matriz de preditores, obtém-se:

$$\text{acidentes em reta} + \text{acidentes em curva} + \text{acidentes em interseção} \approx \text{total de acidentes (Variável Alvo Y)}$$

O algoritmo Random Forest identificou essa decomposição contábil direta. Ele não aprendeu padrões físicos que antecedem a ocorrência de acidentes; apenas executou a soma aritmética de colunas que continham fatias da própria variável resposta coletadas no mesmo mês. Quando o modelo foi avaliado no conjunto de teste de 2024 a 2025, os preditores de entrada já continham a contagem somada dos sinistros ocorridos no período de teste, viabilizando o ajuste aparente de 97% por reprodução contábil do alvo.

### 14.3. O viés de sobrevivência na base de treinamento

O segundo vício identificado nas versões V1 a V5 decorreu do viés de sobrevivência da amostra. A base de dados utilizada na modelagem foi construída aplicando a cláusula de agrupamento exclusivamente sobre a tabela histórica de acidentes fornecida pela PRF.

Sob essa lógica, apenas trechos e meses que registraram ao menos um acidente foram incluídos na matriz consolidada. Trechos rodoviários que passaram meses ou anos sem nenhum registro de sinistro não constavam nos arquivos da PRF e, portanto, foram completamente omitidos da modelagem. A variável alvo assumiu o valor mínimo obrigatório de um acidente ($Y \ge 1$).

Essa restrição impediu o modelo de aprender os fatores determinantes para a segurança viária. Ele operava estimando o volume de ocorrências condicionado à certeza prévia de que um acidente aconteceria ($E[Y \mid Y \ge 1, X]$). Para a operação preventiva das forças policiais, essa formulação era inadequada, pois o sistema era incapaz de discriminar trechos tranquilos de segmentos em situação de perigo iminente.

### 14.4. A Arquitetura V6 auditada e o estudo de ablação

A versão V6 eliminou integralmente o vazamento contábil e corrigiu o viés amostral. A nova arquitetura adotou três intervenções estruturais:

1. Construção da matriz cartesiana completa: a totalidade dos 5.655 trechos de dez quilômetros foi multiplicada pelos 84 meses do período, atribuindo o valor $Y = 0$ a todos os segmentos que não registraram ocorrências no mês correspondente.
2. Remoção das contagens agregadas contemporâneas: todas as variáveis de soma de traçado e de categorias de colisão foram excluídas do conjunto de preditores. As características físicas da via passaram a ser descritas exclusivamente por dados cadastrais estáticos e pelas medições do DNIT.
3. Partição temporal estrita (*out-of-time split*): os anos de 2020 a 2023 (271.440 observações) foram destinados ao treinamento, enquanto os anos de 2024 e 2025 (135.720 observações) foram mantidos como conjunto de teste cego, simulando a aplicação do modelo em produção.

A tabela a seguir apresenta os resultados do modelo oficial V6 em comparação com os modelos de controle no conjunto de teste cego de 2024 a 2025:

| Métrica de avaliação | V6-Full (Poisson Oficial) | V6-MSE (Benchmark Quadrático) | V6-Ablation (Infraestrutura Pura) |
| :--- | :--- | :--- | :--- |
| Critério de perda das árvores | Poisson Deviance | Squared Error (MSE) | Poisson Deviance |
| Coeficiente de determinação ($R^2$) | 0,8678 | 0,8736 | 0,0381 |
| Escores de Deviance Poisson ($D^2$) | 0,7964 | 0,7924 | 0,1024 |
| Erro Médio Absoluto Global (MAE) | 0,6137 | 0,6219 | 1,4550 |
| Raiz do Erro Quadrático Médio (RMSE) | 1,2786 | 1,2502 | 3,4493 |
| MAE em trechos seguros ($Y = 0$) | 0,2889 | 0,3075 | 0,9554 |
| Quantidade de variáveis de entrada | 42 | 42 | 38 |

O modelo V6-Full com critério de perda de Poisson alcançou $R^2 = 0,8678$ sem recorrer a vazamento de dados, apresentando um erro médio absoluto de apenas 0,2889 acidentes por mês para trechos que não registraram sinistros reais ($Y = 0$). Isso confirma a capacidade do modelo de identificar trechos rodoviários seguros.

O estudo de ablação (modelo V6-Ablation), executado sem as variáveis temporais defasadas e alimentado apenas por atributos do DNIT e fatores de calendário, obteve $R^2 = 0,0381$. Essa queda expressiva comprova que o estado físico do pavimento, isoladamente, possui baixo poder para determinar o volume absoluto de colisões. O número de acidentes depende da densidade de veículos em circulação, que é capturada pelos *lags*, enquanto as medições do DNIT atuam ajustando as estimativas locais de severidade e risco residual.

## Capítulo 15. Operação, implantação em lote e monitoramento contínuo

### 15.1. Padrão arquitetural de inferência em lote (*batch servicing*)

A escolha da estratégia de implantação de um sistema de aprendizado de máquina deve refletir os processos de negócio dos usuários atendidos. No contexto de segurança viária e conservação de rodovias federais, as decisões de alocação de viaturas da PRF e as programações de recapeamento do DNIT são planejadas em ciclos mensais, trimestrais e anuais. Não há necessidade operacional de previsões em milissegundos para essas finalidades.

Por essa razão, o software adotou o padrão de inferência em lote (*batch servicing*), evitando custos desnecessários de infraestrutura de computação em nuvem que seriam exigidos por microsserviços de API síncrona.

O fluxo de implantação em lote opera através das seguintes etapas sequenciais:

1. Extração agendada: rotinas automáticas realizam o download mensal das atualizações de sinistros publicadas pela PRF.
2. Execução do pipeline de transformação: os novos dados passam pela limpeza de nulos, junção espacial linear e computação dos atributos defasados em relação aos meses precedentes.
3. Carga do artefato preditivo: o script carrega o arquivo binário serializado com os pesos treinados do modelo (`random_forest_poisson_v6.pkl`).
4. Geração de escores preditivos: o modelo estima a quantidade esperada de acidentes para os trechos de dez quilômetros do ciclo subsequente.
5. Persistência de resultados: os valores previstos são gravados em tabelas de saída no formato Parquet e carregados no banco de dados analítico que alimenta os painéis gerenciais do Power BI.

### 15.2. Monitoramento de deriva de dados (*data drift*) e de conceito (*concept drift*)

Sistemas analíticos em operação contínua sofrem degradação de desempenho provocada por mudanças no ambiente de negócio e na coleta dos dados. O pipeline estabelece procedimentos formais de monitoramento focados em duas frentes:

A deriva de dados (*data drift*) ocorre quando a distribuição estatística das variáveis independentes de entrada sofre alterações em relação aos dados observados na fase de treinamento. O sistema aplica o teste bicaudal de Kolmogorov-Smirnov sobre atributos contínuos (como o ICP do DNIT) e a métrica de divergência de Jensen-Shannon sobre variáveis categóricas (como o tipo de pista). Quando o valor p do teste de Kolmogorov-Smirnov cai abaixo de 0,05 para uma variável com peso relevante no modelo, o sistema dispara um alerta de inconsistência cadastral.

A deriva conceitual (*concept drift*) ocorre quando a relação causal entre os preditores de entrada e a variável dependente de saída se transforma com o tempo. Um exemplo real registrado na base analisada ocorreu entre os anos de 2020 e 2021. As restrições de mobilidade urbana decorrentes da pandemia alteraram temporariamente a relação histórica entre densidade veicular e sinistralidade. A detecção dessa alteração no pipeline é realizada monitorando a curva do Erro Médio Absoluto (MAE) a cada fechamento de trimestre civil. Quando o MAE das previsões em lote se desvia mais de 25% da linha de base de validação, uma notificação técnica indica a necessidade de intervenção.

### 15.3. Política de retreinamento e ciclo de vida de artefatos

O ciclo de vida do modelo preditivo segue diretrizes formais para reexecução de treinamento e aposentadoria de versões:

1. Retreinamento programado: o modelo passa por recalibração completa uma vez ao ano, imediatamente após a consolidação oficial dos relatórios anuais da PRF e do DNIT, incorporando o novo ano civil ao conjunto histórico de treino.
2. Retreinamento emergencial: ocorre caso os alertas de deriva conceitual permaneçam ativos por dois trimestres consecutivos, indicando alterações estruturais permanentes no tráfego das rodovias federais.
3. Critérios de substituição em produção: uma nova versão candidata do modelo só substitui o artefato ativo em produção se apresentar melhora estatisticamente comprovada nas métricas de resíduos absolutos no conjunto de teste cego mais recente, sem apresentar degradação no subconjunto de trechos seguros ($Y = 0$).
4. Aposentadoria de artefatos: arquivos `.pkl` substituídos são transferidos para um repositório de arquivamento com marcas temporais e metadados de execução, garantindo a rastreabilidade histórica e a capacidade de auditoria retroativa.

### 15.4. Rastreamento de linhagem de dados (*data lineage*)

Para atender aos padrões de conformidade da engenharia de software analítica, cada execução do pipeline de treinamento gera um registro de auditoria contendo metadados completos de execução:

1. Código de identificação do *commit* no repositório Git correspondente à versão exata dos scripts executados.
2. Assinatura criptográfica (hash SHA-256) dos arquivos brutos de dados da PRF e do DNIT utilizados como entrada.
3. Dicionário de hiperparâmetros configurados para o Random Forest (número de estimadores, profundidade máxima, semente aleatória e critério de perda).
4. Relatório tabular de métricas de desempenho obtidas nas partições de treino e teste.
5. Localização do arquivo serializado gerado (`.pkl`) e data e hora exatas da conclusão do processamento.

Esse encadeamento impede discrepâncias entre o modelo que opera nos relatórios do Power BI e o código-fonte mantido pela equipe de desenvolvimento.

## Capítulo 16. Gerenciamento de riscos de software e restrições operacionais

### 16.1. Riscos de dependência de fontes externas de dados

O pipeline apoia-se inteiramente no consumo de dados abertos governamentais, o que cria dependências técnicas externas fora do controle da equipe de desenvolvimento:

1. Indisponibilidade de infraestrutura pública: os portais da PRF e do DNIT passam por manutenções não programadas e eventuais períodos de instabilidade que podem interromper o download automatizado de novas bases. O software mitiga esse risco mantendo réplicas locais imutáveis em armazenamento secundário de todos os arquivos brutos já processados.
2. Alterações silenciosas de estrutura: publicações governamentais podem sofrer mudanças no padrão de preenchimento ou supressão de colunas sem aviso prévio. O módulo de ingestão possui asserções formais de validação de esquema (*schema assertions*). Se um arquivo novo não contiver as colunas obrigatórias ou apresentar formatos inconsistentes, o processamento é interrompido de forma controlada antes de afetar os dados analíticos consolidados.
3. Atraso na divulgação de dados consolidados: enquanto a PRF costuma atualizar os boletins com frequência mensal, os dados de condição de pavimento do DNIT possuem defasagens temporais que podem superar doze meses em algumas unidades da federação. O pipeline trata essa assimetria permitindo que a camada preditiva opere com a medição viária mais recente disponível ou utilize o sentinela `-1.0` quando a informação expirar.

### 16.2. Limitações de granularidade e subjetividade dos registros

As bases de dados utilizadas apresentam restrições de granularidade inerentes aos métodos de coleta dos órgãos públicos:

1. Subjetividade nas variáveis climáticas: nos boletins da PRF, as condições meteorológicas no momento da ocorrência são registradas com base na percepção visual do policial que atendeu o evento (como "chuva", "nublado" ou "céu claro"). Não há medição por estações meteorológicas de parâmetros físicos objetivos, tais como precipitação pluviométrica em milímetros por hora, velocidade do vento ou presença de lâmina de água sobre o asfalto.
2. Ausência de medidas de curvatura física contínua: a geometria do traçado da via nos dados da PRF baseia-se em classificações qualitativas ("reta", "curva" ou "interseção"). O modelo não dispõe de grandezas topográficas contínuas, tais como o raio médio de curvatura expresso em metros, o ângulo de deflexão e a taxa de superelevação da pista.
3. Falta de medições contínuas de velocidade de operação: a base não registra a velocidade real dos veículos minutos antes da colisão, dependendo de aproximações indiretas baseadas nos limites regulamentados da via.

### 16.3. Riscos de interpretação gerencial e o paradoxo de compensação de risco

Um dos maiores riscos na implantação de ferramentas preditivas no setor público é a interpretação errônea das saídas estatísticas por gestores operacionais. A análise dos dados de acidentes da PRF cruzados com as avaliações de pavimento do DNIT revelou o Efeito Peltzman, também conhecido como teoria da compensação de risco na engenharia de tráfego.

Os dados comprovaram que rodovias com pavimento em estado de conservação ótimo ($ICP \ge 80$) registraram taxas de ocorrências de acidentes similares ou até ligeiramente superiores às de rodovias em estado regular. Esse comportamento decorre da conduta dos motoristas: quando o asfalto é perfeito, a velocidade média de circulação aumenta expressivamente, elevando a frequência de ultrapassagens e colisões traseiras leves em trechos com grande fluxo de veículos. Em contraste, em vias extremamente degradadas ($ICP \le 20$), os condutores reduzem voluntariamente a velocidade para desviar de buracos, diminuindo colisões rotineiras.

Por outro lado, a análise da severidade dos sinistros revelou a realidade da conservação viária:

| Faixa de Condição do Pavimento (DNIT) | Taxa de mortalidade (% de acidentes com óbito) | Índice de vítimas graves (mortos e feridos graves) |
| :--- | :--- | :--- |
| Péssimo ($ICP \le 20$) | 10,50% | 41,1% |
| Ruim ($ICP \in [21, 40]$) | 8,83% | 37,7% |
| Regular ($ICP \in [41, 60]$) | 7,99% | 36,3% |
| Bom ($ICP \in [61, 80]$) | 7,32% | 31,8% |
| Ótimo ($ICP \in [81, 100]$) | 8,62% | 35,8% |

Em trechos com asfalto péssimo, a taxa de letalidade atinge 10,50%, contra 7,32% em trechos em boa condição. Caso um tomador de decisão analise apenas a contagem total de sinistros prevista pelo modelo, ele poderia chegar à conclusão falsa de que reparar o asfalto aumenta o perigo da via. Os relatórios analíticos integrados ao Power BI foram desenhados para exibir simultaneamente o volume previsto e a gravidade histórica, impedindo que o modelo seja utilizado para justificar a redução de investimentos em conservação.

### 16.4. Salvaguardas operacionais no apoio à decisão

Para prevenir o uso inadequado das saídas geradas pelo modelo, o sistema estabelece três salvaguardas operacionais:

1. Princípio do apoio à decisão: as projeções do software funcionam como subsídio quantitativo para otimizar rotas de patrulhamento da PRF e priorizar trechos para sinalização pelo DNIT, sem substituir as vistorias técnicas de campo realizadas por engenheiros de tráfego.
2. Apresentação explícita de intervalos de resíduo: nos painéis executivos, os trechos ranqueados por risco acompanham o erro médio histórico de previsão daquele segmento, permitindo que as equipes avaliem a incerteza estatística da estimativa.
3. Exclusão de julgamento de causalidade individual: o sistema não emite previsões sobre acidentes isolados nem atribui culpa jurídica a condutores ou concessionárias, limitando sua atuação à estimativa de densidade agregada de eventos por bloco de dez quilômetros.

## Capítulo 17. Manutenção de software e evolução arquitetural

### 17.1. Categorização das atividades de manutenção do sistema

A sustentação do pipeline de dados e predição enquadra-se nas categorias formais de manutenção de software previstas na norma ISO/IEC 14764:

1. Manutenção corretiva: abrange a resolução de falhas imprevistas de processamento, tais como o reparo em rotinas de interpretação de formatos de data/hora após atualizações de bibliotecas, o tratamento de registros corrompidos com coordenadas geográficas nulas ou fora dos limites territoriais do Brasil e a correção de erros de divisão por zero em trechos recém-inaugurados.
2. Manutenção adaptativa: compreende modificações no software para acompanhar alterações no ambiente operacional externo. Isso inclui a adequação a novas versões do Sistema Nacional de Viação do DNIT, a atualização de caminhos de acesso a repositórios do governo federal e o ajuste das variáveis do modelo para refletir mudanças na legislação de trânsito.
3. Manutenção perfectiva: envolve melhorias planejadas para aprimorar o desempenho computacional e a acurácia analítica do sistema. Exemplos dessa frente incluem a otimização dos tempos de junção espacial e a inclusão futura de novas fontes de dados, tais como dados de estações telemétricas do INMET e medições de tráfego em praças de pedágio sob concessão federal.

### 17.2. Plano de evolução técnica e conteinerização

O pipeline foi concebido inicialmente em ambiente interativo baseado em cadernos computacionais (Jupyter Notebooks) executados sobre a plataforma Google Colab. Embora esse ambiente ofereça agilidade para exploração estatística e testes rápidos, ele impõe restrições para implantação sustentável em produção devido à necessidade de intervenção humana para disparar rotinas e à volatilidade do armazenamento temporário.

O plano de evolução arquitetural estabelece a migração progressiva do sistema para uma estrutura de engenharia de software automatizada:

1. Modularização do código-fonte: extração das rotinas dos cadernos para pacotes estruturados em Python, separando claramente os módulos de extração, limpeza, engenharia de atributos, treino de modelos e geração de relatórios.
2. Conteinerização com Docker: criação de imagens padronizadas contendo as dependências de sistema operacional, versões fixadas do interpretador Python e bibliotecas analíticas bloqueadas no arquivo `requirements.txt`. Essa medida garante a portabilidade integral e elimina falhas causadas por variações de ambiente entre máquinas de desenvolvimento e servidores de produção.
3. Orquestração com ferramentas de fluxo de trabalho: substituição dos disparos manuais por esteiras orquestradas em ferramentas como Apache Airflow ou Prefect. A orquestração permitirá a execução programada de tarefas com controle de dependências, tratamento de retentativas em caso de falhas de rede e emissão de alertas automáticos para as equipes responsáveis.
4. Armazenamento colunar otimizado: transição da persistência intermediária baseada em arquivos CSV de grande porte para o formato colunar Parquet particionado por ano e por estado. Essa mudança reduz a pegada de armazenamento em mais de 60% e acelera o tempo de leitura para as rotinas analíticas e para as consultas de atualização do Power BI.
