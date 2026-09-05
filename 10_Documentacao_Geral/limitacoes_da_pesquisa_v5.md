# Limitações da Pesquisa e Diretrizes para Trabalhos Futuros

Este documento formaliza as limitações metodológicas identificadas na modelagem preditiva do Trabalho de Conclusão de Curso (TCC), detalhando o processo investigativo do Modelo V5, as restrições inerentes às bases de dados da PRF e do DNIT e o encadeamento lógico para investigações futuras.

---

## 1. Limitações Metodológicas da Pesquisa

O desenvolvimento de modelos estatísticos sobre dados abertos de segurança pública impõe desafios severos de engenharia. Durante a construção do Modelo V5, a busca pela superação de barreiras estruturais da base revelou uma dinâmica clássica da Ciência de Dados: a tentativa de contornar uma limitação matemática frequentemente introduz efeitos colaterais na etapa seguinte do pipeline.

```
┌───────────────────────────────┐
│   Problema Original (PRF):    │
│  Zero-Inflation & Esparsidade │
└──────────────┬────────────────┘
               │  Solução Aplicada:
               ▼
┌───────────────────────────────┐
│  Buckets Espaço-Temporais     │
│       (10 km / Mês)           │
└──────────────┬────────────────┘
               │  Efeitos Colaterais Emergentes:
               ├─────────────────────────────────────────────┐
               ▼                                             ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐
│   Viés de Sobrevivência       │             │       Target Leakage          │
│(Condicionamento a Y ≥ 1;      │             │(Soma de traçado como proxy    │
│ trechos seguros descartados)  │             │ contábil do total de sinistros)│
└───────────────────────────────┘             └───────────────────────────────┘
```

### 1.1. O Efeito do Agrupamento Espaço-Temporal: A Troca de Gargalos

A distribuição bruta dos boletins de ocorrência da PRF apresenta esparsidade extrema quando analisada em escala métrica linear contínua. Em um determinado dia e quilômetro isolado de rodovia, a probabilidade pontual de colisão é virtualmente nula, gerando uma inflação massiva de zeros (*zero-inflation*) superior a 99% das observações. 

Para viabilizar a convergência de algoritmos baseados em árvores de decisão, estruturou-se a agregação da malha em *buckets* espaço-temporais de 10 quilômetros por mês. A solução atingiu o objetivo imediato de densificar a variável resposta. Contudo, essa consolidação acarretou dois desdobramentos críticos:

#### A. O Viés de Sobrevivência e a Perda dos Trechos Seguros
A agregação foi executada aplicando operações de agrupamento (`groupby`) sobre a tabela histórica de sinistros registrados pela PRF. Como consequência direta:
* Apenas trechos e meses que registraram **pelo menos um acidente** integraram a matriz final de treinamento.
* A variável alvo assumiu valor mínimo unitário ($Y \ge 1$), excluindo da modelagem os milhares de segmentos federais que operaram com total segurança ao longo do ano.

O modelo não estima a probabilidade incondicional de acidentes na malha rodoviária federal ($P(Y \mid X)$). Ele opera estritamente condicionado à preexistência de sinistralidade ($E[Y \mid Y \ge 1, X]$). O sistema perde a capacidade de discriminar ativamente por que um trecho preservou zero ocorrências durante doze meses consecutivos.

#### B. A Ilusão das Métricas: Decomposição Contábil via Somas de Traçado
Na fase de pré-modelagem, as variáveis categóricas de traçado geométrico da via (`is_tracado_via_Reta`, `is_tracado_via_Curva`, `is_tracado_via_Intersecao`) e de calendário (`is_final_semana`) foram computadas no nível do evento e consolidadas no *bucket* por meio de somatórios:

$$\text{is\_tracado\_via\_Reta}_{\text{bucket}} = \sum \text{ocorrências em reta naquele mês}$$

$$\text{is\_tracado\_via\_Curva}_{\text{bucket}} = \sum \text{ocorrências em curva naquele mês}$$

Ao receber essas variáveis somadas como preditores, o algoritmo Random Forest encontrou uma identidade contábil direta:

$$\sum \text{acidentes}_{\text{geometrias}} \approx \text{quantidade\_acidentes (Target Y)}$$

Esse mecanismo explica por que o Modelo V5 alcançou um coeficiente de determinação aparentemente excepcional ($R^2 = 0{,}9726$ e $\text{MAE} = 0{,}4181$) mesmo diante de uma partição temporal rigorosa (treino 2019–2023 / teste 2024–2025). As variáveis preditoras do conjunto de teste já embutiam a contagem agregada dos sinistros daquele mês. O modelo operou como um excelente reconciliador de perfis físicos de sinistros ocorridos, mas com limitações para atuar como preditor causal prévio (*ex-ante*).

---

### 1.2. Cobertura Parcial e Valor Sentinela na Base DNIT

A integração das medições de engenharia viária do DNIT aos registros da PRF revelou assimetrias severas de infraestrutura de dados:

1. **Lacuna Amostral de 82,3%:** Apenas 17,7% dos buckets georreferenciados possuíam avaliação estrutural contemporânea do pavimento cadastrada pelo DNIT. A grande maioria da malha federal não dispunha de séries contínuas para o Índice de Condição da Superfície (ICP) e Índice de Gravidade Global (IGG).
2. **Comportamento do Valor Sentinela (`-1.0`):** Para não distorcer a variância populacional por meio de imputações arbitrárias de média ou mediana, os trechos desprovidos de monitoramento receberam o sentinela `-1.0`. A topologia de divisão das árvores do Random Forest isolou esse valor em ramos próprios. Entretanto, esse artifício transformou a feature contínua em um separador binário de governança: o modelo aprendeu a distinguir locais fiscalizados pelo DNIT daqueles negligenciados, atenuando a identificação da curva de degradação física suave do asfalto.
3. **Ofuscamento de Atributos (*Feature Overshadowing*):** A presença de somatórios quase perfeitos de acidentes nas colunas de traçado fez com que o algoritmo priorizasse cortes de impureza (MDI) na geometria viária (respondendo por mais de 60% do ganho de informação). Variáveis como `icp`, `icc` e `icm` receberam importâncias residuais entre 1% e 2%, limitando a sensibilidade pura do asfalto nas simulações.

---

### 1.3. Restrições na Simulação de Cenários (*Ceteris Paribus*)

Nos testes de sensibilidade operacional (intervenções simuladas de recapeamento ou alterações de fatores climáticos), a manipulação isolada de variáveis revelou uma compressão nas previsões:
* Ao fixar um atributo de cenário como `is_tracado_via_Curva = 1`, o modelo treinado sob somas absolutas interpretou a entrada como a ocorrência de apenas uma colisão naquela condição, em vez de tratá-la como a característica geométrica integral do segmento.
* Esse fator ancorou as estimativas simuladas em uma faixa estreita entre 1,36 e 1,60 acidentes por trecho/mês, exigindo calibrações adicionais de intercepto médio para extrapolações táticas.

---

### 1.4. Granularidade das Variáveis Meteorológicas

A caracterização climática baseou-se nos apontamentos visuais preenchidos pelos policiais rodoviários federais no momento do boletim (ex.: "Céu Claro", "Chuva"). Embora reflitam o contexto imediato da colisão, esses registros padecem de subjetividade humana e não capturam grandezas físicas contínuas de precipitação pluviométrica (milímetros por hora), pressão atmosférica ou acúmulo de lâmina d'água na pista.

---

## 2. Diretrizes e Ganchos para Trabalhos Futuros

As fronteiras encontradas nesta dissertação estabelecem as bases metodológicas para o avanço da linha de pesquisa em segurança viária e inteligência artificial:

### 2.1. Matriz Cartesiana Completa com Presença de Zeros Reais
A evolução natural do pipeline exige a transição de uma base orientada a sinistros para uma **grade espaço-temporal exaustiva**. Deve-se mapear todos os trechos homogêneos da malha rodoviária federal multiplicados por todos os meses do ano civil:
* Trechos sem registros no período receberão explicitamente a contagem $Y = 0$.
* Essa formulação permitirá ao modelo aprender os fatores condicionantes da segurança e da imunidade viária, mitigando o viés de sobrevivência.

### 2.2. Desacoplamento dos Atributos Cadastrais
As características físicas do leito viário devem emanar de dados cadastrais estáticos e contínuos do DNIT, jamais da agregação de acidentes. Propõe-se a substituição das dummies de traçado por:
* Grau e raio médio de curvatura do segmento (em metros).
* Rampa longitudinal e inclinação média de declive (em percentual).
* Largura útil de pista e presença de acostamento pavimentado.

Dessa forma, as features representarão a topografia real da rodovia de forma imutável antes da ocorrência de qualquer colisão.

### 2.3. Modelos Nativos para Dados de Contagem com Zero-Inflation
Para operar sobre a grade completa com predominância de zeros, sugere-se a experimentação com arquiteturas estatísticas e de aprendizado estruturadas especificamente para esse perfil distribucional:
* **Modelos Hurdle e Zero-Inflated:** Arquiteturas *Zero-Inflated Negative Binomial (ZINB)* e *Zero-Inflated Poisson (ZIP)*, que dividem o problema em dois estágios: um classificador binário ($P(Y=0)$ versus $P(Y>0)$) acoplado a um regressor de contagem truncada.
* **Funções de Perda Tweedie:** Treinamento de gradiente boosting (*LightGBM* ou *XGBoost*) com distribuição Tweedie, adequada para modelar densidades com massa pontual em zero aliada a caudas contínuas positivas.

### 2.4. Integração Sensorial Contínua (INMET e Radares de Tráfego)
Substituir o preenchimento manual de clima e a ausência de volumetria de veículos por integrações automatizadas:
* Cruzamento das coordenadas geográficas das rodovias com estações meteorológicas automáticas do Instituto Nacional de Meteorologia (INMET), extraindo precipitação acumulada e visibilidade ótica.
* Inclusão do Volume Diário Médio (VDM) obtido por sensores de contagem e praças de pedágio, transformando a estimativa de acidentes absolutos em uma taxa de risco ponderada por milhão de veículos-quilômetro transitados.
