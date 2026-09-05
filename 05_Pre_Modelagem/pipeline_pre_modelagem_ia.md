# Pipeline de Engenharia de Atributos e Pré-Modelagem (Pós-Integração DNIT)

Este documento atua como um guia de instrução arquitetural voltado para **Agentes e Desenvolvedores IA**. Ele descreve todas as decisões de engenharia de dados, agregações e transformações realizadas *após* a junção das bases PRF + DNIT (`df_master_tcc.csv`) e *antes* do treinamento do modelo Random Forest (`df_model_ready_tcc.csv`).

---

## 1. Escopo e Propósito

Após a fusão das bases (Merge/ETL), o dataset encontrava-se em nível transacional (uma linha = um acidente). O modelo previsor de Machine Learning, no entanto, precisava prever a **frequência de acidentes em um dado trecho durante um dado período**. Toda a etapa de pré-modelagem consistiu em resolver desafios estatísticos (colapso da base) e preparar as matrizes para o framework Scikit-Learn.

---

## 2. Tratamento do DNIT (Resolução de Dados Ausentes)

Antes de engatilhar os algoritmos estatísticos, o pipeline resolveu a latência e a incompletude da base DNIT usando duas diretrizes:

### A. Temporal Backfill (Retroatividade 2019-2021)
- **Problema:** A base da PRF cobria acidentes a partir de 2019, mas dados viáveis do DNIT só começavam em 2022.
- **Estratégia:** O script espelhou as condições físicas/estruturais medidas em **Janeiro de 2022** projetando-as linearmente no tempo para **2019, 2020 e 2021**.
- **Justificativa:** Como a geometria da via e o pavimento são de decomposição lenta, "congelar" a fotografia de Jan/2022 para trás garantiu o aproveitamento do volume formidável de acidentes da PRF sem inflacionar a base com lixo sintético (Interpolação cega).

### B. Sentinela Matemático (`-1.0`)
- **Problema:** O DNIT não cobre (não captura amostras) de todas as rodovias federais. Usar `NaN` quebraria a árvore de regressão, e usar "Mean/Median Imputer" corromperia a amostra, forçando uma via péssima/não monitorada a receber notas neutras irreais.
- **Estratégia:** Foi aplicado o sinal unificador `-1.0` (Sentinela) em todos as features numéricas contínuas faltantes (ICC, ICP, ICM).
- **Justificativa:** Essa manobra intencional forçou o Random Forest a criar um **nó de decisão exclusivo** alertando que "A Via Desconhecida possui um Risco Específico."

---

## 3. Agrupamento Espaço-Temporal (Combate ao *Zero-Inflated*)

- **O Problema:** Agrupar acidentes por dia exato ou km exato (1 km) causava o fenômeno "Zero-Inflated" — 99% das linhas na matriz ficavam com `Y = 0` (dias e quilômetros onde felizmente nenhum acidente rolou), inviabilizando previsões de probabilidade contínua no Random Forest Regressor.
- **A Solução:** Redução de dimensionalidade via Bucketing.
  - O eixo do Tempo foi agrupado por **Mês** (`data_inversa.dt.month`).
  - O eixo do Espaço foi unificado em **Blocos de 10 Km** (`np.floor(km/10)*10`).
- **O Resumo:** As fatias preditivas deixaram de ser "Quantos acidentes tem no km 5 hoje?" e se tornaram `"Quantos acidentes tem entre o km 0-10 da BR-101 no mês de Janeiro?"`.

---

## 4. Colapso das Estruturas e Agregações de Features

Ao converter o dataset de `Transacional` para `Agrupado (Target=Count)`, todas as variáveis exógenas precisaram ser aglutinadas em médias que representassem aquele trecho (Bucket de 10km) e aquele mês:

1. **Variáveis Contínuas (DNIT, Índices, Contagem de Pessoas/Vítimas):** Agregadas pela `Média` (`mean`). A média do ICP em 10km no mês, por exemplo.
2. **Features Booleanas/OneHot Anteriores (`is_*`):** Agregadas pela transposição de `Soma` (`sum`). Retorna a incidência daquele booleano no trecho (ex: quantos acidentes com chuva rolaram naquele bucket).
3. **Variáveis Categóricas / Estrutura da Via (Estado, Sentido, Clima Base, Tipo Pista):** Agregadas extraindo-se a Primeira ocorrência/Moda do trecho (`first`).

Os agregadores agrupados foram cruzados no Dataframe final pela `.groupby(['ano', 'mes', 'br', 'trecho_10km']).size().reset_index()` para gerar a grande variável Target da modelagem.

---

## 5. Variável Resposta (Target Y)

- A coluna alvo do TCC (Variável Dependente) passou a se chamar **`quantidade_acidentes`**.
- Ela expressa a Frequência Absoluta e é estritamente Numérica/Contínua, designando o dataset a um algoritmo supervisionado: `RandomForestRegressor`.

---

## 6. Codificação para Frameworks Clássicos (Scikit-Learn)

Como os frameworks da SciPy/Scikit-Learn são restritos (só entendem matrizes numéricas, diferentemente do CatBoost e LightGBM), duas tratativas protetivas foram postas no final do script `pre_modeling.py`:

- **Abordagem de Alta Cardinalidade:** Variáveis do tipo Categoria (`object` ou string) com mais de 100 classes únicas (nomes exatos de cidades, descrições lexicais puras) foram expurgadas (`drop()`), prevenindo o colapso e explosão de OVM/RAM (`out-of-memory`) no computador local.
- **One-Hot Encoding Progressivo:** A conversão Dummies (`pd.get_dummies`) foi executada ativamente (`dummy_na=False`, `drop_first=False`) mapeando clima, turno e topologias de pista em vetores binários `1/0` `float` puros, garantindo estabilidade algorítmica para os galhos da matriz de decisão.

---

## 7. Prevenção de Data Leakage (Target Leakage) na Modelagem

Uma etapa crítica na virada da pré-modelagem para o fit da árvore de regressão (Train-Test Split) é o isolamento metodológico de variáveis (Feature Selection). Vetores numéricos que descrevem as **consequências** do sinistro vazam a resposta preditiva para o futuro. 

Portanto, **as seguintes colunas SÃO SUMARIAMENTE EXCLUÍDAS da matriz de atributos (`X`)**:
- `pessoas`, `mortos`, `feridos_leves`, `feridos_graves`, `feridos`, `veiculos`
*(Justificativa: Não é factível prever a frequência de acidentes de uma rodovia usando como parâmetro a contagem de vítimas corporais, pois as vítimas só existem cronologicamente após a colisão ocorrer).*

**Quais colunas DEVEMOS TER na Matriz Preditiva (`X`)?**
A Inteligência Artificial baseia seus *splits* decisórios inteiramente em **Variáveis Exógenas Prévias** (fatores incondicionais que existem inertes no trecho antes do veículo passar). O subset correto de modelagem deve focar em:
1. **Ativos DNIT e Degradação Física:** Índices oficias (`icp`, `icc`, `icm`), panela/buraco, trincamento, visibilidade de sinalização horizontal, roçada e drenagem.
2. **Geometria da Rodovia Estrutural:** Traçado predominante (`curva`, `reta`, `declive`), Tipo de Pista (`simples`, `Múltipla`, `dupla`).
3. **Clima e Iluminação:** Condição Meteorológica (`chuva`, `nevoeiro`, `céu claro`), e Luminosidade/Fase do Dia.
4. **Calendário Base:** Ocorreu em dia útil ou final de semana? Tem feriado nacional? 

---

## 8. Decisão Arquitetural: Ausência de Normalização (Scale-Invariance)

Diferente de Redes Neurais (Deep Learning) ou Regressões Lineares clássicas, o algoritmo central escolhido para o TCC foi o **RandomForestRegressor** (conjunto de árvores de decisão não-paramétricas).

Por restrição metodológica deliberada documentada no projeto:
- **NENHUMA** técnica de *Scaling* (`StandardScaler`, `MinMaxScaler`) foi aplicada às variáveis numéricas contínuas (como o ICP ou ICC, que variam de 0 a 100).
- Árvores de decisão lidam puramente com cortes de ganho de informação (*Information Gain/MSE Split*), tornando o modelo nativamente imune a escalas discrepantes (**Scale-Invariant**).
- Preservar a magnitude bruta dos valores viabiliza a etapa final de interpretabilidade (Machine Learning Explicável) no script `analise_dnit_infraestrutura.py`. Sem escalas distorcidas, o **SHAP Value** e o **PDP (Partial Dependence Plot)** conseguem extrair regras legíveis e factuais para o humano (ex: "A rodovia ficou perigosa quando o ICP caiu para 20", ao invés de ler um esotérico "Z-score caiu para -1.89").

---

## 9. Exportação Final

Todos os buckets validados sem ruídos que sobreviveram aos funis de merge salvaram sua volumetria em `df_model_ready_tcc.csv`. Essa base (onde os `100%` da nossa consulta repousam e onde o modelo roda o `Ceteris Paribus` a posteriori) está completamente "limpa", balanceada, numerificada e sem buracos nulos implodindo a memória do SciPy.
