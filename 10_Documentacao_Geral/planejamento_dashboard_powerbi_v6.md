# Especificação de Engenharia de Dashboard (Power BI) — Arquitetura V6

**Projeto:** Painel Integrado de Inteligência Viária e Predição de Sinistralidade (PRF + DNIT)  
**Base Analítica:** `df_model_ready_v6.csv` (475.020 observações, 5.655 trechos de 10 km, 84 meses)  
**Modelo Preditivo:** Random Forest Poisson Deviance V6 ($R^2 = 0{,}8678$ Out-of-Time 2024–2025)  
**Objetivo:** Prover uma solução analítica e prescritiva de Business Intelligence para suportar a tomada de decisão da Polícia Rodoviária Federal (alocação de viaturas e radares) e do DNIT (priorização de investimentos em pavimentação).

---

## 1. Arquitetura da Informação e Casos de Uso por Persona

O painel é estruturado em quatro páginas concebidas para perfis de consumo e responsabilidades distintas:

```
                                  PAINEL INTEGRADO PRF + DNIT
                                               |
     +-------------------+---------------------+---------------------+--------------------+
     |                   |                     |                     |                    |
  PÁGINA 1            PÁGINA 2              PÁGINA 3              PÁGINA 4             TOOLTIP PAGE
Visão Estratégica   Operações Táticas     Engenharia Viária     Auditoria de XAI      Detalhe de Trecho
  (Executivo)          (PRF Rondas)       & Obras (DNIT)        e ML (Auditores)       (Drill-Through)
```

| Página | Persona Alvo | Frequência de Uso | Pergunta Central de Negócio Respondida |
| :--- | :--- | :--- | :--- |
| **1. Visão Estratégica Nacional** | Direção-Geral PRF, Diretoria DNIT, Ministério dos Transportes | Mensal / Trimestral | *Como evolui o risco geral da malha e qual o saldo predito de vidas preservadas?* |
| **2. Alocação Tática de Efetivo** | Inspetorias e Delegacias Operacionais da PRF | Semanal / Pré-Feriado | *Em quais trechos exatos de 10 km o risco vai explodir neste próximo feriadão/mês?* |
| **3. Priorização de Obras e Pavimento** | Engenharia de Manutenção do DNIT, Superintendências | Semestral / Orçamentária | *Onde cada real investido em recapeamento evita o maior número de mortes e traumas?* |
| **4. Auditoria de Machine Learning** | Cientistas de Dados, Analistas de BI e Auditoria Pública | Contínua / Monitoramento | *O modelo mantém calibração, distribuição de resíduos e assertividade fora do tempo?* |

---

## 2. Modelagem Relacional (Star Schema)

Para garantir alto desempenho de processamento em memória (*VertiPaq Engine*), o modelo abandona estruturas *flat* e consolida um esquema estrela estrito:

```
             +-----------------------+              +-----------------------+
             |      dCalendario      |              |     dTrechosMalha     |
             |-----------------------|              |-----------------------|
             | PK_Data               | 1          1 | PK_Trecho             |
             | Ano                   |              | Codigo_BR             |
             | Mes_Num               |              | KM_Inicial            |
             | Mes_Nome              |              | KM_Final              |
             | Trimestre             |              | UF                    |
             | Dias_No_Mes           |              | Regiao                |
             | Dias_Fim_Semana       |              | Latitude_Inicio       |
             | Dias_Feriado          |              | Longitude_Inicio      |
             | Estacao_Ano           |              | Superintendencia_DNIT |
             +-----------+-----------+              +-----------+-----------+
                         |                                      |
                         | 1                                    | 1
                         |                                      |
                         |           +--------------------+     |
                         +---------> |   fModelReadyV6    | <---+
                                   N |--------------------| N
                                     | FK_Data            |
                                     | FK_Trecho          |
                                     | quantidade_acidentes|
                                     | pred_v6            |
                                     | total_mortos       |
                                     | total_feridos_graves|
                                     | total_veiculos     |
                                     | is_zero_real       |
                                     | icp                |
                                     | icc                |
                                     | icm                |
                                     | has_dnit_data      |
                                     | acidentes_lag_1m   |
                                     | acidentes_lag_2m   |
                                     | media_movel_3m     |
                                     | acidentes_lag_12m  |
                                     +--------------------+
```

### Dicionário das Tabelas
1. **`fModelReadyV6` (Fato Central):**
   - Granularidade: 1 linha por Trecho de 10 km por Mês/Ano ($475.020$ linhas).
   - Relacionamento com `dCalendario`: Chave `FK_Data` no formato inteiro `YYYYMM01` (1:N, filtro único).
   - Relacionamento com `dTrechosMalha`: Chave `FK_Trecho` composta `BR_KM_UF` (1:N, filtro único).
2. **`dCalendario` (Dimensão Temporal):**
   - Intervalo: 01/01/2019 a 31/12/2025 (84 meses).
   - Contém flags de calendário civil: `dias_fim_semana`, `dias_feriado`, `estacao_ano`.
3. **`dTrechosMalha` (Dimensão Espacial):**
   - 5.655 linhas correspondentes aos segmentos discretos de 10 km.
   - Contém atributos geográficos de roteamento, microrregião, UF e jurisdição de delegacias.

---

## 3. Catálogo de Visuais por Página

### PÁGINA 1: VISÃO ESTRATÉGICA E EXECUTIVA

#### Visual 1.1: Cartões KPI em Destaque (Novo Cartão Visual)
- **Tipo:** New Card Visual com cartões agrupados e micrográficos Sparkline.
- **Campos e Métricas:**
  - Card A: `[Total_Acidentes_Real]` (Sinistros Observados no período filtrado).
  - Card B: `[Total_Acidentes_Predito]` acompanhado da variação `[Desvio_Percentual_GAP]`.
  - Card C: `[Taxa_Letalidade_Pct]` (Percentual de acidentes com óbito).
  - Card D: `[Taxa_Vitimas_Severas_Pct]` (Percentual de acidentes com mortos ou feridos graves).
  - Card E: `[ICP_Medio_Malha]` (Média do pavimento nos trechos monitorados).
- **Racional Analítico:** Apresenta em segundos a fotografia executiva consolidada do sistema viário.

#### Visual 1.2: Série Temporal Consolidada Brasil — Previsto vs. Real
- **Tipo:** Gráfico de Linhas com marcadores duplos.
- **Eixo X:** `dCalendario[Ano_Mes]` (Linha contínua cronológica).
- **Eixo Y:**
  - Linha 1 (Vermelho Sólido `#DC2626`, largura 2.5px): `[Total_Acidentes_Real]`.
  - Linha 2 (Azul Tracejado `#2563EB`, largura 2.5px): `[Total_Acidentes_Predito]`.
- **Dica de Ferramenta (Tooltip):** Mês/Ano, Volume Real, Volume Previsto, Resíduo do Mês, Quantidade de Feriados no Mês.
- **Racional Analítico:** Responde: *"O modelo Poisson acompanha os picos de sazonalidade de dezembro e julho e as quedas pontuais sem desvios estruturais?"*.

#### Visual 1.3: O Painel do Efeito Peltzman e Severidade
- **Tipo:** Gráfico de Colunas Agrupadas com Linha Secundária.
- **Eixo X:** `Faixa_ICP_Descricao` (`Péssimo [0-20]`, `Ruim [21-40]`, `Regular [41-60]`, `Bom [61-80]`, `Ótimo [81-100]`).
- **Eixo Y Colunas (Azul e Ambar):** `[Frequencia_Media_Acidentes]` (evidencia o paradoxo de estabilidade da contagem em asfalto bom).
- **Eixo Y Linha (Vermelho Escuro `#991B1B`):** `[Taxa_Letalidade_Pct]` (evidencia o salto de $7{,}32\%$ no Bom para $10{,}50\%$ no Péssimo).
- **Racional Analítico:** Fornece a justificativa científica inquestionável para alocação de recursos em recomposição de pista.

#### Visual 1.4: Mapa Geográfico de Calor da Malha Federal
- **Tipo:** Azure Map / Shapefile das rodovias federais com formatação por gradiente.
- **Camada Geográfica:** Coordenadas lat/long dos trechos de 10 km.
- **Espessura da Linha / Tamanho:** `[Total_Acidentes_Real]`.
- **Cor do Gradiente:** `[Taxa_Letalidade_Pct]` (Verde $\rightarrow$ Amarelo $\rightarrow$ Vermelho).
- **Tooltip de Página:** Mini-relatório suspenso detalhando o histórico do trecho selecionado.
- **Racional Analítico:** Identifica em mapa os corredores geográficos prioritários de intervenção nacional.

---

### PÁGINA 2: OPERAÇÕES TÁTICAS E POLICIAMENTO (PRF)

#### Visual 2.1: Ranking Top 20 Trechos Mais Perigosos da Malha
- **Tipo:** Gráfico de Barras Horizontais ordenado de forma decrescente.
- **Eixo Y:** `dTrechosMalha[Identificador_Trecho]` (ex.: "BR-101/RJ km 310-320", "BR-381/MG km 340-350").
- **Eixo X:** `[Media_Mensal_Predita_V6]`.
- **Formatação Condicional de Barras:**
  - $> 5{,}0$ acidentes/mês: Vermelho Intenso (`#B91C1C`).
  - Entre $2{,}0$ e $5{,}0$: Âmbar Alerta (`#D97706`).
  - $< 2{,}0$: Azul Seguro (`#3B82F6`).
- **Tooltip:** `BR`, `Trecho km`, `UF`, `Predição Próximo Mês`, `Média Histórica Real`, `ICP do DNIT`, `Total de Óbitos Históricos`.
- **Racional Analítico:** Entrega à chefia de operações da PRF a lista imediata dos segmentos onde as viaturas devem permanecer posicionadas em horários de pico.

#### Visual 2.2: Matriz de Calor — Risco por Mês e Concentração de Calendário
- **Tipo:** Matriz com formatação de preenchimento condicional (*Heatmap*).
- **Linhas:** `dCalendario[Mes_Nome]` (Janeiro a Dezembro ordenado por `Mes_Num`).
- **Colunas:** `Faixa_Dias_Fim_Semana` (8 dias, 9 dias, 10 dias).
- **Valores:** `[Media_Mensal_Predita_V6]`.
- **Escala de Cor:** Verde Menta (`#E6F4EA`) a Vermelho Rubro (`#FCE8E6`).
- **Racional Analítico:** Responde: *"Como a escala de policiais rodoviários deve ser reforçada em meses com 10 dias de fim de semana associados a feriados prolongados?"*.

#### Visual 2.3: Dispersão Inércia vs. Predição (Detector de Anomalias)
- **Tipo:** Gráfico de Dispersão (Scatter Plot).
- **Eixo X:** `fModelReadyV6[acidentes_lag_1m]` (Comportamento no mês anterior).
- **Eixo Y:** `fModelReadyV6[pred_v6]` (Risco estimado para o mês corrente).
- **Tamanho da Bolha:** `fModelReadyV6[total_mortos]`.
- **Legenda:** `dTrechosMalha[Regiao]`.
- **Racional Analítico:** Trechos situados muito acima da diagonal $y = x$ indicam locais onde o risco predito dispara por fatores exógenos (sazonalidade turística ou calendário), exigindo patrulhamento proativo imediato.

---

### PÁGINA 3: ENGENHARIA VIÁRIA E PRIORIZAÇÃO DE OBRAS (DNIT)

#### Visual 3.1: Matriz de Retorno Social da Pavimentação (Quadrantes de Decisão)
- **Tipo:** Gráfico de Dispersão com Linhas de Referência de Médias (Quatro Quadrantes).
- **Eixo X:** `[ICP_Medio]` (Invertido: da direita para a esquerda, de 100 a 0 — quanto mais à esquerda, pior a via).
- **Eixo Y:** `[Vitimas_Severas_Evitaveis_Simuladas]` (Volume de mortos e sequelados que seriam poupados com recomposição para ICP 70).
- **Tamanho da Bolha:** `[Total_Veiculos_Envolvidos]` (medida indireta da intensidade de exposição).
- **Divisão dos Quadrantes:**
  - *Quadrante Prioridade Ouro (Canto Superior Esquerdo):* Asfalto em ruínas ($ICP \le 20$) e alto volume de vidas em risco. **Intervenção:** Recapeamento emergencial contratado no primeiro trimestre.
  - *Quadrante Prioridade Prata (Canto Inferior Esquerdo):* Asfalto destruído, mas rodovia de baixíssimo fluxo. **Intervenção:** Tapa-buraco paliativo e sinalização horizontal.
- **Racional Analítico:** Fornece às superintendências regionais a fórmula matemática de máxima eficiência social do recurso orçamentário.

#### Visual 3.2: Comparador Antes vs. Depois da Restauração Viária
- **Tipo:** Gráfico de Barras Agrupadas Duplas.
- **Eixo Y:** Top 10 Trechos com Pavimento Crítico ($ICP \le 20$).
- **Barra 1 (Vermelha):** `[Sinistros_Projetados_Cenario_Atual]` (ICP degradado).
- **Barra 2 (Azul):** `[Sinistros_Projetados_Pos_Reforma]` (Simulação de restauração com ICP elevado a 70).
- **Rótulo de Dados:** Percentual de redução estimado e número absoluto de sinistros evitados.
- **Racional Analítico:** Demonstra aos órgãos de controle (TCU, CGU) o benefício tangível projetado antes da abertura de licitações rodoviárias.

---

### PÁGINA 4: AUDITORIA DE MACHINE LEARNING E XAI

#### Visual 4.1: Curva de Dispersão Real vs. Predito no Teste Fora do Tempo
- **Tipo:** Scatter Plot com Amostragem Aleatória de 10.000 pontos de 2024–2025.
- **Eixo X:** `quantidade_acidentes` (Real).
- **Eixo Y:** `pred_v6` (Predição).
- **Linha de Referência:** Reta diagonal pontilhada $y = x$.
- **Racional Analítico:** Prova visual da ausência de compressão de variância e calibração linear das previsões.

#### Visual 4.2: Histograma de Distribuição dos Resíduos
- **Tipo:** Gráfico de Colunas em Bins (de -10 a +10).
- **Eixo X:** Faixa Residual ($Y - \hat{Y}$).
- **Eixo Y:** Contagem de Observações.
- **Linha de Referência:** Média dos resíduos em $x = 0$ (Verde sólida).
- **Racional Analítico:** Comprova a homocedasticidade e a ausência de viés sistemático de superestimação ou subestimação.

#### Visual 4.3: Ranking MDI de Importância das Variáveis (Feature Importance)
- **Tipo:** Barras Horizontais.
- **Eixo Y:** Variáveis do Modelo V6.
- **Eixo X:** Percentual de Redução de Impureza (% MDI).
- **Cores Segmentadas por Família:**
  - Azul: Lags Temporais e Inércia ($t-1, t-2, t-3, t-12$).
  - Verde: Métricas de Engenharia DNIT ($ICP, ICC, ICM, has\_dnit\_data$).
  - Laranja: Variáveis de Calendário e Clima ($dias\_fim\_semana, dias\_feriado$).
- **Racional Analítico:** Transparência algorítmica total exigida para sistemas públicos de apoio à decisão (XAI).

---

## 4. Medidas DAX Oficiais do Modelo

As medidas a seguir foram escritas em DAX padronizado, testadas para o modelo tabular e prontas para implantação:

```dax
/* =========================================================================
   GRUPO 1: VOLUMETRIA E RESULTADOS TRANSACIONAIS DA PRF
   ========================================================================= */

// 1.1 Total Real de Sinistros Observados
Total_Acidentes_Real = 
SUM(fModelReadyV6[quantidade_acidentes])

// 1.2 Média de Sinistros por Trecho/Mês
Media_Acidentes_Real_Trecho = 
AVERAGE(fModelReadyV6[quantidade_acidentes])

// 1.3 Total de Vítimas Fatais Registradas
Total_Mortos = 
SUM(fModelReadyV6[total_mortos])

// 1.4 Total de Feridos Graves
Total_Feridos_Graves = 
SUM(fModelReadyV6[total_feridos_graves])

// 1.5 Total de Vítimas Severas (Óbitos + Traumas Graves)
Total_Vitimas_Severas = 
[Total_Mortos] + [Total_Feridos_Graves]

// 1.6 Taxa de Letalidade Real (%)
Taxa_Letalidade_Pct = 
DIVIDE(
    [Total_Mortos],
    [Total_Acidentes_Real],
    0
) * 100

// 1.7 Índice de Vítimas Severas (%)
Taxa_Vitimas_Severas_Pct = 
DIVIDE(
    [Total_Vitimas_Severas],
    [Total_Acidentes_Real],
    0
) * 100


/* =========================================================================
   GRUPO 2: PREVISÕES DO MODELO V6 E INDICADORES DE CALIBRAÇÃO
   ========================================================================= */

// 2.1 Total de Acidentes Previstos pelo Modelo V6
Total_Acidentes_Predito = 
SUM(fModelReadyV6[pred_v6])

// 2.2 Média de Acidentes Previstos por Trecho/Mês
Media_Mensal_Predita_V6 = 
AVERAGE(fModelReadyV6[pred_v6])

// 2.3 Desvio Residual Bruto (Real - Previsto)
Desvio_Residual_Total = 
[Total_Acidentes_Real] - [Total_Acidentes_Predito]

// 2.4 GAP Percentual Entre Real e Previsto
Desvio_Percentual_GAP = 
DIVIDE(
    [Total_Acidentes_Real] - [Total_Acidentes_Predito],
    [Total_Acidentes_Real],
    0
) * 100

// 2.5 Erro Absoluto Médio (MAE DAX em nível de observação)
MAE_Global = 
AVERAGEX(
    fModelReadyV6,
    ABS(fModelReadyV6[quantidade_acidentes] - fModelReadyV6[pred_v6])
)

// 2.6 Erro Médio Absoluto Específico em Trechos Seguros (Zeros Reais)
MAE_Trechos_Seguros = 
CALCULATE(
    [MAE_Global],
    fModelReadyV6[quantidade_acidentes] = 0
)


/* =========================================================================
   GRUPO 3: INFRAESTRUTURA DNIT E SIMULAÇÃO DE IMPACTO (XAI)
   ========================================================================= */

// 3.1 Média do Índice de Condição do Pavimento (apenas dados válidos)
ICP_Medio_Malha = 
CALCULATE(
    AVERAGE(fModelReadyV6[icp]),
    fModelReadyV6[has_dnit_data] = 1
)

// 3.2 Percentual de Cobertura de Dados do DNIT na Malha
Percentual_Cobertura_DNIT = 
DIVIDE(
    COUNTROWS(FILTER(fModelReadyV6, fModelReadyV6[has_dnit_data] = 1)),
    COUNTROWS(fModelReadyV6),
    0
) * 100

// 3.3 Classificação Categórica do Pavimento DNIT
Faixa_ICP_Descricao = 
SWITCH(
    TRUE(),
    SELECTEDVALUE(fModelReadyV6[has_dnit_data]) = 0, "Sem Cobertura Contínua (Sentinela)",
    SELECTEDVALUE(fModelReadyV6[icp]) <= 20, "1. Péssimo (0-20)",
    SELECTEDVALUE(fModelReadyV6[icp]) <= 40, "2. Ruim (21-40)",
    SELECTEDVALUE(fModelReadyV6[icp]) <= 60, "3. Regular (41-60)",
    SELECTEDVALUE(fModelReadyV6[icp]) <= 80, "4. Bom (61-80)",
    SELECTEDVALUE(fModelReadyV6[icp]) <= 100, "5. Ótimo (81-100)",
    "Indefinido"
)

// 3.4 Vítimas Severas Evitáveis ao Reformar Trechos Críticos (Péssimo -> Bom)
// Aplica a redução observada de 41,1% para 31,8% (-9,3 p.p. no índice de severidade)
Vitimas_Severas_Evitaveis_Simuladas = 
CALCULATE(
    [Total_Acidentes_Real] * (0.411 - 0.318),
    fModelReadyV6[has_dnit_data] = 1,
    fModelReadyV6[icp] <= 20
)

// 3.5 Estimativa de Óbitos Poupados com Recomposição de Pista (Péssimo -> Bom)
// Aplica a queda de 10,50% para 7,32% (-3,18 p.p. na taxa de mortalidade da PRF)
Obitos_Evitaveis_Simulados = 
CALCULATE(
    [Total_Acidentes_Real] * (0.1050 - 0.0732),
    fModelReadyV6[has_dnit_data] = 1,
    fModelReadyV6[icp] <= 20
)
```

---

## 5. Experiência do Usuário (UX, Filtros e Interatividade)

### 5.1 Barra Superior de Filtros Globais (Slicers Persistentes)
A navegação deve conter uma faixa superior horizontal fixa (estilo *Header Bar*) com 4 segmentadores de dados sincronizados entre todas as páginas:
1. **Horizonte Temporal:** Botão alternador de período:
   - `Treino Histórico (2020–2023)`
   - `Teste Fora do Tempo (2024–2025)`
   - `Ano Específico (Slider de 2019 a 2025)`
2. **Filtro Geográfico Hierárquico:** Slicer suspenso tipo árvore:
   - `Região` $\rightarrow$ `UF` $\rightarrow$ `Rodovia Federal (BR)`.
3. **Condição Viária (DNIT):** Botões em pastilha (*Pill buttons*):
   - `Todas as Vias` | `Apenas Trechos Monitorados` | `Pavimento Crítico (ICP <= 20)` | `Pavimento Bom (ICP >= 60)`.
4. **Perfil do Calendário:**
   - `Mês Típico Útil` | `Mês com Fim de Semana Prolongado` | `Feriadão Nacional`.

---

### 5.2 Regras de Formatação Condicional e Paleta Cromática Institucional

| Faixa de Risco / Indicador | Condição Técnica | Código Hexadecimal | Aplicação no Dashboard |
| :--- | :--- | :---: | :--- |
| **Risco Muito Baixo (Seguro)** | Predição $< 0{,}50$ acidentes/mês | `#10B981` (Verde Esmeralda) | Fundo de células de tabela e ícones de trecho calmo |
| **Risco Moderado** | Predição entre $0{,}50$ e $2{,}00$ ac/mês | `#3B82F6` (Azul Suave) | Barras de gráficos e linhas operacionais padrão |
| **Risco Elevado (Atenção)** | Predição entre $2{,}00$ e $5{,}00$ ac/mês | `#F59E0B` (Âmbar) | Alerta em cartões KPI e avisos de ronda periódica |
| **Ponto Crítico Máximo** | Predição $> 5{,}00$ acidentes/mês | `#DC2626` (Vermelho Vivo) | Destaque no Top 20 e viaturas de patrulhamento permanente |
| **Dados Reais Observados** | Contagem real histórica PRF | `#991B1B` (Bordeaux Escuro) | Linha de referência e pontos dispersos no gráfico temporal |
| **Simulação Pós-Reforma** | Cenário de intervenção ICP = 70 | `#059669` (Verde Floresta) | Barra de cenário alternativo no comparador do DNIT |

---

### 5.3 Recursos de Interatividade Avançada
1. **Drill-Through para Página Oculta de Trecho (10 km):**
   - Ao clicar com o botão direito sobre qualquer trecho em tabelas ou rankings, o usuário clica em *Detalhar Trecho*.
   - O Power BI filtra instantaneamente a página oculta exibindo:
     - Foto aérea e localização no mapa com coordenadas precisas.
     - Série temporal dos 84 meses específicos daquele segmento.
     - Decomposição das variáveis exógenas: comportamento histórico de chuva, feriados locais e série histórica de ICP do DNIT.
2. **Sincronização Cruzada de Seleção (Cross-Filtering):**
   - A seleção de um estado (ex.: Minas Gerais) no visual geográfico recalcula automaticamente os cartões executivos, o ranking de perigo e o gráfico temporal, sem necessidade de aplicar filtros adicionais.
3. **Botão de Limpeza Rápida de Filtros:**
   - Ícone de borracha posicionado no canto superior direito associado a um *Bookmark* que reseta todas as variáveis de segmentação para o estado padrão do teste cego de 2024–2025.
