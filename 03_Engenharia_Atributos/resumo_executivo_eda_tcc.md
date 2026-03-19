# Resumo Executivo: Análise Exploratória de Dados (EDA) - Acidentes PRF (2019-2025)

Este documento sintetiza as principais descobertas, insights analíticos e decisões arquiteturais tomadas durante a fase de **Análise Exploratória de Dados (EDA)** do projeto de TCC. As referências foram extraídas dos logs operacionais (`execution_log.txt`) e dos relatórios de auditoria gerados ao longo do pipeline.

---

## 1. O Problema da "Cauda Longa" (Zero-Inflated e Assimetria)
Um dos maiores aprendizados da etapa analítica que guiou a engenharia do projeto foi a descoberta da alta assimetria e do comportamento inflacionado de zeros (Zero-Inflated) na rodovia.

*   **Evidência Estatística (Skewness):** A análise estatística descritiva comprovou alta dispersão. Variáveis alvo como `feridos` apresentaram um *Skewness* extremo de `10.36`, e mortos `9.68`. A enorme maioria dos acidentes (e dos trechos da via) possui histórico nulo ou de levíssima gravidade. 
*   **Insight de Negócio:** "O perigo não se distribui uniformemente no mapa." A vasta extensão das rodovias brasileiras experimenta 0 acidentes, enquanto curtíssimos trechos (pontos cegos, declives) acumulam dezenas de mortes por mês.
*   **Decisão Tomada:** Descartou-se a predição linear ou de quilômetro exato. Adotou-se algoritmos robustos a *outliers* (Random Forest) e operou-se o **Bucketing Espacial** (janelas de 10km) para concentrar e viabilizar o diagnóstico.

## 2. Diagnóstico de Correlação e "Feature Importance"
Analisando a correlação de Pearson pré-modelagem sobre a variável `feridos`:

*   **Alta Importância Linear:** `feridos_leves` (0.858) e `pessoas` envolvidas no evento (0.539).
*   **A Causa vs Efeito (O Perigo do Data Leakage):** Durante a EDA inicial em dados brutos para gerar os gráficos analíticos, descobriu-se que variáveis da PRF como `causa_acidente`, `tipo_acidente` e `ilesos` só existem *depois* que a tragédia acontece.
*   **Decisão Tomada:** Remover rigidamente essas colunas (Redução de Dimensionalidade de 12 features) para evitar vazamento de dados (*Data Leakage*), garantindo que o modelo apenas utilizasse variáveis *a priori* (clima, pista, sazonalidade).

## 3. O Peso da Sazonalidade (Insights Temporais)
A mineração quantitativa dos gráficos analíticos validou as hipóteses do calendário de tráfego brasileiro:
*   Feriados e fins de semana apresentam um salto quantitativo e letal. A *Feature Engineering* de variáveis booleanas (`is_final_semana`, `is_feriado`) se provou estatisticamente relevante nas primeiras quebras das Árvores de Decisão do ML.
*   **A "Tempestade Perfeita" (Sazonalidade Mensal):** O gráfico temporais expôs Dezembro como grande outlier (Pico do Verão + Festas). Em trechos críticos (BR-101/BA KM 200), a média anual geral passa de 4/5 para mais de 9 acidentes ao mês (salto maior do que 70% sob chuva).

## 4. Reflexos de Engenharia (A Infraestrutura do DNIT)
O *Join ASOF* com o DNIT trouxe a materialização da infraestrutura, onde as variáveis estruturais ganharam força analítica:
*   **O "Perigo Silencioso":** O EDA cruzado expõs que traçados de `Declive` e `Curva` multiplicam o risco natural em mais de +50% numa rodovia de serra (BR-116/SP Registro) do que `Reta` em dias idênticos (Dias úteis sob sol).
*   Ficou comprovado que a *sensação* de segurança de pistas retas e duplas gera risco comportamental acelerado: o aumento marginal de acidentes em uma pista dupla sob chuva é sensivelmente maior que sua equivalente em pista simples, possivelmente pelo excesso de velocidade por desatenção do condutor ao perder atrito.

## 5. Geografia da Sinistralidade (Top BRS Letais)
Os gráficos analíticos de volume por UF (`g7_volume_por_uf.png`) e rodovias (`eda5_top10_brs_letais.png`) mapearam onde está a crise gerencial da PRF:
*   **BR-116:** A maior artéria nacional cortante, que engloba desafios serranos (Régis Bittencourt SP/PR) que a validam como o laborioso foco de obras do DNIT.
*   **BR-101:** O corredor litorâneo (Nordeste - SC), epicentro dos choques anuais em finais de ano pelo migração agressiva de veranistas sem familiaridade com a estrada sob chuvas instantâneas de verão.
*   **BR-381:** "A Rodovia da Morte" (MG). Possui trechos críticos (Ex: KM 340 João Monlevade), onde os cenários provaram que o choque multiplicador (Feriado + Chuva + Curva + Simples) instiga um nível alerta máximo operacional (acima de 2.5 eventos num trecho milimetricamente curto por mês).

---
### Sintese
A Fase de EDA provou que o sistema rodoviário brasileiro demanda um estudo focado na interseção (*interação não-linear*) do Clima vs. Pavimento vs. Data. O Machine Learning foi a saída escolhida após o analítico visual e matemático do EDA demonstrar formalmente a inviabilidade de tratativas de Regressão Linear sobre esses dados ruidosos.
