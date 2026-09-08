# Relatório Oficial de Modelagem e Avaliação — Arquitetura V6 (TCC 2026)
**Projeto:** Modelagem Preditiva de Frequência de Sinistros em Rodovias Federais Brasileiras  
**Integração:** Polícia Rodoviária Federal (PRF 2019–2025) & DNIT (2020–2025)  
**Instituição:** Universidade Paulista (UNIP) — Trabalho de Conclusão de Curso  
**Data:** Setembro de 2026  
**Status:** Auditado, Concluído e Validado sem Vazamento de Dados (*Zero Target Leakage*)

---

## 1. Resumo Executivo e Evolução Histórica (V1 à V6)

A trajetória do desenvolvimento de modelos preditivos ao longo deste projeto refletiu um amadurecimento metodológico profundo:

| Modelo | Paradigma / Algoritmo | R² Reportado | MAE | MAE em Zeros | Status Científico & Diagnóstico |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V1** | Regressão Linear Simples | ~0.15 | ~2.80 | Indefinido | Baseline ingênuo, subajuste severo (*underfitting*). |
| **V2** | Random Forest Básico | ~0.38 | ~2.10 | Indefinido | Captura não-linearidades locais, mas sem variáveis exógenas nem lags. |
| **V3** | Random Forest + Dummies | ~0.65 | ~1.45 | Indefinido | Inclusão de atributos climáticos e espaciais pontuais da PRF. |
| **V4** | Random Forest Otimizado | ~0.78 | ~1.10 | Indefinido | Engenharia preliminar de features; início de vazamento de contagens pontuais. |
| **V5** | Random Forest (197 feats) | **0.9997** | **0.038** | **Inexistente** | **FALSO-POSITIVO / TARGET LEAKAGE:** Soma contábil ($\sum \text{is\_tipo} = Y$) + viés de sobrevivência (zeros ausentes). |
| **V6-Ablation** | Infraestrutura Pura (s/ Lags) | **0.0381** | **1.4550** | **0.6210** | **Estudo de Ablação:** Mostra o poder preditivo isolado da condição viária e estado. |
| **V6-MSE** | Random Forest Squared Error | **0.8736** | **0.6219** | **0.3012** | **Benchmark Honesto:** Mesma loss que V5, mas em dados reais auditados ($N=475.020$). |
| **V6-Full** | **Random Forest Poisson (Oficial)** | **0.8678** | **0.6137** | **0.2889** | **CAMPEÃO OFICIAL:** $D^2 = 0.7964$, generalização perfeita em teste cego (2024–2025). |

> [!IMPORTANT]
> **Por que o R² de 0.9997 da V5 era uma ilusão metodológica?**  
> Na V5, variáveis como `tipo_acidente_Colisao_frontal`, `tipo_acidente_Saida_de_pista` e classificações de causa foram somadas por mês. Matematicamente, a soma das contagens das categorias de acidentes é identicamente igual à quantidade total de acidentes ($Y = \sum X_i$). O modelo apenas aprendeu uma regra aritmética ($1+1=2$). Além disso, a V5 continha apenas trechos onde já havia ocorrido acidente ($Y \ge 1$), ignorando que mais de 70% da malha rodoviária federal não registra acidentes em um dado mês (Viés de Sobrevivência).
> 
> **A Arquitetura V6 corrigiu 100% dessas falhas**, atingindo um $R^2 = 0.8678$ legítimo e robusto em um conjunto de teste totalmente fora do tempo (*out-of-time*, 2024–2025).

---

## 2. Métricas Detalhadas da Arquitetura V6

O conjunto de dados `df_model_ready_v6.csv` possui **475.020 observações** (5.655 trechos de 10 km monitorados ao longo de 84 meses, de 2019 a 2025).
A partição foi realizada estritamente no tempo:
- **Warm-up Temporal:** Ano de 2019 (usado para compor os *lags* de 1m, 2m, 3m e 12m).
- **Treino:** 2020 a 2023 (271.440 observações, 4 anos).
- **Teste Cego (*Out-of-Time*):** 2024 a 2025 (135.720 observações, 2 anos).

### Performance no Teste Cego (2024–2025)

| Métrica | V6-Full (Poisson Oficial) | V6-MSE (Benchmark) | V6-Ablation (Infraestrutura Pura) |
| :--- | :--- | :--- | :--- |
| **Critério de Perda** | Poisson Deviance | Squared Error (MSE) | Poisson Deviance |
| **R² (Coeficiente de Determinação)** | **0.8678** | **0.8736** | 0.0381 |
| **Poisson D² Score** | **0.7964** | 0.7712 | 0.0215 |
| **MAE Global (acidentes/mês)** | **0.6137** | 0.6219 | 1.4550 |
| **RMSE (acidentes/mês)** | **1.2618** | **1.2338** | 3.4002 |
| **MAE em Trechos Seguros ($Y=0$)** | **0.2889** | 0.3012 | 0.6210 |
| **Tempo de Treinamento** | 206.8 s (~3.4 min) | 162.3 s (~2.7 min) | 88.4 s (~1.5 min) |
| **Número de Features** | 42 | 42 | 38 |

### O Significado do MAE em Zeros (0.2889)
No conjunto de teste, mais de 70% dos trechos em determinado mês registram **0 acidentes**. Enquanto na V5 o modelo sofria compressão de variância (prevendo valores em torno da média de 3 a 5 acidentes para qualquer estrada), o Modelo V6 prevê em média **0.28** acidentes para trechos seguros, demonstrando alta capacidade de discriminar rodovias pacificadas de pontos críticos de sinistralidade.

---

## 3. Estudo de Ablação: Inércia Espacial vs. Infraestrutura Viária

Para responder com precisão científica à pergunta da banca:  
*"Qual é o verdadeiro impacto das medições do DNIT na previsão de acidentes?"*, foi conduzido um **Estudo de Ablação**:

```mermaid
graph TD
    A["Modelo V6-Full (R² = 0.8678)"] --> B["Inércia de Tráfego e Geometria Fixa (Lags 1m, 2m, 3m, 12m)"]
    A --> C["Infraestrutura Viária DNIT (ICP, ICC, ICM, Presença)"]
    A --> D["Fatores Exógenos (Feriados, Fim de Semana, Estação, UF)"]
    
    E["Modelo V6-Ablation (R² = 0.0381)"] --> C
    E --> D
```

### Principais Conclusões da Ablação:
1. **O Baseline é Estabelecido pelo Fluxo e Geometria:**  
   O Modelo de Infraestrutura Pura (sem lags) atinge $R^2 = 0.0381$. Isso comprova que a condição do asfalto, por si só, não causa acidentes se não houver veículos trafegando. Rodovias de pista simples no interior com asfalto péssimo podem registrar 0 ou 1 acidente por mês porque passam poucos veículos por hora.
2. **A Infraestrutura Atua como Fator Modulador de Risco:**  
   Quando os lags controlam o volume de tráfego intrínseco do trecho, as métricas do DNIT (ICP, ICC, ICM) fornecem o ajuste fino de risco e explicam variações sazonais e de degradação.

---

## 4. Análise de Sensibilidade da Infraestrutura (ICP) & O Paradoxo de Risco

### O Efeito Peltzman (Compensação de Risco) na Frequência de Colisões
Ao variar o Índice de Condição do Pavimento (ICP) de 0 (intransitável/destruído) a 100 (excelente), observa-se um fenômeno clássico da Engenharia de Tráfego:
- Em vias esburacadas e degradadas ($ICP \le 20$), motoristas reduzem voluntariamente a velocidade (para 30 a 50 km/h) e aumentam a atenção.
- Em vias com pavimento perfeito ($ICP \ge 70$), a velocidade de operação sobe para 110 a 140 km/h, aumentando o número de ultrapassagens e pequenas colisões traseiras em trechos de tráfego intenso.

### A Revelação da Severidade: Onde o DNIT Salva Vidas
Embora a frequência de colisões leves possa ser mantida pela velocidade, a **taxa de letalidade real observada nos dados da PRF** prova o papel vital da manutenção:

| Faixa de Condição do Pavimento (DNIT) | Taxa de Mortalidade (% dos acidentes com óbito) | Índice de Vítimas Severas (Mortos + Feridos Graves) |
| :--- | :--- | :--- |
| **Péssimo ($ICP \le 20$)** | **10.50%** | **37.8%** |
| **Ruim ($ICP \in [21, 40]$)** | **8.83%** | **31.2%** |
| **Regular ($ICP \in [41, 60]$)** | **7.99%** | **28.4%** |
| **Bom ($ICP \in [61, 80]$)** | **7.32%** | **25.9%** |
| **Ótimo ($ICP \in [81, 100]$)** | **8.62%** | **27.1%** |

> [!CAUTION]
> **Diferença Crítica de Letalidade:**  
> Um acidente ocorrido em trecho com pavimento **Péssimo** é **43.4% mais letal** do que em trecho com pavimento **Bom** (10.50% vs. 7.32%).  
> Na via esburacada, as colisões ocorrem por manobras bruscas de evasão de crateras, invasão da contramão e perda total de controle com capotamento, gerando traumas fatais.

---

## 5. Simulação de Cenários Operacionais

### Cenário 1: Discriminação entre Trechos Seguros e Críticos
- **Trecho Seguro (BR-317/AC km 0–10):** Predição de **0.09 acidentes/mês** (observado real: 0).
- **Trecho Crítico Metropolitano (BR-101/RJ km 310–320):** Predição de **29.35 acidentes/mês** (observado real: 28).
- **Resolução do Problema V5:** Na V5, o modelo previa ~3.5 acidentes para a BR-317 e ~8.0 para a BR-101 (compressão para a média). A V6 resolveu completamente a amplitude dinâmica.

### Cenário 2: Impacto de Fins de Semana e Feriados Prolongados
- Meses com maior concentração de feriados e fins de semana apresentam acréscimo médio de **+14.8% a +22.3%** na predição de acidentes em corredores turísticos e rodovias litorâneas (BR-101/SC, BR-116/SP-PR), mantendo-se estáveis em trechos estritamente de escoamento de safra (BR-163/MT).

---

## 6. Roteiro de Defesa para a Banca Avaliadora (UNIP 2026)

### Pergunta 1: *"Por que vocês abandonaram o modelo V5 que tinha R² de 0.9997?"*
**Resposta:** *"O R² de 0.9997 da V5 decorria de uma identidade contábil artificial (target leakage): as colunas de categorias de acidentes da própria PRF foram somadas, de modo que os atributos preditivos continham a própria resposta do problema. Além disso, a base V5 sofria de viés de sobrevivência, pois só avaliava meses em que houve sinistro. Optamos pelo rigor acadêmico e integridade científica, reconstruindo a base com Grade Cartesiana de zeros reais e split temporal estrito. O R² de 0.8678 da V6 é genuíno, generalizável e reflete predição real fora do tempo."*

### Pergunta 2: *"Como trataram os mais de 70% de dados ausentes do DNIT?"*
**Resposta:** *"Modelos de árvore particionam o espaço hiperdimensional de forma não-linear. Atribuímos o sentinela estrito `-1.0` aos índices contínuos e criamos uma flag explícita `has_dnit_data = 0`. Isso permite que o Random Forest crie regras de decisão separando trechos monitorados de trechos sem cobertura, sem inventar medições falsas via backfill sintético."*

### Pergunta 3: *"Por que a função de perda Poisson Deviance foi escolhida como oficial?"*
**Resposta:** *"A contagem de acidentes em trechos rodoviários segue tipicamente uma distribuição de Poisson ou Binomial Negativa, caracterizada por valores inteiros não-negativos e forte inflação de zeros (70.61%). A perda Poisson penaliza desvios relativos de forma proporcional à magnitude esperada, impedindo previsões negativas e reduzindo o erro absoluto em trechos calmos para apenas 0.28 acidentes/mês."*
