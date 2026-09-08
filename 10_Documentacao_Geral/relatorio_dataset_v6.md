# Relatório Técnico de Engenharia do Dataset V6 (PRF + DNIT)
**Projeto:** Modelagem Preditiva de Acidentes Rodoviários Federais  
**Data da Execução:** Setembro de 2026  
**Status:** Concluído com Sucesso e Auditado  

---

## 1. Resumo Executivo das Mudanças (V5 vs. V6)

A construção do **Dataset V6** marca uma transição metodológica fundamental em relação à Arquitetura V5:

| Dimensão | Arquitetura V5 | Arquitetura V6 (Novo Dataset) |
| :--- | :--- | :--- |
| **Universo Amostral** | Apenas onde houve sinistros ($Y \ge 1$) | **Grade Cartesiana Completa** ($Y = 0$ e $Y \ge 1$) |
| **Tamanho da Matriz** | $139.590$ observações | **$475.020$ observações** ($5.655$ trechos $\times$ $84$ meses) |
| **Proporção de Zeros** | $0\%$ (Viés de Sobrevivência) | **$70{,}61\%$ de Zeros Reais** ($335.430$ observações seguras) |
| **Prevenção de Leakage** | Vazamento contábil ($\sum \text{is\_*} \approx Y$) | **Zero Vazamento** (Variáveis contextuais pós-evento expurgadas) |
| **Base DNIT 2023** | $38.529$ linhas perdidas (sem ano) | **$38.526$ linhas recuperadas** com rótulo auditado |
| **Tratamento de Ausências** | Inconsistência entre código e documentação | **Sentinela `-1.0` estrito** + flag binária `has_dnit_data` |
| **Componente Temporal** | Sem defasagem histórica (*lags*) | **Lags puros ($t-1, t-2, t-3, t-12$)** por trecho |

---

## 2. Estrutura e Dicionário do Dataset `df_model_ready_v6.csv`

O dataset consolidado gerado possui **$475.020$ linhas** e **$22$ colunas** ($32{,}6\text{ MB}$), distribuído uniformemente em $67.860$ observações por ano (2019 a 2025):

### A. Chaves Identificadoras e Geográficas
1. `ano`: Ano civil (2019 a 2025).
2. `mes`: Mês de referência (1 a 12).
3. `br`: Identificador numérico da rodovia federal (ex.: 116, 101, 381).
4. `trecho_10km`: Ponto quilométrico inicial do segmento de 10 km (ex.: 0, 10, 20...).
5. `uf`: Unidade Federativa predominante do segmento.

### B. Atributos Exógenos de Calendário (Determinação Pura Pré-Evento)
6. `dias_no_mes`: Quantidade de dias no mês (28 a 31).
7. `dias_fim_semana`: Contagem de sábados e domingos no calendário civil.
8. `dias_feriado`: Contagem oficial de feriados nacionais no mês.
9. `estacao_ano`: Estação climática (Verão, Outono, Inverno, Primavera).

### C. Atributos de Infraestrutura Física (DNIT)
10. `icp`: Média do Índice de Condição do Pavimento no trecho de 10 km ($-1.0$ se não monitorado).
11. `icc`: Média do Índice de Conservação no trecho ($-1.0$ se não monitorado).
12. `icm`: Média do Índice de Manutenção no trecho ($-1.0$ se não monitorado).
13. `has_dnit_data`: Flag binária indicadora ($1$ se monitorado, $0$ se ausência/sentinela).

### D. Variáveis Autoregressivas Defasadas (*Lags Históricos do Trecho*)
14. `acidentes_lag_1m`: Sinistros no mesmo trecho no mês anterior ($t-1$).
15. `acidentes_lag_2m`: Sinistros no mesmo trecho há 2 meses ($t-2$).
16. `acidentes_media_movel_3m`: Média móvel dos últimos 3 meses ($t-1, t-2, t-3$).
17. `acidentes_lag_12m`: Sinistros no mesmo trecho no mesmo mês do ano anterior ($t-12$, sazonalidade).

### E. Variáveis de Diagnóstico e Alvo
18. `total_mortos`: Óbitos registrados no trecho/mês (para estudos de severidade).
19. `total_feridos_graves`: Vítimas graves registradas.
20. `total_veiculos`: Quantidade de veículos envolvidos.
21. `is_zero_real`: Flag indicadora de trecho com zero acidentes ($1$ para seguro, $0$ para sinistro).
22. **`quantidade_acidentes` (VARIÁVEL ALVO $Y$):** Contagem real de acidentes no trecho/mês.

---

## 3. Validação Estatística de Integridade

1. **Combate ao Viés de Sobrevivência:**
   * $70{,}61\%$ das observações ($335.430$ linhas) possuem $Y = 0$.
   * A média global é de $0{,}997$ acidentes por trecho/mês, com desvio padrão compatível com distribuições de Poisson/Binomial Negativa.
2. **Cobertura do DNIT:**
   * $137.352$ observações ($28{,}91\%$) possuem medição contemporânea do DNIT.
   * As demais $337.668$ observações preservam `-1.0` de forma íntegra e auditável.
3. **Ausência de Vazamento Contábil:**
   * Nenhuma variável preditiva apresenta a dependência linear espúria de soma que inflava o modelo V5. As variáveis de lag refletem a inércia espacial real dos trechos (trechos urbanos densos mantêm maior volume histórico que trechos remotos).

---

## 4. Recomendações para o Treinamento do Modelo V6
- **Warm-up Temporal:** Utilizar o ano de 2019 exclusivamente para inicialização dos *lags* temporais.
- **Divisão de Treino:** Anos de **2020 a 2023** ($271.440$ observações).
- **Divisão de Teste Fora do Tempo (*Out-of-Time*):** Anos de **2024 e 2025** ($135.720$ observações).
- **Algoritmo Recomendado:** `RandomForestRegressor` com `criterion='poisson'` ou `LightGBM` com função de perda `Tweedie`.
