# Análise e Modelagem Preditiva de Acidentes em Rodovias Federais (PRF + DNIT)

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RandomForest-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-green)

Este repositório contém todo o pipeline de engenharia de dados, as 5 arquiteturas de modelagem de Machine Learning (V1 a V5) e a explicabilidade (XAI) do Trabalho de Conclusão de Curso (TCC) em Ciência da Computação (UNIP, 2026). O objetivo central do projeto é **provar matematicamente o impacto isolado da infraestrutura rodoviária** (medida pelo DNIT) na letalidade e frequência de colisões mapeadas pela Polícia Rodoviária Federal (PRF), isolando fatores comportamentais e climáticos.

> ⚠️ **Nota sobre os Dados:** Devido aos limites de armazenamento do GitHub, os arquivos massivos (`.csv` originais de > 200MB) e os modelos serializados `.pkl` **não estão versionados** neste repositório. O foco aqui é a demonstração do Código-Fonte, Metodologia, Scripts de Treinamento e Gráficos de Resultados.

---

## 🏗️ Arquitetura e Estrutura dos Módulos

O código-fonte está estritamente dividido em 10 módulos operacionais mapeando a estrutura da Monografia:

- **[`01_Extracao/`](file:///Users/marciohugo/Documents/tcc-machine-learning-prf/01_Extracao):** Dicionários de dados e rotinas de extração estrutural do DNIT e PRF.
- **[`02_Transformacao/`](file:///Users/marciohugo/Documents/tcc-machine-learning-prf/02_Transformacao):** Scripts de ETL (limpeza pesada, remoção de nulos vitais, otimização de memória e tratamento de IQR).
- **[`03_Engenharia_Atributos/`](file:///Users/marciohugo/Documents/tcc-machine-learning-prf/03_Engenharia_Atributos):** Criação de variáveis de sazonalidade (estação do ano, finais de semana, feriados nacionais) e traçado geométrico.
- **[`04_Integracao_DNIT/`](file:///Users/marciohugo/Documents/tcc-machine-learning-prf/04_Integracao_DNIT):** Algoritmo customizado de **Interval Join ASOF Espacial-Temporal**, cruzando acidentes da PRF com os segmentos avaliados pelo DNIT (BR e KM).
- **[`05_Pre_Modelagem/`](file:///Users/marciohugo/Documents/tcc-machine-learning-prf/05_Pre_Modelagem):** Agrupamento da variável alvo em **Buckets de 10km/Mês**, análise de assimetria (*skewness*) e documentação do pipeline.
- **[`06_Modelagem/`](file:///Users/marciohugo/Documents/tcc-machine-learning-prf/06_Modelagem):** Implementação completa das **5 Arquiteturas de IA (V1 à V5)**, script comparador unificado e relatório oficial (`model_comparison.txt`).
- **[`07_Variaveis_Importantes/`](file:///Users/marciohugo/Documents/tcc-machine-learning-prf/07_Variaveis_Importantes):** Inspeção da importância das variáveis via MDI (*Mean Decrease Impurity*) e *Permutation Importance*.
- **[`08_Fatores_Climaticos/`](file:///Users/marciohugo/Documents/tcc-machine-learning-prf/08_Fatores_Climaticos):** Testes de cenários auditados (pico de verão, finais de semana, chuva em serras e casos reais críticos).
- **[`09_Analise_Infraestrutura/`](file:///Users/marciohugo/Documents/tcc-machine-learning-prf/09_Analise_Infraestrutura):** Simulação *Ceteris Paribus* com intercepto PDP, curvas de sensibilidade do Índice de Condição da Superfície (ICP) e SHAP *TreeExplainer*.
- **[`10_Documentacao_Geral/`](file:///Users/marciohugo/Documents/tcc-machine-learning-prf/10_Documentacao_Geral):** Especificações CRISP-DM, dicionário de bibliotecas, logs e o **Relatório Técnico de Auditoria Metodológica**.

---

## 🤖 As 5 Arquiteturas de Modelagem (V1 à V5)

Conforme documentado no Capítulo 4 do TCC, foram desenvolvidas e comparadas cinco configurações progressivas do `RandomForestRegressor`:

| Arquitetura | Script | Divisão | Hiperparâmetros Chave | R² Teste | MAE Teste |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **V1 (Baseline)** | `model_training_v1.py` | Random 80/20 | 100 árvores, `max_features=1.0`, sem poda | 0,9892* | 0,2276 |
| **V2 (Otimizado)** | `model_training_v2.py` | Random 70/30 | 200 árvores, `max_features='sqrt'`, OOB=True | 0,9770* | 0,3450 |
| **V3 (Poisson + Poda)** | `model_training_v3.py` | Random 70/30 | 200 árvores, `criterion='poisson'`, depth=25, leaf=5 | 0,9678* | 0,4011 |
| **V4 (Log-Transform)** | `model_training_v4.py` | Random 70/30 | 300 árvores, `np.log1p(y)`, max_samples=0.8, depth=30 | 0,9556* | 0,4107 |
| **V5 (Temporal)** | `model_training_v5.py` | Temporal (2019–23 / 2024–25) | 200 árvores, max_samples=0.9, min_leaf=2 | **0,9726*** | **0,4181** |
| **V6-Ablation (Infra)**| `model_training_v6.py` | Out-of-Time (2024–25) | 200 árvores, Poisson, sem lags | **0,0381** | **1,4550** |
| **V6-Full (Oficial)** | `model_training_v6.py` | **Out-of-Time (2024–25)** | **200 árvores, Poisson, Lags + DNIT + Grade Real** | **0,8678** | **0,6137** |

*\*Nota: Os modelos V1 a V5 apresentavam vazamento contábil de soma de categorias pós-evento e viés de sobrevivência (ausência de zeros reais).*

---

## 🏆 Arquitetura V6 (Definitiva e Auditada — Zero Leakage)

A **Arquitetura V6** foi construída a partir do zero para eliminar todas as fragilidades metodológicas identificadas na auditoria:
1. **Grade Cartesiana Completa (Zeros Reais):** Incorporação de todos os 5.655 trechos de 10 km monitorados ao longo de 84 meses (2019 a 2025), totalizando **475.020 observações** com **70,61% de zeros reais** (meses sem sinistro).
2. **Zero Target Leakage:** Expurgadas todas as contagens pós-evento agregadas (`is_*`).
3. **Inércia Espacial e Lags Temporais:** Criação de defasagens históricas puras ($t-1, t-2, t-3, t-12$) para controlar a densidade de tráfego intrínseca do trecho.
4. **Sentinela DNIT Estrito:** Ausência de medições DNIT tratada com sentinela `-1.0` e flag binária `has_dnit_data`, sem backfill sintético artificial.
5. **Divisão Fora do Tempo (Out-of-Time):** Treino entre 2020 e 2023 ($N=271.440$) e teste estritamente cego em 2024 e 2025 ($N=135.720$).

### Performance do Modelo Oficial V6:
- **$R^2$ Teste Fora do Tempo:** **$0{,}8678$**
- **Poisson Deviance $D^2$:** **$0{,}7964$**
- **MAE Global:** **$0{,}6137$ acidentes/mês**
- **MAE em Trechos Seguros ($Y=0$):** **$0{,}2889$ acidentes/mês**
- **Ablação Sem Lags (Infraestrutura Pura):** $R^2 = 0{,}0381$ (comprovando que o fluxo de tráfego determina a escala base, enquanto o DNIT modula a severidade e o risco marginal).

### Execução dos Módulos V6:
```bash
# Treinamento e Estudo de Ablação V6:
python3 06_Modelagem/model_training_v6.py

# Diagnósticos e Gráficos de Resíduos V6:
python3 06_Modelagem/model_report_v6.py

# Importância de Features (XAI):
python3 07_Variaveis_Importantes/feature_importance_v6.py

# Simulação de Cenários Operacionais:
python3 08_Fatores_Climaticos/cenarios_v6.py

# Análise de Sensibilidade do Pavimento DNIT (ICP):
python3 09_Analise_Infraestrutura/analise_impacto_icp_v6.py

# Gerar todos os 20 gráficos consolidados da V6 (pasta graficos_v6/):
python3 gerar_todos_graficos_v6.py
```

> 📄 **Documentação Detalhada V6:** Consulte o [Relatório Técnico do Dataset V6](10_Documentacao_Geral/relatorio_dataset_v6.md), o [Relatório Oficial do Modelo V6](10_Documentacao_Geral/relatorio_modelo_v6.md) e a pasta [`documento_tcc/`](documento_tcc/) com a monografia escrita.
> 
> 📊 **Galeria Consolidada de Gráficos:** A pasta [`graficos_v6/`](graficos_v6/) reúne os 20 gráficos correspondentes à evolução histórica do projeto, todos padronizados com dados e predições legítimas do Modelo V6.

---

## 📈 Principais Resultados Empíricos

- **Impacto do Asfalto (ICP) e Paradoxo de Risco (Efeito Peltzman):** Em vias com pavimento em estado péssimo ($ICP \le 20$), os motoristas reduzem a velocidade por precaução, mantendo a frequência de pequenas colisões baixa. Contudo, **a taxa de letalidade real da PRF em trechos péssimos é de 10,50% de óbitos por acidente, contra 7,32% em trechos bons (+43,4% mais letal que o Bom)**.
- **Índice de Vítimas Severas:** A recuperação asfáltica (Péssimo $\rightarrow$ Bom) reduz acidentes com vítimas graves e mortos de 41,1% para 31,8%, comprovando que o investimento do DNIT salva vidas e minimiza traumas severos.
- **Fatores de Calendário e Feriados:** Fins de semana prolongados e feriados nacionais elevam a expectativa de acidentes em até +22,3% nos principais corredores turísticos federais.
- **Inércia Espacial:** A inércia histórica dos últimos meses responde por cerca de 80% do poder explicativo espacial, capturando a exposição volumétrica de veículos onde não há dados de tráfego contínuo.
