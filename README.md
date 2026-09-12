# Análise e Modelagem Preditiva de Acidentes em Rodovias Federais (PRF + DNIT)

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RandomForest%20Poisson-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-green)
![Dataset](https://img.shields.io/badge/Dataset-475k%20Linhas%20(Zeros%20Reais)-purple)
![Auditoria](https://img.shields.io/badge/Status-Auditado%20V6%20(Zero%20Leakage)-success)

Este repositório contém o pipeline completo de engenharia de dados, as arquiteturas de modelagem preditiva (V1 a V6 Oficial e o experimento V7), a monografia acadêmica e os módulos de explicabilidade (XAI) do Trabalho de Conclusão de Curso em Ciência da Computação (UNIP, 2026). O objetivo central reside em mensurar o impacto da infraestrutura viária (avaliada pelo DNIT) na frequência e letalidade dos sinistros registrados pela Polícia Rodoviária Federal (PRF), isolando variáveis comportamentais, meteorológicas e a inércia espacial de tráfego.

> **Nota sobre os Dados:** Em conformidade com os limites de armazenamento do GitHub, bases volumétricas originais (`.csv` > 200 MB) e binários serializados `.pkl` não são versionados diretamente neste repositório. O repositório prioriza o código-fonte reprodutível, a metodologia, a monografia modular e o catálogo consolidado de gráficos e diagnósticos.

---

## Destaque Científico: O Efeito Peltzman e a Severidade do DNIT

Uma das descobertas centrais da pesquisa é a demonstração empírica do paradoxo de compensação de risco e o efeito mensurável do pavimento na **severidade e letalidade das vítimas**:

![O Efeito Peltzman e a Severidade: Onde o DNIT Realmente Salva Vidas](graficos_v6/efeito_peltzman_e_severidade_dnit_v6.png)

1. **O Paradoxo da Frequência (Efeito Peltzman):** Em rodovias com pavimento excelente ($ICP \ge 70$), a velocidade operacional média atinge patamares elevados (110 a 140 km/h) e há maior atratividade de fluxo, resultando em maior densidade de colisões leves e moderadas.
2. **A Taxa Real de Letalidade da PRF:** O pavimento **Péssimo ($ICP \le 20$) é +43,4% mais letal que o Bom ($ICP \in [61, 80]$)** (10,50% de óbitos por acidente contra 7,32%). Desvios bruscos de buracos, deformações plásticas e estouro de pneus geram perda de controle direcional e colisões frontais com alta energia dissipada.
3. **Índice de Vítimas Severas:** A recuperação estrutural da pista (Péssimo para Bom) reduz os acidentes com vítimas graves e mortos de **41,1% para 31,8%**, comprovando onde o investimento em recapeamento viário efetivamente preserva vidas e minimiza sequelas permanentes.

---

## Evolução das Arquiteturas de Modelagem (V1 a V7)

O projeto percorreu um amadurecimento metodológico completo, culminando na **Arquitetura V6 Oficial** (modelo da monografia) e no **Experimento V7** (estudo contrafactual):

| Arquitetura | Script | Partição Amostral | Algoritmo / Função de Perda | R² Teste | MAE Teste | Diagnóstico Metodológico |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **V1 (Baseline)** | `model_training_v1.py` | Aleatória 80/20 | 100 árvores, `max_features=1.0`, MSE | 0,9892* | 0,2276 | Baseline preliminar ingênuo. |
| **V2 (Otimizado)** | `model_training_v2.py` | Aleatória 70/30 | 200 árvores, `max_features='sqrt'`, OOB | 0,9770* | 0,3450 | Captura não-linearidades locais. |
| **V3 (Poisson)** | `model_training_v3.py` | Aleatória 70/30 | 200 árvores, `criterion='poisson'`, depth=25 | 0,9678* | 0,4011 | Primeira adaptação para contagem. |
| **V4 (Log-Transform)** | `model_training_v4.py` | Aleatória 70/30 | 300 árvores, `np.log1p(y)`, max_samples=0.8 | 0,9556* | 0,4107 | Regularização da cauda longa. |
| **V5 (Temporal)** | `model_training_v5.py` | Temporal (2019–23 / 2024–25) | 200 árvores, max_samples=0.9, min_leaf=2 | **0,9726*** | **0,4181** | **Target Leakage:** Soma de categorias pós-evento e viés de sobrevivência (zeros ausentes). |
| **V6-Ablation (Infra)**| `model_training_v6.py` | Out-of-Time (2024–2025) | 200 árvores, Poisson, **sem lags temporais** | **0,0381** | **1,4550** | **Estudo de Ablação:** Mostra que a infraestrutura pura modula o risco, mas não cria fluxo. |
| **V6-MSE (Benchmark)**| `model_training_v6.py` | Out-of-Time (2024–2025) | 200 árvores, Squared Error, Lags + Grade Real | **0,8736** | **0,6219** | **Benchmark Direto:** Mesma loss da V5 sobre a matriz de 475k observações auditadas. |
| **V6-Full (Oficial)** | `model_training_v6.py` | **Out-of-Time (2024–2025)** | **200 árvores, Poisson Deviance, Lags + DNIT** | **0,8678** | **0,6137** | **MODELO OFICIAL DO TCC:** $D^2 = 0{,}7964$, generalização fora do tempo sem vazamento. |
| **V7 (Experimental)** | `experimento_v7_dnit_imputado.py` | **Out-of-Time (2024–2025)** | **200 árvores, Poisson, DNIT 100% Imputado** | **0,8701** | **0,6208** | **Estudo Contrafactual:** Simulação de cobertura universal do DNIT (MDI saltou para 8,09%). |

*\*Nota: Os modelos V1 a V5 continham vazamento contábil por agregação de atributos gerados após a ocorrência do sinistro e operavam condicionados à ocorrência ($Y \ge 1$), excluindo trechos seguros.*

---

## Fundamentos da Arquitetura V6 Oficial (Zero Target Leakage)

A **Arquitetura V6 Oficial** foi implementada a partir do zero para sanar as fragilidades identificadas durante a auditoria técnica:
1. **Grade Cartesiana Completa (Zeros Reais):** Mapeamento integral de 5.655 trechos de 10 km da malha federal ao longo de 84 meses (2019 a 2025), totalizando **475.020 observações com 70,61% de zeros reais** (meses sem sinistro).
2. **Eliminação do Target Leakage:** Expurgadas todas as variáveis contextuais pós-evento (`tipo_acidente_*`, `causa_*`, contagens de traçado de via). A matriz preditiva baseia-se unicamente em atributos pré-evento e medições físicas.
3. **Inércia Espacial e Lags Temporais:** Criação de defasagens históricas rigorosas ($t-1, t-2, t-3, t-12$) como proxy instrumental do Volume Diário Médio (VDM) e densidade de tráfego do trecho.
4. **Sentinela DNIT Estrito:** Ausência de medições contemporâneas tratada com sentinela `-1.0` e flag binária `has_dnit_data`, evitando imputações arbitrárias na matriz de referência.
5. **Partição Temporal Fora do Tempo (Out-of-Time):** Aquecimento dos lags em 2019, treino de 2020 a 2023 ($N=271.440$) e teste cego nos anos de 2024 e 2025 ($N=135.720$).
6. **Acurácia em Trechos Seguros:** Erro absoluto médio de **0,3075 acidentes/mês** nos trechos sem ocorrências ($Y=0$), eliminando a compressão de variância presente na V5.

---

## Estudo Experimental Contrafactual: Arquitetura V7 (Simulação DNIT 100%)

Com o objetivo de avaliar o retorno analítico de uma eventual cobertura universal da malha pelo DNIT, estruturou-se o experimento de laboratório **Modelo V7** (`06_Modelagem/experimento_v7_dnit_imputado.py`). 

A simulação eliminou o valor sentinela `-1.0` por meio de uma interpolação espacial-temporal fisiográfica: os trechos sem vistoria contemporânea receberam valores estimados a partir do histórico do próprio trecho, da média da rodovia (BR) no estado (UF) e de médias regionais calibradas exclusivamente no conjunto de treino (2020 a 2023).

### Comparativo Direto: V6 Oficial vs. V7 Experimental

| Métrica / Parâmetro | V6 Oficial (Base Real) | V7 Experimental (DNIT 100% Imputado) | Diagnóstico da Simulação |
| :--- | :---: | :---: | :--- |
| **Cobertura do Pavimento DNIT** | 17,7% (Sentinela `-1.0`) | 100,0% (Imputação Contínua) | Simulação de sensoriamento universal |
| **max_depth** | `None` (Ilimitado) | `28` (Regularizado) | Prevenção de sobreajuste em ruídos de cauda |
| **max_features** | `sqrt` (~15% das colunas) | `0.35` (35% das colunas) | Maior oportunidade para atributos físicos competirem |
| **R² Global (Teste 2024–2025)** | **0,8678** | **0,8701** | Ganho marginal (+0,23 p.p.) |
| **RMSE Global** | **1,2786** | **1,2674** | Redução do erro médio quadrático |
| **MAE Global** | **0,6137 ac/mês** | **0,6208 ac/mês** | Equivalência na escala dimensional |
| **Poisson D² Score** | **0,7964** | **0,7911** | Alta capacidade explicativa em contagens |
| **R² em Trechos com Sinistros ($Y \ge 1$)** | ~0,62 | **0,8513** | **Salto substancial de aderência nos trechos críticos** |
| **Peso Relativo do DNIT (MDI)** | **2,24%** | **8,09%** | **O peso da infraestrutura quase quadruplicou (3,6x)** |

### Hierarquia de Importância das Variáveis no Modelo V7

```text
  01. acidentes_media_movel_3m      : 37,43%  (Inércia de Tráfego)
  02. acidentes_lag_1m              : 22,16%  (Inércia de Tráfego)
  03. acidentes_lag_2m              : 16,98%  (Inércia de Tráfego)
  04. acidentes_lag_12m             : 11,56%  (Inércia de Tráfego / Sazonalidade Anual)
  05. icp (Condição do Pavimento)   :  3,16%  [DNIT] -> Entrou no Top 5 de relevância
  06. icm (Macrotextura Superficial):  2,67%  [DNIT] -> Entrou no Top 6 de relevância
  07. icc (Conservação da Faixa)    :  2,25%  [DNIT] -> Entrou no Top 7 de relevância
  08. dias_fim_semana               :  0,45%  (Calendário)
  09. uf_PA                         :  0,41%  (Geográfico)
  10. dias_feriado                  :  0,33%  (Calendário)
```

### Principais Aprendizados do Experimento V7
1. **Confirmação da Inércia de Tráfego:** A inércia temporal continuou responsável por aproximadamente 88% da força preditiva do modelo, validando a tese de que o volume absoluto de veículos é o principal impulsionador do volume de sinistros.
2. **Ganho de Expressividade da Engenharia Viária:** Ao dispor de 100% de cobertura, as variáveis do DNIT superaram todos os condicionantes de calendário civil e geográficos, alcançando 8,09% de relevância global. A variável de controle `has_dnit_data` caiu para 0,00%, atestando que a árvore abandonou a distinção entre trechos fiscalizados e não fiscalizados para focar exclusivamente na condição física da via.
3. **Calibração de Trechos de Risco:** O $R^2$ restrito aos segmentos com pelo menos uma ocorrência subiu para 0,8513, evidenciando que a qualidade do pavimento auxilia decisivamente na quantificação precisa de sinistros em trechos perigosos.

---

## Estrutura dos Módulos do Repositório

```
├── documento_tcc/                                      <- Monografia completa estruturada em capítulos modulares
│   ├── 00_pre_textual/                                 <- Capa, resumo, abstract e sumário
│   ├── 01_introducao/                                  <- Introdução, problema de pesquisa e objetivos
│   ├── 02_revisao_literatura/                          <- Fundamentação teórica (segurança viária, ML, DNIT)
│   ├── 03_materiais_e_metodos/                         <- Metodologia CRISP-DM, Grade Cartesiana e Poisson
│   ├── 04_resultados/                                  <- Análise comparativa, ablação e diagnósticos
│   ├── 05_discussao/                                   <- Discussão do Efeito Peltzman e letalidade
│   ├── 06_conclusao/                                   <- Conclusão, contribuições e trabalhos futuros
│   └── 07_referencias/                                 <- Referências bibliográficas (ABNT NBR 6023)
│
├── graficos_v6/                                        <- Galeria consolidada com as 20 visualizações oficiais
│
├── 01_Extracao/
│   └── etl_dnit_v6.py                                  <- Pipeline de extração e consolidação DNIT
├── 02_Transformacao/
│   └── transform_prf_v6.py                             <- Agregação transacional da PRF (sem variáveis pós-evento)
├── 04_Integracao_DNIT/
│   └── build_cartesian_grid_v6.py                      <- Construção da Grade Cartesiana com zeros reais
├── 05_Pre_Modelagem/
│   └── feature_engineering_v6.py                       <- Engenharia de lags temporais e calendário civil
├── 06_Modelagem/
│   ├── model_training_v6.py                            <- Treinamento dos modelos V6 (Poisson, MSE, Ablação)
│   ├── model_report_v6.py                              <- Diagnósticos de resíduos e dispersão
│   ├── experimento_v7_dnit_imputado.py                 <- Script do estudo contrafactual V7 (DNIT 100%)
│   └── relatorio_experimento_v7.txt                    <- Relatório analítico do experimento V7
├── 07_Variaveis_Importantes/
│   └── feature_importance_v6.py                        <- Importância MDI e interpretabilidade XAI
├── 08_Fatores_Climaticos/
│   └── cenarios_v6.py                                  <- Simulações de cenários táticos operacionais
├── 09_Analise_Infraestrutura/
│   └── analise_impacto_icp_v6.py                       <- Sensibilidade do pavimento (ICP) e letalidade
├── 10_Documentacao_Geral/
│   ├── relatorio_dataset_v6.md                         <- Auditoria estatística do dataset V6
│   ├── relatorio_modelo_v6.md                          <- Relatório técnico executivo do Modelo V6
│   └── planejamento_dashboard_powerbi_v6.md            <- Especificação técnica do dashboard Power BI
├── 11_Apresentacao_e_Dashboards/                       <- Materiais para banca e pitch executivo
│   ├── ROTEIRO_APRESENTACAO_STORYTELLING_V6.txt        <- Roteiro oficial de defesa (5 atos, 15 min + Q&A)
│   ├── apresentacao_storytelling_pitch_v6.md           <- Roteiro de pitch para banca examinadora
│   └── planejamento_dashboard_powerbi_v6.md            <- Modelo dimensional e medidas DAX
│
├── ROTEIRO_APRESENTACAO_STORYTELLING_V6.txt            <- Roteiro oficial de apresentação na raiz
├── PLANEJAMENTO_DASHBOARD_POWERBI_V6.md                <- Planejamento do Dashboard Executivo na raiz
├── TCC_2026_V6.txt                                     <- Monografia consolidada na íntegra
└── gerar_todos_graficos_v6.py                          <- Script gerador automatizado das figuras oficiais
```

---

## Catálogo de Visualizações na Pasta `graficos_v6/`

| Arquivo | Categoria | Descrição Técnica |
| :--- | :--- | :--- |
| `01_previsto_vs_real_v6.png` | Diagnóstico | Dispersão Real contra Predito no teste cego fora do tempo (2024–2025). |
| `residuos_v6.png` | Diagnóstico | Histograma e curva KDE dos resíduos, centrados rigorosamente em zero. |
| `03_discriminacao_zeros_v6.png` | Diagnóstico | Capacidade do modelo de discriminar trechos sem acidentes de trechos com sinistros. |
| `feature_importance_v6.png` | Explicabilidade | Ranking MDI das variáveis preditivas (Lags, Infraestrutura, Calendário, UF). |
| `dnit_mdi_importancia_v6.png` | Explicabilidade | Importância das variáveis de engenharia viária (`icp`, `icc`, `icm`). |
| `dnit_mdi_vs_permutation_v6.png` | Explicabilidade | Comparação entre ganho de impureza (MDI) e Permutation Importance. |
| `c1_lineplot_previsto_vs_real_v6.png` | Série Temporal | Série temporal mês a mês no Brasil (2024–2025) comparando previsto e real. |
| `c1_barplot_impacto_feriado_clima_v6.png` | Cenários | Impacto de fins de semana e feriados nacionais no volume predito. |
| `c4_barplot_sazonalidade_mensal_v6.png` | Sazonalidade | Padrão sazonal anual ao longo dos doze meses com picos em julho e dezembro. |
| `cenario_01_final_semana_v6.png` | Cenários | Sensibilidade do modelo à quantidade de dias de final de semana no mês. |
| `cenario_05_extremos_v6.png` | Cenários | Amplitude dinâmica: trecho pacificado (0,09 ac/mês) versus crítico (29,35 ac/mês). |
| `cenario_05_top5_trechos_v6.png` | Cenários | Top 5 trechos federais com maior volume predito de sinistros. |
| `comparativo_geral_cenarios_v6.png` | Cenários | Síntese consolidada comparativa dos cenários operacionais da PRF. |
| `impacto_geral_icp_v6.png` | Infraestrutura | Curva de sensibilidade do ICP (0 a 100): Modelo V6-Full versus V6-Ablation. |
| `dnit_pdp_icp_acidentes_v6.png` | Infraestrutura | Curva de Dependência Parcial (PDP) do pavimento com faixas conceituais do DNIT. |
| `dnit_ceterisparibus_ab_test_v6.png` | Infraestrutura | Experimento *Ceteris Paribus* isolando o asfalto no trecho crítico da BR-381/MG. |
| `maior_reducao_icp5_70_v6.png` | Infraestrutura | Maior ganho individual absoluto obtido por recapeamento viário. |
| `top5_reducao_icp5_70_v6.png` | Infraestrutura | Top 5 trechos com maior redução absoluta de sinistros pós-reforma. |
| `dnit_showcase2_ranking_top10_v6.png` | Infraestrutura | Priorização de dez trechos críticos vistoriados pelo DNIT para intervenção. |
| **`efeito_peltzman_e_severidade_dnit_v6.png`** | **Destaque Central** | **Painel tríplice: Paradoxo da Frequência versus Letalidade e Vítimas Graves.** |

---

## Como Reproduzir os Resultados

```bash
# 1. Configurar o ambiente virtual:
python3 -m venv .venv
source .venv/bin/activate
pip install -r 10_Documentacao_Geral/bibliotecas_tcc.txt

# 2. Executar o Treinamento Oficial V6 (Poisson + MSE + Ablação):
python3 06_Modelagem/model_training_v6.py

# 3. Gerar os Diagnósticos Estatísticos:
python3 06_Modelagem/model_report_v6.py

# 4. Executar os Módulos de Explicabilidade e Cenários:
python3 07_Variaveis_Importantes/feature_importance_v6.py
python3 08_Fatores_Climaticos/cenarios_v6.py
python3 09_Analise_Infraestrutura/analise_impacto_icp_v6.py

# 5. Executar o Experimento Contrafactual V7 (Opcional / Laboratório):
python3 06_Modelagem/experimento_v7_dnit_imputado.py

# 6. Gerar a Galeria Consolidada de Gráficos (pasta graficos_v6/):
python3 gerar_todos_graficos_v6.py
```

---

## Autores e Instituição

* **Autores:** Victor Hugo Moreira e Guilherme Figueira
* **Orientação:** Prof. Me. Edy Hayashida
* **Instituição:** Universidade Paulista (UNIP) — Ciência da Computação (2026)
