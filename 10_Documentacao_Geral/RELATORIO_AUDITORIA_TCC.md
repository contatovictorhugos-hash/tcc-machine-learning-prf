# Relatório Técnico de Auditoria, Conciliação e Reorganização do TCC
**Projeto:** Análise e Modelagem Preditiva de Acidentes em Rodovias Federais Brasileiras (PRF + DNIT)  
**Autores:** Victor Hugo Moreira e Guilherme Figueira  
**Orientador:** Prof. Edy Hayashida — Universidade Paulista (UNIP, 2026)  
**Repositório Oficial:** [tcc-machine-learning-prf](https://github.com/contatovictorhugos-hash/tcc-machine-learning-prf.git)  

---

## 1. Resumo Executivo da Reorganização

Inicialmente, o repositório oficial `tcc-machine-learning-prf` continha apenas uma versão parcial do pipeline (o script `model_training.py` implementava exclusivamente o Modelo V2). Os scripts que deram origem à evolução das **5 Arquiteturas (V1 a V5)** descritas no Capítulo 4 da monografia escrita estavam dispersos no repositório de laboratório `tcc-2`.

Através desta consolidação:
1. **Todas as 5 arquiteturas** foram integradas e padronizadas na pasta `06_Modelagem/`.
2. Os scripts com caminhos fixos de máquinas individuais (`c:\Users\yuri\...`) foram convertidos para **caminhos dinâmicos e relativos**.
3. O script comparativo unificado [`train_compare_all_models.py`](file:///Users/marciohugo/Documents/tcc-machine-learning-prf/06_Modelagem/train_compare_all_models.py) e o relatório [`model_comparison.txt`](file:///Users/marciohugo/Documents/tcc-machine-learning-prf/06_Modelagem/model_comparison.txt) foram restaurados.
4. Os módulos de **Assimetria/Outliers** (`05_Pre_Modelagem/`), **Simulação de Cenários** (`08_Fatores_Climaticos/`) e **Impacto do Asfalto ICP** (`09_Analise_Infraestrutura/`) receberam os gráficos e rotinas de suporte correspondentes.

---

## 2. As 5 Arquiteturas de Modelagem do TCC

A Tabela abaixo consolida o estudo evolutivo das arquiteturas conforme documentado na **Tabela 4 e Tabela 5 da Monografia**:

| Parâmetro / Métrica | Modelo V1 (Baseline) | Modelo V2 (Otimizado) | Modelo V3 (Poisson + Poda) | Modelo V4 (Log-Transform) | Modelo V5 (Temporal Oficial) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Script Implementado** | `model_training_v1.py` | `model_training_v2.py` | `model_training_v3.py` | `model_training_v4.py` | `model_training_v5.py` |
| **Estratégia de Split** | Random 80% / 20% | Random 70% / 30% | Random 70% / 30% | Random 70% / 30% | **Temporal (Treino: 2019–23 / Teste: 2024–25)** |
| **Pré-processamento** | Nenhum | Nenhum | Nenhum | `np.log1p(y)` | Nenhum (Escala Original) |
| **n_estimators** | 100 árvores | 200 árvores | 200 árvores | 300 árvores | 200 árvores |
| **max_features** | 1.0 (Todas) | `'sqrt'` (Subspace) | `'sqrt'` (Subspace) | `'sqrt'` | `'sqrt'` (Feature Subspace) |
| **criterion** | `squared_error` (MSE) | `squared_error` (MSE) | `poisson` | `squared_error` | `squared_error` (MSE) |
| **max_samples** | 1.0 | 1.0 | 1.0 | 0.8 (Subsampling) | 0.9 (Bagging) |
| **Regularização** | `depth=None` | `depth=None` | `depth=25, leaf=5` | `depth=30, leaf=3` | `depth=None, min_leaf=2` |
| **$R^2$ Reportado** | **0,9892** | **0,9770** | **0,9678** | **0,9556** | **0,9726** |
| **MAE Reportado** | **0,2276** | **0,3450** | **0,4011** | **0,4107** | **0,4181** |
| **MSE Reportado** | 0,2985 | 0,6418 | 0,8980 | 1,2398 | 0,8787 |

---

## 3. Diagnóstico Metodológico: Inconsistências Críticas

Durante a análise forense do código-fonte em relação ao texto da monografia, foram identificadas **três divergências metodológicas fundamentais** que todo pesquisador e avaliador de banca deve compreender:

### 3.1. A Causa do Alto $R^2$ ($0{,}9726$) no Modelo V5 (*Target Leakage*)
* **O que o texto afirma:** A monografia relata que o Modelo V5 obteve $R^2 = 0{,}9726$ e $MAE = 0{,}4181$ no teste temporal (2024–2025), provando que o modelo generaliza para o futuro sem *data leakage*.
* **O que o código realmente fez:** No script de pré-modelagem (`pre_modeling.py` / `pre_modeling_and_train.py`), os dados transacionais de acidentes foram agrupados em buckets de 10 km por mês. As colunas booleanas foram agregadas via **soma**:
  ```python
  elif col.startswith('is_') or col.startswith('is'):
      agg_dict[col] = 'sum'
  ```
* **O vazamento matemático:**  
  Como cada registro na base da PRF é um acidente:
  $$\text{sum}(is\_tracado\_via\_Reta) = \text{Qtd de acidentes em linha reta naquele trecho/mês}$$
  $$\text{sum}(is\_tracado\_via\_Curva) = \text{Qtd de acidentes em curva naquele trecho/mês}$$
  $$\text{sum}(is\_tracado\_via\_Intersecao) = \text{Qtd de acidentes em interseção naquele trecho/mês}$$
  
  Somando essas variáveis, temos:
  $$\text{acidentes\_reta} + \text{acidentes\_curva} + \text{acidentes\_outros} \approx \text{quantidade\_acidentes (Variável Alvo Y)}$$

* **Conclusão:** O modelo alcançou $R^2 = 0{,}97$ no conjunto de teste de 2024–2025 não porque previu o futuro a partir de condições físicas da rodovia, mas porque **a soma dos acidentes por traçado de 2024–2025 já estava presente nas colunas de entrada do conjunto de teste**. O modelo aprendeu uma decomposição contábil da variável alvo.

### 3.2. Viés de Sobrevivência (O Problema dos Zeros Reais)
* O TCC defende que a agregação em buckets de 10 km resolveu o problema de *Zero-Inflation*.
* Contudo, a base utilizada foi obtida executando `groupby()` diretamente sobre a base de acidentes da PRF.
* **Onde estão os trechos sem acidentes?** Se um trecho de 10 km passou meses sem registrar nenhum acidente, ele simplesmente não aparece na tabela da PRF. Logo, a base modelada contém apenas $Y \ge 1$.
* O modelo, portanto, estima o volume condicionado à ocorrência prévia de pelo menos um acidente, excluindo a maioria segura da malha federal brasileira.

### 3.3. Comportamento das Simulações *Ceteris Paribus*
Nos scripts de simulação de cenários (`gerar_cenarios_showcase.py`, `cenarios_tcc.py`), ao simular "pista em curva" ou "chuva", o código passa `is_tracado_via_Curva = 1`.
Como o modelo foi treinado com somas de acidentes e não com variáveis booleanas, o valor `1` foi interpretado como *"ocorreu exatamente 1 acidente nessa condição"*. Isso explica por que quase todos os cenários simulados convergiram para previsões estreitas em torno de **1,36 a 1,60 acidentes/mês**.

---

## 4. Guia Estratégico para a Apresentação na Banca

Se um membro da banca questionar as métricas ou a metodologia, **nunca tente defender que um modelo de previsão de acidentes futuros atinge $R^2 = 0{,}97$ sem vazamento**. Na literatura científica de engenharia de tráfego, acidentes futuros possuem estocasticidade elevada, e $R^2$ reais costumam ficar entre $0{,}25$ e $0{,}50$.

### Resposta Recomendada para a Banca:
> *"Durante a evolução do projeto, investigamos a fundo o comportamento das variáveis agregadas. Identificamos que a agregação por soma das variáveis contextuais (`is_*`) funcionou como um classificador da composição geométrica dos sinistros, o que explica a elevada correlação matemática observada nos modelos V1 a V5. Em trabalhos futuros e em um ambiente de produção contínua, o pipeline deve migrar para atributos físicos cadastrais da rodovia (ex.: percentual de curvatura do trecho via DNIT) e matrizes de presença/ausência completa, garantindo previsões causais puras."*

Essa postura demonstra:
1. **Maturidade científica e honestidade intelectual** (altamente valorizadas por bancas avaliadoras).
2. Domínio pleno sobre os dados, o pipeline e as limitações inerentes ao tratamento de dados públicos.
3. Compreensão clara da diferença entre correlação agregada e causalidade preditiva.

---

## 5. Estrutura Final Consolidada do Repositório

```
tcc-machine-learning-prf/
│
├── 01_Extracao/                      # Scripts de ingestão da PRF e DNIT
├── 02_Transformacao/                 # Limpeza, nulos, IQR e tipagem
├── 03_Engenharia_Atributos/          # Sazonalidade, flags de calendário e traçado
├── 04_Integracao_DNIT/               # Interval Join ASOF espacial (BR x KM)
│
├── 05_Pre_Modelagem/                 # Bucketing espaço-temporal (10km / Mês)
│   ├── pre_modeling.py               # Agrupamento e One-Hot Encoding
│   ├── analise_assimetria_outliers.py# Estudo de skewness e IQR (portado do tcc-2)
│   ├── pipeline_pre_modelagem_ia.md  # Documentação técnica do bucket
│   └── graficos_assimetria_outliers/ # 7 gráficos analíticos de assimetria e cauda
│
├── 06_Modelagem/                     # As 5 Arquiteturas Completas do TCC
│   ├── model_training_v1.py          # V1 Baseline (Random 80/20, 100 árvores, max_features=1.0)
│   ├── model_training_v2.py          # V2 Otimizado (Random 70/30, 200 árvores, sqrt)
│   ├── model_training_v3.py          # V3 Otimizado + Poda (Poisson, depth=25, leaf=5)
│   ├── model_training_v4.py          # V4 Log-Transform (log1p, 300 árvores, max_samples=0.8)
│   ├── model_training_v5.py          # V5 Temporal Oficial (2019-23 treino / 2024-25 teste)
│   ├── train_compare_all_models.py   # Executor comparativo unificado gerando a Tabela 5
│   ├── model_comparison.txt          # Tabela oficial das métricas V1-V5
│   ├── resultados_modelo_tcc.txt     # Log de métricas detalhadas
│   └── model_report.py               # Diagnóstico de resíduos e dispersão
│
├── 07_Variaveis_Importantes/         # XAI - Importância das Variáveis
│   ├── feature_importance_v5.png     # Gráfico oficial do Modelo V5
│   ├── dnit_mdi_importancia.png      # MDI features do modelo
│   └── dnit_mdi_vs_permutation.png   # Comparativo MDI vs Permutação
│
├── 08_Fatores_Climaticos/            # Simulação de Cenários Contextuais
│   ├── gerar_cenarios_showcase.py    # Gerador dos cenários 1 a 5
│   ├── cenario5_caso_real.py         # Estudo de caso real de alta criticidade
│   ├── cenarios_tcc.py               # Cenários C1 a C4 auditados
│   ├── analise_cenarios_tcc.txt      # Relatório oficial dos cenários
│   └── relatorio_cenarios_showcase.txt # Relatório do showcase de cenários
│
├── 09_Analise_Infraestrutura/        # Impacto do DNIT (ICP/ICC/ICM)
│   ├── analise_impacto_icp_v5.py     # Curvas de sensibilidade de asfalto V5
│   ├── analise_dnit_shap.py          # TreeExplainer SHAP
│   ├── analise_dnit_infraestrutura.py# MDI, PDP e Ceteris Paribus A/B
│   └── Gráficos de impacto V5:       # residuos_v5.png, impacto_geral_icp_v5.png,
│                                     # maior_reducao_icp5_70_v5.png, top5_reducao_icp5_70_v5.png
│
├── 10_Documentacao_Geral/            # Governança e Relatórios Oficiais
│   ├── ContextArquiteture.txt        # Especificação CRISP-DM
│   ├── bibliotecas_tcc.txt           # Justificativa das dependências
│   ├── execution_log.txt             # Logs consolidados
│   └── RELATORIO_AUDITORIA_TCC.md    # Este relatório completo de auditoria
│
└── README.md                         # Atualizado com a arquitetura completa V1-V5
