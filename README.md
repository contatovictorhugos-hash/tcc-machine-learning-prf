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
| **V1 (Baseline)** | `model_training_v1.py` | Random 80/20 | 100 árvores, `max_features=1.0`, sem poda | 0,9892 | 0,2276 |
| **V2 (Otimizado)** | `model_training_v2.py` | Random 70/30 | 200 árvores, `max_features='sqrt'`, OOB=True | 0,9770 | 0,3450 |
| **V3 (Poisson + Poda)** | `model_training_v3.py` | Random 70/30 | 200 árvores, `criterion='poisson'`, depth=25, leaf=5 | 0,9678 | 0,4011 |
| **V4 (Log-Transform)** | `model_training_v4.py` | Random 70/30 | 300 árvores, `np.log1p(y)`, max_samples=0.8, depth=30 | 0,9556 | 0,4107 |
| **V5 (Temporal Oficial)** | `model_training_v5.py` | **Temporal (2019–23 / 2024–25)** | 200 árvores, max_samples=0.9, min_leaf=2 | **0,9726** | **0,4181** |

### Execução dos Modelos:
```bash
# Executar a comparação completa e imprimir a Tabela 5 consolidada:
python3 06_Modelagem/train_compare_all_models.py

# Ou executar individualmente a arquitetura campeã (V5):
python3 06_Modelagem/model_training_v5.py
```

> 📄 **Auditoria Metodológica:** Para entender a análise forense sobre o comportamento do Modelo V5, a agregação de variáveis de traçado e as recomendações de defesa perante a banca examinadora, consulte o [Relatório Técnico de Auditoria](10_Documentacao_Geral/RELATORIO_AUDITORIA_TCC.md).

---

## 📈 Principais Resultados Empíricos

- **Impacto do Asfalto (ICP):** A varredura nas simulações indicou que, em rodovias críticas com Índice de Condição da Superfície (ICP) abaixo de 25 pontos, um recapeamento local (elevação do ICP de 5 para 70) gera uma **redução estatisticamente prevista entre 12% e 17%** na contagem total de colisões operacionais.
- **Fatores Climáticos e Velocidade:** Chuvas em pistas retas e duplicadas apresentam acréscimo contra-intuitivo (+6,7% no cenário C3), associado à falsa percepção de segurança e excesso de velocidade.
- **Finais de Semana e Feriados:** A sinistralidade atinge picos expressivos aos sábados e domingos (+52,9% no cenário C2) e durante feriados prolongados com alto fluxo turístico.
