# Análise e Modelagem Preditiva de Acidentes em Rodovias Federais (PRF + DNIT)

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RandomForest-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-green)

Este repositório contém todo o pipeline de engenharia de dados, modelagem, e explicabilidade (XAI) do meu Trabalho de Conclusão de Curso (TCC). O objetivo central do projeto é **provar matematicamente o impacto isolado da infraestrutura rodoviária** (medida pelo DNIT) na letalidade e frequência de colisões mapeadas pela Polícia Rodoviária Federal (PRF), isolando os fatores comportamentais e climáticos.

> ⚠️ **Nota sobre os Dados:** Devido aos limites de armazenamento do GitHub, os arquivos massivos (`.csv` originais de > 200MB) e o modelo serializado (`random_forest_model.pkl` de 720MB) **não estão versionados** neste repositório. O foco aqui é a demonstração do Código-Fonte, Metodologia e Gráficos de Resultados.

---

## 🏗️ Arquitetura do Projeto

Para facilitar a validação acadêmica e a leitura da banca, o código-fonte foi estritamente dividido em 10 capítulos operacionais, mapeando exatamente a estrutura da Monografia:

- **01_Extracao:** Dicionários de dados e amostras estruturais do DNIT.
- **02_Transformacao:** Scripts de ETL (limpeza pesada, remoção de nulos vitais e duplicatas).
- **03_Engenharia_Atributos:** Criação de variáveis de sazonalidade e One-Hot Encoding geográfico, além da geração de gráficos EDA iniciais.
- **04_Integracao_DNIT:** Algoritmo customizado de **Interval Join Espacial-Temporal**, mesclando os pontos exatos da PRF com as faixas métricas contínuas avaliadas pelo DNIT.
- **05_Pre_Modelagem:** Agrupamento inovador da variável alvo (Frequência) usando **Buckets de 10km/Mês** para mitigar o problema de zero-inflation (classes esparsas).
- **06_Modelagem:** Treinamento do `RandomForestRegressor` em validação Out-Of-Bag (OOB) e extração de métricas sistêmicas (R² e Erro Médio Absoluto).
- **07_Variaveis_Importantes:** Inspeção da "Caixa-Preta" do modelo revelando via *MDI* e *Permutation Importance* que Características Extrínsecas (Geometria/Clima) governam o risco bruto de acidentes.
- **08_Fatores_Climaticos:** Testes de cenários auditados (ex: Pico de Verão ou Chuvas em Serras), provando não-linearidade.
- **09_Analise_Infraestrutura:** O coração da tese. Simulação estrita *Ceteris Paribus* usando o intercepto gerado pelo algoritmo PDP (Partial Dependence Plot) e Busca de Força Bruta pelo maior Showcase Operacional do Brasil (BR-40 / BR-101), ranqueando o impacto empírico do asfalto (ICP).
- **10_Documentacao_Geral:** Logs detalhados de execução das rotinas locais, dicionários de bibliotecas com motivadores arquiteturais e fluxograma do sistema.

---

## 🔬 Principais Tecnologias e Técnicas Empregadas

- **Stack:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib, Joblib, SHAP.
- **Engenharia de Features:** Modelagem de Buckets Espaciais-Temporais, Imputação de Sentinelas (`-1.0` para vias sem avaliação DNIT respeitando a topologia de Árvores).
- **XAI (Explicabilidade):** MDI (Mean Decrease Impurity) para rank de colunas, PDP (Partial Dependence Plot) para curvas térmicas do Asfalto e simulação estática de cenários limitando as dimensões comportamentais via *Mediana Histórica*.

---

## 📈 Resultados Em Destaque

A varredura completa nas simulações indicou que, em rodovias críticas classificadas pelo Índice de Condição da Superfície (ICP) abaixo de 25 pontos, a infraestrutura atua como catalisador das falhas humanas. Um recapeamento local (alteração do ICP simulado de 5 para 70) nestes buckets críticos gera uma **redução real prevista estatisticamente entre 12% a 17%** na contagem total de colisões operacionais – provando altíssimo Retorno sobre o Investimento de Manutenção (ROMI) na segurança viária.

*(Consulte a pasta `09_Analise_Infraestrutura` para os Gráficos Oficiais comprobatórios).*
