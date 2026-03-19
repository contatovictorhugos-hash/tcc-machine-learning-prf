# Instruções de Integração: Base DNIT x Base PRF (TCC)

**Contexto:**
Você está atuando no processamento e limpeza dos dados do DNIT (Departamento Nacional de Infraestrutura de Transportes) para um modelo de Machine Learning (Random Forest Regressor) focado em prever a quantidade de acidentes em Rodovias Federais. Eu sou a IA responsável pelo tratamento da base da PRF (Polícia Rodoviária Federal). Nossas bases precisam ser cruzadas perfeitamente para que o modelo aprenda a relação entre infraestrutura (DNIT) e acidentes (PRF).

Aqui estão as regras estritas e padronizações mecânicas que você deve seguir na base do DNIT para garantir que o nosso JOIN seja um sucesso:

## 1. A Chave Universal (BR + KM)
Nossa integração **DEVE OBRIGATORIAMENTE** ocorrer pelos eixos geográficos estruturais da via. Esqueça Latitude ou Longitude para o Join, pois eles têm muita variação. Tudo será feito pela Rodovia e pelo Quilômetro.
- Na base da PRF, eu já construí uma chave composta primária chamada `br_km`. 
- **O que você deve fazer:** Na base do DNIT, garanta que você isole rigorosamente o número da rodovia (ex: `116`, `316`, `262`) e a quilometragem correspondente daquele trecho.
- **Formatação da Chave:** Para que o `.merge()` do Pandas funcione perfeitamente, você deve criar uma coluna na base do DNIT exatamente com a formatação: `str(br) + '-' + str(km)` (Ex: `"316-84"`, `"262-380.9"`). *Nota: O km não pode ter vírgulas, apenas pontos ou inteiros limpos, sem espaços.*

## 2. Granularidade e Agrupamento (O Problema do Trecho)
O DNIT costuma mapear as rodovias pro subtrechos contínuos (Ponto Inicial X -> Ponto Final Y). A PRF mapeia acidentes em um quilômetro (Ponto exato Z).
- **O que você deve fazer:** Você precisa explodir ou interpolar os trechos contínuos do DNIT para que eles possuam uma linha por KM, ou criar uma função que mapeie o KM do acidente da PRF (ex: KM 84) para cair dentro do intervalo do trecho do DNIT (ex: KM 80 ao KM 90). Caso consiga expandir isso por KM exato antecipadamente e criar a coluna com a chave `br-km`, nosso merge será direto por O(1).

## 3. O Que Deve Ser Trazido do DNIT (Feature Selection)
Eu já gerei *features* a partir das métricas da PRF (como se é reta ou curva, tipo de pista natural, etc). A base do DNIT nos interessa por fornecer métricas que a polícia *não tem*:
- Traga variáveis qualitativas e de conservação: Estado do Pavimento (Ótimo, Bom, Ruim, Péssimo).
- Traga variáveis de Sinalização: Estado da Pintura, Semáforos, Placas.
- Traga o VDM (Volume Diário Médio): O fluxo de veículos que passa ali por dia é a variável exógena de trafegabilidade mais forte para o modelo RF.
- Atenção: Converta todas as variáveis qualitativas textuais do DNIT em **variáveis categóricas** (`df['coluna'].astype('category')`) ou já aplique um método *One-Hot Encoding* se preferir, pois isso economizará extrema memória do Colab.

## 4. O Que Fazer Quando Terminar?
Após limpar e organizar as condições rodoviárias do DNIT baseadas nos KMs:
- Deixe o dataframe do DNIT (ex: `df_dnit_processed.csv`) salvo e pronto.
- Nós (ou o Cientista de Dados humano executando o fluxo) faremos um `Left Join` onde a Esquerda (Left) é a base de Infraestrutura Espacial / Temporal ou o Painel completo de KMs do DNIT, e a Direita (Right) serão os countings de acidentes da PRF que eu finalizei, agrupados pela mesma chave.

**Restrição Metodológica Crítica:**
Lembre-se da regra de ouro do TCC: O modelo usará **Random Forest Regressor** e sob hipótese alguma os dados de entrada (X) podem sofrer normalização ou padronização linear (Min-Max Scaling, Z-score, etc). Apenas os limpe, trate nulos (preservando o conhecimento real do dado faltante) e formate a tipagem!

Bom trabalho! Sincronizamos as chaves quando a sua base gerar a coluna `br_km`.
