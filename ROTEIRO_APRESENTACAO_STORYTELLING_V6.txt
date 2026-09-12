# Roteiro Oficial de Apresentação e Data Storytelling — TCC 2026

**Projeto:** Modelagem Preditiva de Sinistralidade em Rodovias Federais Brasileiras  
**Integração:** Polícia Rodoviária Federal (PRF 2019–2025) & DNIT (2020–2025)  
**Pesquisadores:** Victor Hugo Moreira & Guilherme Figueira  
**Orientador:** Prof. Me. Edy Hayashida  
**Instituição:** Universidade Paulista (UNIP) — Bacharelado em Ciência da Computação  

---

## 1. Visão Geral e Estratégia Narrativa

### O Arco Dramático da Apresentação
A apresentação foi estruturada segundo a metodologia de narrativa orientada a dados (*Data Storytelling*), dividida em 5 atos:
1. **O Contexto e a Dor Real (Slides 1 e 2):** A dimensão humana e econômica dos acidentes e a desconexão histórica entre fiscalização policial e engenharia de rodovias.
2. **O Conflito Metodológico (Slide 3):** O falso-positivo do modelo V5 ($R^2 = 0{,}9997$) e a decisão de descartá-lo por integridade científica.
3. **A Solução Técnica (Slides 4 e 5):** A construção da Grade Cartesiana com $70{,}61\%$ de zeros reais e a validação cega fora do tempo (*out-of-time*, 2024–2025) com Random Forest Poisson.
4. **O Ponto de Virada Científico (Slides 6 e 7):** O Estudo de Ablação e o Efeito Peltzman — por que o asfalto bom atrai velocidade, enquanto o asfalto péssimo eleva a letalidade em $+43{,}4\%$.
5. **Ação e Futuro (Slides 8 a 10):** Diretrizes prescritivas de intervenção viária para o DNIT, escalas dinâmicas para a PRF e roteiro de defesa para a banca.

---

## 2. Detalhamento Slide a Slide

```
================================================================================
SLIDE 1: ABERTURA INSTITUCIONAL E O CUSTO DA INÉRCIA
================================================================================
```

### 1. Objetivo Estratégico
Prender a atenção imediata da audiência, demonstrando que o projeto responde a um problema de alto custo econômico e de saúde pública, ultrapassando um mero exercício acadêmico de regressão.

### 2. Raciocínio & Justificativa do Storytelling
Antes de expor tabelas de hiperparâmetros ou código, é mandatório estabelecer a relevância do objeto de estudo. Gestores públicos e membros de banca acadêmica conectam-se primeiro à dor e à escala do problema antes de julgar a sofisticação da técnica.

### 3. Roteiro de Fala (Pitch)
> "Todos os anos, as rodovias federais brasileiras registram mais de 60 mil sinistros. Além da perda irreparável de milhares de vidas, o Instituto de Pesquisa Econômica Aplicada estima um prejuízo superior a 12 bilhões de reais anuais ao país.  
> *(Pausa reflexiva de 2 segundos)*  
> A resposta tradicional do Estado costuma ser reativa: a viatura da PRF socorre as vítimas no local e, meses depois, uma frente de obras do DNIT tapa os buracos daquele trecho.  
> Mas e se pudéssemos mudar essa lógica? E se a inteligência computacional permitisse prever, trecho a trecho de 10 quilômetros, onde o risco de colisão e a probabilidade de mortes vão atingir níveis críticos no próximo mês?  
> É essa a resposta deste trabalho. Ao longo dos próximos minutos, demonstraremos como unificamos sete anos de registros transacionais da PRF aos levantamentos estruturais do DNIT, superamos uma grave armadilha metodológica e construímos um modelo auditado capaz de orientar investimentos e salvar vidas."

### 4. Design e Elementos Visuais
- **Paleta de Cores:** Fundo Dark Navy (`#0B132B`), tipografia em Branco Fosco (`#F8FAFC`), destaques em Azul Ciano (`#00EBFF`) e Laranja Alerta (`#F97316`).
- **Layout:** Título limpo à esquerda (corpo 32, negrito); subtítulo contextualizador (corpo 18, cinza claro `#94A3B8`).
- **Tríade de Cards Numéricos (Grid Horizontal):**
  - Card 1: `75.000 km` — Extensão da malha federal analisada.
  - Card 2: `475.020` — Observações espaço-temporais auditadas (2019–2025).
  - Card 3: `R$ 12,3 Bi` — Custo socioeconômico anual de sinistros no Brasil (IPEA).
- **Rodapé:** Victor Hugo Moreira & Guilherme Figueira | Orientação: Prof. Me. Edy Hayashida | Bacharelado em Ciência da Computação (UNIP 2026).

---

```
================================================================================
SLIDE 2: O PONTO CEGO — O ABISMO ENTRE POLICIAMENTO E INFRAESTRUTURA
================================================================================
```

### 1. Objetivo Estratégico
Expor a fragmentação de dados no setor público, justificando o esforço de engenharia reversa e consolidação de bases heterogêneas.

### 2. Raciocínio & Justificativa do Storytelling
Prepara a audiência para a complexidade do pipeline. Se a banca não compreender que a PRF e o DNIT não conversavam em termos de esquema de dados, o mérito da Grade Cartesiana desenvolvida não seria devidamente reconhecido.

### 3. Roteiro de Fala (Pitch)
> "Historicamente, o Brasil gerencia suas rodovias por meio de duas instituições de excelência, mas que operam como ilhas isoladas de informação.  
> De um lado, temos a Polícia Rodoviária Federal. A PRF gera registros ricos, transacionais e imediatos. Para a polícia, o acidente é predominantemente comportamental: excesso de velocidade, ultrapassagem indevida, ingestão de álcool, condições climáticas.  
> Do outro lado, temos o DNIT. O departamento avalia a integridade física da via: afundamento de trilha de roda, trincamento do asfalto, remendos e desgaste superficial, gerando o Índice de Condição do Pavimento (ICP). Porém, os levantamentos do DNIT são anuais e cobrem apenas 28,91% das observações da grade espaço-temporal federal.  
> *(Pergunta retórica dirigida à banca)*  
> Qual é o peso exato da qualidade do asfalto na probabilidade de alguém sofrer um acidente grave?  
> Até hoje, a literatura nacional carecia de uma modelagem unificada que isolasse o impacto da infraestrutura física sem contaminar a previsão com o fluxo inercial de veículos."

### 4. Design e Elementos Visuais
- **Layout:** Split-screen (duas colunas simétricas em contraste).
- **Coluna da Esquerda (PRF — Azul `#2563EB`):**
  - Ícone de viatura/policiamento.
  - Tópicos: Mais de 400 mil registros transacionais; foco comportamental e climático; granularidade diária pontual; alta precisão de coordenadas geográficas.
- **Coluna da Direita (DNIT — Laranja `#EA580C`):**
  - Ícone de pavimento/perfilômetro rodoviário.
  - Tópicos: Índices de engenharia (ICP, ICC, ICM); foco estrutural; frequência de medição anual; cobertura contemporânea em 28,91% das observações (137.352 de 475.020 registros).
- **Centro do Slide:** Símbolo gráfico de quebra de fluxo com badge vermelho: *"Fragmentação de Governança de Dados"*.

---

```
================================================================================
SLIDE 3: O CONFLITO METODOLÓGICO — A FALÁCIA DO R² = 0,9997 (MODELO V5)
================================================================================
```

### 1. Objetivo Estratégico
Apresentar com transparência o processo de auditoria científica que desmontou o modelo preliminar V5, demonstrando rigor técnico e maturidade acadêmica.

### 2. Raciocínio & Justificativa do Storytelling
Este é o divisor de águas da apresentação. Muitos trabalhos acadêmicos escondem erros preliminares. Expor como um $R^2$ de 0,9997 foi identificado como uma ilusão contábil consolida autoridade imediata perante os avaliadores.

### 3. Roteiro de Fala (Pitch)
> "No estágio preliminar deste TCC, desenvolvemos a versão V5 do nosso pipeline — um Random Forest com 197 features treinado sobre 139.590 observações. Quando rodamos as métricas de validação, os números pareciam extraordinários: um $R^2$ de 0,9997 e um erro absoluto médio de apenas 0,038 acidentes por mês.  
> Na prática, qualquer modelo preditivo com essa precisão quase perfeita deve acender um alerta vermelho no laboratório.  
> *(Pausa dramática)*  
> Decidimos abrir o modelo e auditar linha por linha. O diagnóstico foi implacável: tínhamos um duplo vício metodológico.  
> O primeiro foi o **Target Leakage (Vazamento de Dados Contábil)**. Na agregação mensal, colunas que registravam o tipo de sinistro — como 'colisão frontal' e 'saída de pista' — foram mantidas na matriz de entrada. Ora, a soma dos tipos de sinistro é algebricamente idêntica ao total de sinistros. O modelo não estava prevendo acidentes; estava apenas somando parcelas já ocorridas.  
> O segundo vício foi o **Viés de Sobrevivência**. A base continha unicamente trechos onde havia registro de sinistro ($Y \ge 1$). Ignorava-se que mais de 70% da malha rodoviária federal não tem acidentes em um determinado mês. O modelo previa ~3,5 acidentes até para rodovias desertas do Norte do país.  
> Diante desse achado, tomamos a única decisão eticamente aceitável na ciência: descartar o modelo V5 e reconstruir toda a engenharia de dados do zero."

### 4. Design e Elementos Visuais
- **Layout:** Comparação visual de contraste ("O Erro Auditado vs. O Diagnóstico Técnico").
- **Card Vermelho (A Armadilha V5):**
  - Destaque numérico: `$R^2 = 0{,}9997$` com tarja diagonal *"DESQUALIFICADO POR TARGET LEAKAGE"*.
  - Fórmula conceitual em destaque: $Y = \sum \text{tipo\_acidente}_k$ (Identidade $1+1=2$).
  - Detalhe: 197 features, 139.590 observações (apenas $Y \ge 1$), MAE = 0,038.
- **Card Amarelo (Viés de Sobrevivência):**
  - Gráfico ilustrando a compressão para a média: predição de ~3,5 acidentes/mês até mesmo para rodovias desertas do Norte do país.
  - Detalhe: 0% de zeros reais na base V5 — todos os meses seguros foram descartados.
- **Gráfico Conceitual:** Diagrama de fluxo com uma seta vermelha indicando variáveis pós-evento retroalimentando a matriz de treino.

---

```
================================================================================
SLIDE 4: A ENGENHARIA DE DADOS V6 — GRADE CARTESIANA E ZEROS REAIS
================================================================================
```

### 1. Objetivo Estratégico
Apresentar a solução arquitetural que viabilizou o modelo oficial V6: a criação da matriz de 475 mil observações com zeros reais e tratamento de ausências.

### 2. Raciocínio & Justificativa do Storytelling
Demonstra que a integridade preditiva nasce da engenharia dos dados. Sem explicar a Grade Cartesiana, a banca não compreenderia a robustez do particionamento out-of-time do slide seguinte.

### 3. Roteiro de Fala (Pitch)
> "Para eliminar o viés de sobrevivência e o vazamento contábil, construímos a **Grade Cartesiana Integral da Malha Federal**.  
> Segmentamos toda a extensão sob circunscrição federal em 5.655 trechos padronizados de exatamente 10 quilômetros. Em seguida, cruzamos cada um desses trechos com todos os 84 meses do período de 2019 a 2025.  
> O resultado é uma matriz de **475.020 linhas** com **22 colunas** — um dataset de 32,6 MB completamente auditado.  
> Deste universo, **70,61% das observações são zeros reais** — são 335.430 linhas que representam trechos pacificados em que a rodovia operou com segurança ao longo do mês. A média global é de apenas 0,997 acidentes por trecho/mês.  
> Para alimentar o modelo sem vazar o futuro, geramos variáveis defasadas puras: lags de 1, 2, 3 meses e o lag sazonal de 12 meses — variáveis autorregressivas que funcionam como proxy instrumental do Volume Diário Médio de veículos.  
> E para os 71,09% de observações sem medição contemporânea do DNIT, recusamos qualquer preenchimento artificial por média. Adotamos o sentinela estrito `-1.0` acompanhado da flag binária `has_dnit_data`, permitindo ao algoritmo aprender regras distintas para trechos monitorados e não monitorados."

### 4. Design e Elementos Visuais
- **Layout:** Linha de montagem de dados (Pipeline visual em 4 etapas sequenciais com ícones modernos).
- **Etapa 1:** `5.655 trechos (10 km) x 84 meses` $\rightarrow$ Matriz com $475.020$ observações.
- **Etapa 2:** `Inclusão de 70,61% de Zeros Reais` $\rightarrow$ $335.430$ linhas sem acidentes mapeadas.
- **Etapa 3:** `Lags Temporais Puros` $\rightarrow$ Defasagens $t-1, t-2, t-3, t-12$ para capturar inércia.
- **Etapa 4:** `Sentinela DNIT Estrito` $\rightarrow$ `-1.0` + flag `has_dnit_data` (137.352 obs. com cobertura = 28,91%).
- **Visual de Destaque:** Inserção do gráfico `03_discriminacao_zeros_v6.png`:
  - Barra Verde: Trechos Seguros (Real = 0) $\rightarrow$ Predição média de **0,28 acidentes/mês**.
  - Barra Vermelha: Trechos com Sinistro (Real $\ge$ 1) $\rightarrow$ Predição média de **2,71 acidentes/mês**.

---

```
================================================================================
SLIDE 5: A ARQUITETURA PREDITIVA V6 — TESTE FORA DO TEMPO E LOSS POISSON
================================================================================
```

### 1. Objetivo Estratégico
Apresentar a parametrização do Random Forest com função de perda Poisson Deviance e a validação estrita em conjunto de teste out-of-time (2024–2025).

### 2. Raciocínio & Justificativa do Storytelling
Responde diretamente ao rigor acadêmico. Justifica o uso de uma loss apropriada para dados de contagem inflacionados em zero e comprova a capacidade de generalização no futuro.

### 3. Roteiro de Fala (Pitch)
> "Como testar se o modelo realmente aprendeu a antecipar o futuro das rodovias? Jamais com validação cruzada aleatória tradicional, que mistura registros do mesmo mês entre treino e teste.  
> Adotamos um **Split Temporal Rígido (Out-of-Time)**:  
> O ano de 2019 serviu para aquecimento dos lags temporais; os anos de 2020 a 2023 constituíram o conjunto de treino com 271.440 observações; e colocamos o modelo à prova nos anos de 2024 e 2025, com 135.720 registros que o algoritmo nunca viu durante o treinamento.  
> O Modelo V6 Oficial utiliza um Random Forest Regressor com 200 árvores de decisão, critério de partição por **Poisson Deviance**, amostragem bootstrap de 90% por árvore, mínimo de 5 amostras por folha e seleção aleatória de $\sqrt{p}$ features por nó — totalizando 42 variáveis preditivas.  
> Em teste cego fora do tempo, o Modelo V6 Oficial alcançou um $R^2$ legítimo de **0,8678**, score $D^2$ Poisson de **0,7964**, RMSE de **1,2786** e um Erro Absoluto Médio global de **0,6137 acidentes por mês**, caindo para apenas **0,2889** em trechos seguros."

### 4. Design e Elementos Visuais
- **Layout:** Tabela de Benchmarking à esquerda e Gráficos de Resíduos à direita.
- **Tabela de Benchmarking (Fonte: relatorio_modelo_v6.md):**
  | Arquitetura | Divisão Amostral | Função de Perda | $R^2$ Teste | MAE Teste | MAE Zeros | Diagnóstico |
  | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
  | **V5 (Descartado)** | Temporal (19–23 / 24–25) | MSE | *0,9997* | *0,038* | *Inexist.* | Target Leakage + Viés de Sobrevivência |
  | **V6-Ablation** | Out-of-Time (24–25) | Poisson Deviance | 0,0381 | 1,4550 | 0,9554 | Estudo de Controle (sem lags) |
  | **V6-MSE** | Out-of-Time (24–25) | Squared Error | 0,8736 | 0,6219 | 0,3075 | Benchmark MSE |
  | **V6-Full (Oficial)** | **Out-of-Time (24–25)** | **Poisson Deviance** | **0,8678** | **0,6137** | **0,2889** | **Campeão Oficial** |
- **Nota Técnica:** O V6-MSE obteve R² ligeiramente superior (0,8736 vs. 0,8678), porém o V6-Full (Poisson) supera em D² Score (0,7964 vs. 0,7924) e em MAE nos trechos seguros (0,2889 vs. 0,3075), sendo mais adequado para dados de contagem com excesso de zeros.
- **Hiperparâmetros do Modelo Oficial V6-Full:**
  | Parâmetro | Valor |
  | :--- | :--- |
  | `n_estimators` | 200 |
  | `criterion` | `'poisson'` (Poisson Deviance) |
  | `max_features` | `'sqrt'` (~15% das 42 features por nó) |
  | `min_samples_leaf` | 5 |
  | `max_samples` | 0.9 (bootstrap 90%) |
  | `max_depth` | `None` (ilimitado) |
  | `random_state` | 42 |
- **Gráficos Integrados:**
  - `01_previsto_vs_real_v6.png`: Nuvem de dispersão aderente à reta $y = x$.
  - `residuos_v6.png`: Histograma de resíduos centrado estritamente no zero (média dos resíduos $\approx 0{,}000$).

---

```
================================================================================
SLIDE 6: ESTUDO DE ABLAÇÃO — FLUXO INERCIAL VS. INFRAESTRUTURA VIÁRIA
================================================================================
```

### 1. Objetivo Estratégico
Isolar cientificamente o papel do asfalto frente à inércia de fluxo veicular, respondendo à pergunta sobre a capacidade preditiva da infraestrutura pura.

### 2. Raciocínio & Justificativa do Storytelling
Previne interpretações errôneas da banca de que a condição viária explicaria isoladamente toda a variação de acidentes. O estudo de ablação é o padrão-ouro de rigor em Machine Learning.

### 3. Roteiro de Fala (Pitch)
> "Diante desses resultados, uma dúvida legítima da banca poderia ser: 'O modelo prevê bem porque o asfalto é determinante ou simplesmente porque decorou quais trechos têm tráfego intenso?'  
> Para sanar essa questão, executamos um **Estudo de Ablação**: treinamos um modelo eliminando todos os 4 lags temporais e variáveis de inércia, mantendo puramente os índices do DNIT, o calendário e as variáveis geográficas — 38 features no total.  
> O $R^2$ despencou de 0,8678 para **0,0381** e o MAE saltou de 0,61 para **1,45 acidentes/mês**.  
> *(Pausa com contato visual firme)*  
> Esse resultado é fundamental para a engenharia de tráfego. Ele prova matematicamente que o asfalto degradado, isolado, não causa sinistros se não houver veículos trafegando. Uma rodovia de pista simples no interior do Maranhão, com pavimento péssimo, pode registrar zero acidentes no mês por ausência de volume veicular.  
> A inércia de tráfego define a **exposição ao risco**. A infraestrutura física do DNIT atua como **fator modulador** desse risco. E é essa modulação que examinaremos a seguir."

### 4. Design e Elementos Visuais
- **Layout:** Gráfico de importância de variáveis (`feature_importance_v6.png`) com anotações de categorização.
- **Visual Central:** Barras horizontais agrupadas por classes cromáticas:
  - Azul (`#2563EB`): Lags de Tráfego (`acidentes_lag_1m`, `media_movel_3m`, `lag_12m`) compondo os primeiros lugares em MDI.
  - Verde (`#10B981`): Métricas de Pavimento DNIT (`icp`, `icc`, `icm`, `has_dnit_data`).
  - Laranja (`#F59E0B`): Variáveis de Calendário (`dias_fim_semana`, `dias_feriado`).
  - Roxo (`#8B5CF6`): Unidade Federativa (`uf`).
- **Box Comparativo:**
  $$\text{V6-Full (42 features): } R^2 = 0{,}8678 \quad \longleftrightarrow \quad \text{V6-Ablação (38 features, sem lags): } R^2 = 0{,}0381$$

---

```
================================================================================
SLIDE 7: A DESCOBERTA CIENTÍFICA — O EFEITO PELTZMAN E A SEVERIDADE REAL
================================================================================
```

### 1. Objetivo Estratégico
Apresentar a principal contribuição empírica da pesquisa: a comprovação do Efeito Peltzman na frequência de colisões e a revelação do impacto do pavimento na letalidade e severidade de vítimas.

### 2. Raciocínio & Justificativa do Storytelling
Este é o ponto alto do discurso. Revela uma verdade contra-intuitiva suportada pelos dados e justifica todo o investimento público em infraestrutura rodoviária.

### 3. Roteiro de Fala (Pitch)
> "Quando analisamos o comportamento da frequência de acidentes em função do Índice de Condição do Pavimento (ICP), deparamo-nos com o clássico **Efeito Peltzman** — a teoria da compensação de risco.  
> Em rodovias com pavimento excelente ($ICP \ge 70$), a frequência de pequenas colisões não se reduz a zero. Pelo contrário: com o asfalto impecável, os motoristas elevam a velocidade operacional e aumentam a confiança, mantendo registros frequentes de colisões traseiras e toques laterais leves.  
> *(Mudança de tom para ênfase máxima)*  
> Contudo, ao cruzarmos os dados de pavimento com os registros de vítimas da PRF, a realidade se revela:  
> Um acidente ocorrido em pavimento **Péssimo ($ICP \le 20$) é 43,4% mais letal do que em pavimento Bom**!  
> A taxa de óbito salta de 7,32% para 10,50% dos acidentes registrados.  
> Em pista destruída, as colisões não são toques urbanos: são estouros de pneu com capotamento e invasões da contramão para desviar de crateras, resultando em choques frontais de altíssima energia dissipada.  
> A restauração viária (Péssimo $\rightarrow$ Bom) reduz o Índice de Vítimas Severas — que soma mortos e feridos graves — de **41,1% para 31,8%**, uma redução de 9,3 pontos percentuais no risco de traumas severos e mortes. O asfalto bom do DNIT não serve apenas para conforto; ele é uma intervenção profilática contra mortes e traumatismos irreversíveis."

### 4. Design e Elementos Visuais
- **Layout:** O infográfico mestre tríplice gerado pelo script (`efeito_peltzman_e_severidade_dnit_v6.png`) como elemento central da tela.
- **Painel 1 (Azul):** O Paradoxo da Frequência (Efeito Peltzman) — barras mostrando estabilidade da contagem mensal em vias rápidas.
- **Painel 2 (Gradiente Vermelho-Verde):** Taxa Real de Letalidade da PRF — Péssimo ($10{,}50\%$) vs. Bom ($7{,}32\%$).
- **Painel 3 (Verde Vitória):** Redução do Índice de Vítimas Severas de $41{,}1\%$ para $31{,}8\%$ (-9,3 p.p.).
- **Tabela Completa de Severidade por Faixa de Pavimento (Fonte: relatorio_modelo_v6.md):**
  | Faixa ICP | Mortalidade | Vítimas Severas |
  | :--- | :---: | :---: |
  | Péssimo (0–20) | 10,50% | 41,1% |
  | Ruim (21–40) | 8,83% | 37,7% |
  | Regular (41–60) | 7,99% | 36,3% |
  | Bom (61–80) | 7,32% | 31,8% |
  | Ótimo (81–100) | 8,62% | 35,8% |
- **Badge Superior em Destaque:** `+43,4% DE LETALIDADE EM PAVIMENTOS PÉSSIMOS (ICP <= 20)`.

---

```
================================================================================
SLIDE 8: OPERACIONALIZAÇÃO — PRESCRIÇÃO PARA PRF E DNIT
================================================================================
```

### 1. Objetivo Estratégico
Demonstrar a aplicação prática dos achados em políticas públicas, dividindo as prescrições entre fiscalização (PRF) e obras estruturais (DNIT).

### 2. Raciocínio & Justificativa do Storytelling
Garante que a pesquisa não termine como um arquivo estático de código, mas como um instrumento direto de tomada de decisão governamental.

### 3. Roteiro de Fala (Pitch)
> "Como essa inteligência transforma a rotina operacional a partir de amanhã?  
> Para o **DNIT**, o modelo encerra a era dos recapeamentos baseados unicamente em critérios de fluxo ou demanda política. A partir dos nossos testes de intervenção *ceteris paribus*, criamos o **Ranking de Retorno Social da Pavimentação**: trechos críticos como os segmentos da BR-277/PR (km 580–590, ICP=19,5) e da BR-101/BA (km 500–510, ICP=18,1) oferecem o maior volume de vidas salvas por milhão de reais investido em recomposição asfáltica. No trecho da BR-277, a restauração evitaria cerca de 8 vítimas severas por ano.  
> Para a **PRF**, entregamos uma ferramenta de alocação dinâmica de patrulhamento. O modelo quantificou que feriados prolongados geram incrementos de **+14,8% a +22,3%** no risco predito em corredores turísticos e rodovias litorâneas, enquanto trechos de escoamento agrícola mantêm padrão constante.  
> A amplitude dinâmica do modelo é comprovada: a BR-317/AC (km 0–10) registra predição de apenas 0,09 acidentes/mês, enquanto a BR-101/RJ (km 310–320) atinge 29,35 acidentes/mês — amplitude que o modelo V5 era incapaz de reproduzir.  
> Em vez de manter escalas uniformes, as delegacias da PRF podem posicionar viaturas e radares cirurgicamente nos trechos que entram em faixa de alto risco no calendário civil."

### 4. Design e Elementos Visuais
- **Layout:** Duas colunas com cards de ação governamental.
- **Coluna Esquerda (Prescrição DNIT):**
  - Exibição de recorte do gráfico `top5_reducao_icp5_70_v6.png` e `dnit_showcase2_ranking_top10_v6.png`.
  - Diretriz: Critério de custo-benefício de restauração indexado a *Vítimas Graves Evitáveis*.
- **Coluna Direita (Prescrição PRF):**
  - Exibição de recorte do gráfico `c1_barplot_impacto_feriado_clima_v6.png` e `cenario_01_final_semana_v6.png`.
  - Diretriz: Deslocamento preditivo de viaturas para os trechos de alto volume em feriados e fins de semana.

---

```
================================================================================
SLIDE 9: CONCLUSÃO EXECUTIVA E IMPACTO DA PESQUISA
================================================================================
```

### 1. Objetivo Estratégico
Sintetizar as contribuições originais do trabalho, reforçando o rigor científico, a ética nos dados e as entregas operacionais — todas ancoradas exclusivamente no Modelo V6 Oficial.

### 2. Raciocínio & Justificativa do Storytelling
Amarra o início e o fim da narrativa, relembrando a superação do problema e deixando uma impressão definitiva de maturidade técnica.

### 3. Roteiro de Fala (Pitch)
> "Chegamos à conclusão deste trabalho com três entregas fundamentadas:  
> Primeira: Provamos que a honestidade metodológica compensa. Detectamos e expurgamos o target leakage da V5, construindo a primeira Grade Cartesiana da malha federal com 475.020 registros e 70,61% de zeros reais.  
> Segunda: Consolidamos o Modelo V6 com loss Poisson Deviance — 200 árvores, 42 features auditadas, split temporal cego — alcançando $R^2 = 0{,}8678$, $D^2 = 0{,}7964$ e erro de apenas 0,28 acidentes/mês em trechos seguros, em teste fora do tempo com 135.720 observações inéditas.  
> Terceira: Evidenciamos o Efeito Peltzman e quantificamos que o asfalto destruído aumenta a letalidade em 43,4% e eleva o índice de vítimas severas de 31,8% para 41,1% (redução de 9,3 p.p. com a restauração viária), fornecendo subsídio matemático para priorização orçamentária no setor de transportes.  
> Mais do que algoritmos e métricas, este projeto entrega uma ponte funcional entre a engenharia de computação e a proteção à vida humana nas estradas brasileiras."

### 4. Design e Elementos Visuais
- **Layout:** Três pilares verticais de impacto, com ícones de validação e métricas em caixas com bordas suaves.
- **Pilar 1:** Integridade e Auditoria (Zero Leakage, $N=475.020$, Split Temporal Cego 2024–2025).
- **Pilar 2:** Performance e Robustez ($R^2 = 0{,}8678$, $D^2 = 0{,}7964$, MAE Zeros $= 0{,}2889$, RMSE $= 1{,}2786$).
- **Pilar 3:** Contribuição Científica (Efeito Peltzman, Letalidade $+43{,}4\%$, Vítimas Severas $41{,}1\% \rightarrow 31{,}8\%$).

---

```
================================================================================
SLIDE 10: ENCERRAMENTO E ARGÜIÇÃO DA BANCA
================================================================================
```

### 1. Objetivo Estratégico
Finalizar a apresentação formal, agradecer aos envolvidos e disponibilizar canais de transparência de código e documentação para o início da sabatina.

### 2. Roteiro de Fala (Pitch)
> "Agradecemos à banca examinadora pelo tempo e pela oportunidade de avaliação, ao nosso orientador Professor Edy Hayashida pelo direcionamento técnico e à UNIP pela formação proporcionada.  
> O repositório com todos os scripts, relatórios de auditoria e rotinas de reprodutibilidade encontra-se à disposição.  
> Estamos prontos para as considerações e perguntas da banca."

### 3. Design e Elementos Visuais
- **Elementos:** QR Code apontando para o repositório GitHub auditado; dados de contato e referências institucionais.

---

## 3. Roteiro de Defesa: Perguntas Prováveis da Banca e Respostas Técnicas

### Pergunta 1: *"Por que abandonar um modelo com R² de 0,9997 (V5) em favor de um com 0,8678 (V6)?"*
- **Resposta do Candidato:**  
  *"O $R^2$ de 0,9997 da versão V5 decorria de uma armadilha clássica de Machine Learning: o vazamento de alvo (target leakage). Na agregação mensal, variáveis como `tipo_acidente_Colisao_frontal` e `tipo_acidente_Saida_de_pista` foram somadas e mantidas como preditores. Matematicamente, a soma das contagens das categorias de acidentes é identicamente igual ao total de acidentes ($Y = \sum \text{tipo\_acidente}_k$). O modelo não previa; apenas somava parcelas já ocorridas. Além disso, a V5 só continha 139.590 observações onde houve acidente ($Y \ge 1$), gerando um viés de sobrevivência que fazia o modelo prever ~3,5 acidentes para qualquer estrada, mesmo em regiões desertas. O $R^2$ de 0,8678 da Arquitetura V6 é honesto: ele foi calculado em um conjunto de teste estritamente cego (2024–2025) sobre uma grade de 475.020 pontos com 70,61% de zeros reais e 42 features auditadas sem vazamento."*

### Pergunta 2: *"Por que escolheram o Random Forest como algoritmo e não um modelo mais moderno como XGBoost, LightGBM ou redes neurais?"*
- **Resposta do Candidato:**  
  *"A escolha do Random Forest foi deliberada e sustentada por três fatores técnicos:*  
  *Primeiro, a **interpretabilidade nativa**. O Random Forest oferece importância de variáveis por Mean Decrease in Impurity (MDI) e por Permutation Importance sem necessidade de bibliotecas externas de explicabilidade. Em um trabalho de TCC que visa orientar políticas públicas para a PRF e o DNIT, a capacidade de explicar quais fatores elevam o risco é tão importante quanto a acurácia numérica.*  
  *Segundo, a **robustez ao overfitting em dados tabulares com alta dimensionalidade esparsa**. Nosso dataset possui 42 features com one-hot encoding de 27 UFs e 4 estações do ano, além de 70,61% de zeros. A agregação por bagging do Random Forest é naturalmente resistente a ruído nesse cenário, sem exigir a calibração extensiva de taxa de aprendizado, regularização L1/L2 e early stopping que os modelos de boosting demandam.*  
  *Terceiro, a **compatibilidade nativa com a função de perda Poisson Deviance** no scikit-learn (`criterion='poisson'`), que é fundamental para dados de contagem com excesso de zeros. Essa implementação garante previsões estritamente não-negativas por construção, já que as folhas armazenam médias ponderadas de Poisson.*  
  *Ressalto que, como validação cruzada, comparamos o V6-Full (Poisson) contra o V6-MSE (Squared Error) no mesmo dataset e split temporal. O V6-Poisson apresentou melhor D² Score (0,7964 vs. 0,7924) e menor MAE em trechos seguros (0,2889 vs. 0,3075), confirmando a adequação do critério de Poisson. A experimentação com LightGBM e XGBoost é apontada na seção de trabalhos futuros da monografia como uma frente de investigação posterior."*

### Pergunta 3: *"Como o modelo lida com os mais de 71% de dados ausentes do DNIT sem imputação sintética?"*
- **Resposta do Candidato:**  
  *"Modelos baseados em árvores de decisão particionam o espaço hiperdimensional de forma não-linear. Em vez de realizar um backfill artificial que inventaria condições fictícias de pavimento, adotamos uma estratégia de sentinela estrito: o valor `-1.0` nas métricas contínuas (ICP, ICC, ICM) acompanhado da variável binária indicadora `has_dnit_data`. Quando `has_dnit_data = 0`, o Random Forest isola automaticamente esses registros em ramos de decisão separados, sem distorcer o cálculo do gradiente de impureza nos ramos onde há medição real do DNIT. Das 475.020 observações, 137.352 (28,91%) possuem medição contemporânea — é nesse subconjunto que o modelo efetivamente aprende a relação entre degradação do asfalto e sinistralidade. Validamos essa abordagem com o Experimento V7, que simulou cobertura universal do DNIT por imputação espacial-temporal: o peso relativo das variáveis de infraestrutura quase quadruplicou (de 2,24% para 8,09% no MDI), confirmando que a esparsidade real é o principal limitante do peso do DNIT no modelo atual."*

### Pergunta 4: *"Por que a função de perda Poisson Deviance foi escolhida no lugar do Erro Quadrático Médio (MSE)?"*
- **Resposta do Candidato:**  
  *"Contagens de acidentes em segmentos espaciais homogêneos seguem distribuições discretas com inflação de zeros e assimetria positiva à direita. O critério tradicional de MSE pressupõe normalidade e homocedasticidade dos erros, o que tende a subestimar eventos raros e pode gerar previsões negativas para trechos de baixíssimo risco — algo fisicamente impossível para uma contagem. A função de perda Poisson Deviance penaliza desvios proporcionalmente à intensidade esperada ($\lambda$), garantindo previsões estritamente não-negativas e ajustando o peso do erro conforme a magnitude do trecho. Na prática, isso reduziu o erro absoluto em trechos seguros ($Y=0$) de 0,3075 (V6-MSE) para 0,2889 (V6-Poisson), enquanto o D² Score saltou de 0,7924 para 0,7964 — uma métrica que mede diretamente a proporção do desvio de Poisson explicada pelo modelo."*

### Pergunta 5: *"As variáveis de lag temporal não são apenas uma forma sofisticada de 'decorar' o histórico do trecho? Qual a limitação de usar lags como proxy do volume de tráfego?"*
- **Resposta do Candidato:**  
  *"Essa é uma limitação reconhecida e documentada na monografia. Os lags temporais ($t-1, t-2, t-3, t-12$) são proxies instrumentais do Volume Diário Médio (VDM), que idealmente seria medido por contadores de tráfego do DNIT ou praças de pedágio. O lag de 12 meses captura a sazonalidade anual e o perfil estrutural do trecho (urbano vs. rural, corredor de carga vs. turístico), enquanto os lags de curto prazo capturam choques transitórios de fluxo.*  
  *A principal limitação é que o modelo não capta mudanças abruptas e exógenas no padrão de tráfego — como a abertura de um novo contorno rodoviário que desvia veículos de um trecho historicamente perigoso. O Estudo de Ablação comprova essa dependência: sem os lags, o R² despenca para 0,0381. Reconhecemos que a inércia temporal responde por mais de 85% da força preditiva e que a incorporação do VDM real, quando disponível, é uma frente obrigatória de trabalhos futuros. Isso está explícito nas propostas da Seção 6.4 da monografia."*

### Pergunta 6: *"O Experimento V7 com DNIT 100% imputado apresentou resultados melhores em trechos críticos ($R^2 = 0{,}8513$ para $Y \ge 1$). Por que ele não foi adotado como modelo oficial?"*
- **Resposta do Candidato:**  
  *"A decisão de manter o V6 como oficial e posicionar o V7 como estudo experimental foi uma escolha de integridade científica. O V7 utiliza imputação espacial-temporal — preenchendo os 71% de ausências do DNIT com valores estimados a partir do histórico do próprio trecho e de médias regionais. Embora as métricas globais sejam comparáveis ($R^2 = 0{,}8701$ no V7 vs. $0{,}8678$ no V6), o V7 introduz uma premissa forte: que a condição do pavimento pode ser inferida retroativamente para trechos nunca vistoriados. Isso seria equivalente a inventar dados de sensor onde o perfilômetro do DNIT nunca transitou.*  
  *O V6, ao contrário, opera exclusivamente com dados reais verificados. O sentinela `-1.0` é honesto: onde não há dado, o modelo sabe que não há dado. O V7 serve como prova de conceito de que, caso o DNIT amplie sua cobertura de medição de 28,91% para patamares próximos de 100%, o peso das variáveis de infraestrutura pode saltar de 2,24% para 8,09% no ranking de importância, tornando o modelo muito mais sensível ao estado do asfalto."*

### Pergunta 7: *"O modelo pode ser generalizado para rodovias estaduais ou municipais? Quais as condições para isso?"*
- **Resposta do Candidato:**  
  *"O modelo atual foi treinado exclusivamente com dados da malha federal sob jurisdição da PRF, que tem cobertura censitária dos boletins de ocorrência e esquema de dados padronizado nacionalmente. Rodovias estaduais e municipais apresentam três barreiras para generalização direta:*  
  *Primeira, a heterogeneidade dos registros de acidentes entre os 27 estados e os mais de 5.500 municípios — cada polícia militar estadual adota taxonomias diferentes para classificar os sinistros.*  
  *Segunda, a ausência quase total de dados de condição de pavimento em escala estadual — o DNIT fiscaliza apenas rodovias federais.*  
  *Terceira, os perfis de tráfego são estruturalmente distintos — rodovias estaduais em regiões metropolitanas têm padrões de commuting urbano que diferem radicalmente dos corredores de carga interestadual do modelo V6.*  
  *Dito isso, a arquitetura metodológica é generalizável: a Grade Cartesiana com zeros reais, o split temporal estrito e a loss Poisson podem ser replicados para qualquer jurisdição que disponibilize dados transacionais de sinistros e medições de pavimento com cobertura mínima. A adaptação seria mais um esforço de engenharia de dados do que de reformulação algorítmica."*
