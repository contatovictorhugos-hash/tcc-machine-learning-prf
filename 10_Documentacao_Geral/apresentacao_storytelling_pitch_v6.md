# Roteiro Oficial de Apresentação e Data Storytelling — TCC 2026

**Projeto:** Modelagem Preditiva de Sinistralidade em Rodovias Federais Brasileiras  
**Integração:** Polícia Rodoviária Federal (PRF 2019–2025) & DNIT (2020–2025)  
**Pesquisador:** Victor Hugo Moreira  
**Orientador:** Prof. Me. Edy Hayashida  
**Instituição:** Universidade Paulista (UNIP) — Bacharelado em Ciência da Computação  

---

## 1. Visão Geral e Estratégia Narrativa

### O Arco Narrativo da Apresentação
A apresentação foi estruturada em 5 atos claros e diretos:
1. **O Contexto e a Dor Real (Slides 1 e 2):** A dimensão humana e econômica dos acidentes e a desconexão histórica entre fiscalização policial e engenharia de rodovias.
2. **O Conflito Metodológico (Slide 3):** Por que o modelo V5 parecia perfeito demais ($R^2 = 0{,}9726$) e a decisão de descartá-lo por rigor científico.
3. **A Solução Técnica (Slides 4 e 5):** A construção da Grade Cartesiana com $70{,}61\%$ de zeros reais e a validação no tempo (anos futuros de 2024 e 2025) com Random Forest Poisson.
4. **O Ponto de Virada Científico (Slides 6 e 7):** O Estudo de Controle (Ablação) e o Efeito Peltzman: por que o asfalto bom atrai velocidade, enquanto o asfalto péssimo eleva a letalidade em $+43{,}4\%$.
5. **Ação e Futuro (Slides 8 a 10):** Priorização de obras para o DNIT (onde salvar mais vidas), escalas dinâmicas para a PRF e roteiro de defesa para a banca.

---

## 2. Detalhamento Slide a Slide com Imagens Oficiais

```
================================================================================
SLIDE 1: ABERTURA INSTITUCIONAL E O CUSTO DA INÉRCIA
================================================================================
```

### 1. Objetivo Estratégico
Prender a atenção imediata da banca, mostrando que o projeto resolve uma dor real de alto custo econômico e social, indo além de um simples exercício acadêmico.

### 2. Roteiro de Fala (Pitch)
> "Todos os anos, o trânsito brasileiro consome cerca de 50 bilhões de reais da nossa economia, segundo o IPEA. E analisando apenas as rodovias federais, a Confederação Nacional do Transporte calcula um prejuízo de quase 13 bilhões de reais anuais, com mais de 73 mil acidentes e 6 mil mortes registradas pela PRF em 2024.  
> *(Pausa de 2 segundos)*  
> A resposta tradicional do poder público é quase sempre reativa: a viatura da polícia socorre os feridos após a batida e, muito tempo depois, uma equipe de obras do DNIT tapa os buracos daquele trecho.  
> Mas e se pudéssemos mudar essa lógica? E se a inteligência de dados permitisse antecipar, trecho a trecho de 10 quilômetros, onde o risco de acidentes e mortes vai ser crítico no próximo mês?  
> Essa é a proposta deste trabalho. Ao longo dos próximos minutos, mostrarei como unificamos sete anos de registros da PRF com os dados de infraestrutura do DNIT, superamos armadilhas estatísticas e construímos um modelo validado no tempo, capaz de orientar investimentos e salvar vidas."

### 3. Design e Elementos Visuais
- **Paleta de Cores:** Fundo Dark Navy (`#0B132B`), texto em Branco Fosco (`#F8FAFC`), destaques em Azul Ciano (`#00EBFF`) e Laranja Alerta (`#F97316`).
- **Layout:** Título limpo à esquerda; subtítulo contextualizador em cinza claro (`#94A3B8`).
- **Três Cards Numéricos de Impacto:**
  - Card 1: `5.655 trechos` — Malha federal monitorada em segmentos de 10 km.
  - Card 2: `475.020` — Registros mensais analisados ao longo de 7 anos (2019 a 2025).
  - Card 3: `R$ 12,9 Bi` — Custo anual de acidentes nas rodovias federais (CNT).
- **Rodapé:** Victor Hugo Moreira | Orientação: Prof. Me. Edy Hayashida | Bacharelado em Ciência da Computação (UNIP 2026).

### 4. Imagem Recomendada para o Slide (da Monografia)
* **Figura:** `05_analise_exploratoria_posETL/g1_evolucao_temporal.png` (ou `01_analise_exploratoria/eda5_top10_brs_letais.png`)
* **O que mostra:** Gráfico de evolução anual de acidentes e mortes da PRF (Figura 2 da monografia), evidenciando que 2024 registrou o pico de letalidade recente.
* **Como apontar na fala:** *"Vejam neste gráfico histórico que os acidentes não estão diminuindo sozinhos: 2024 teve o maior número de mortes dos últimos anos, provando a falência do modelo reativo."*

---

```
================================================================================
SLIDE 2: O PONTO CEGO — O ABISMO ENTRE POLICIAMENTO E INFRAESTRUTURA
================================================================================
```

### 1. Objetivo Estratégico
Expor a separação de dados entre PRF e DNIT, justificando o trabalho de integração das duas bases.

### 2. Roteiro de Fala (Pitch)
> "Historicamente, o Brasil cuida das suas rodovias por meio de duas instituições importantes, mas que operam de forma isolada.  
> De um lado, temos a Polícia Rodoviária Federal. A PRF tem dados ricos e diários, com foco no comportamento do motorista: velocidade, ultrapassagem, álcool e chuva.  
> Do outro lado, temos o DNIT. O departamento avalia as condições físicas da pista: buracos, trincas, remendos e desgaste, gerando o Índice de Condição do Pavimento (ICP). Porém, essas vistorias são anuais e cobrem ativamente menos de 30% da malha contínua.  
> Qual é o impacto real da qualidade do asfalto no risco de um acidente grave acontecer?  
> Até hoje, faltava uma análise integrada que cruzasse esses dois mundos de forma rigorosa, isolando a qualidade do asfalto do simples volume de veículos que passa na via."

### 3. Design e Elementos Visuais
- **Layout:** Comparativo em duas colunas (contraste visual).
- **Coluna da Esquerda (PRF — Azul `#2563EB`):**
  - Ícone de viatura / fiscalização.
  - Tópicos: Mais de 470 mil ocorrências tratadas; foco comportamental e climático; registros por evento diário; coordenadas da rodovia e quilômetro.
- **Coluna da Direita (DNIT — Laranja `#EA580C`):**
  - Ícone de pavimento / asfalto.
  - Tópicos: Índices físicos de engenharia (ICP, ICC, ICM); foco estrutural; frequência de vistoria anual; medições ativas em 28,9% da malha federal.
- **Centro do Slide:** Símbolo gráfico de união de dados com a frase: *"Integração Inédita entre Segurança Policial e Engenharia Viária"*.

### 4. Imagem Recomendada para o Slide (da Monografia)
* **Figura:** `01_analise_exploratoria/eda3_causas_acidente.png` (Figura 9 da monografia — Causas Condicionantes)
* **O que mostra:** Ranking de causas registradas pela PRF, amplamente dominadas por fatores humanos (falta de atenção, reação tardia e velocidade incompatível).
* **Como apontar na fala:** *"Para a polícia, a causa quase sempre é humana, como mostra o gráfico da PRF. Mas o estado da pista é o fator silencioso que agrava esses erros, e era exatamente esse cruzamento que faltava no setor público."*

---

```
================================================================================
SLIDE 3: O CONFLITO METODOLÓGICO — A ILUSÃO DO R² DE QUASE 98% (MODELO V5)
================================================================================
```

### 1. Objetivo Estratégico
Apresentar a auditoria que identificou falhas no modelo V5 anterior, demonstrando maturidade acadêmica e integridade científica.

### 2. Roteiro de Fala (Pitch)
> "No estágio anterior deste projeto, chegamos a uma versão chamada Modelo V5. Quando calculamos as métricas de validação, os números pareciam quase perfeitos: um $R^2$ de 0,9726 e um erro médio de apenas 0,41 acidentes por mês.  
> Em problemas complexos de segurança pública, um modelo que acerta quase tudo precisa ser auditado com muita cautela.  
> *(Pausa reflexiva)*  
> Decidimos abrir o pipeline e auditar cada linha de transformação. E descobrimos dois problemas sérios:  
> O primeiro foi o **vazamento de dados contábil**. Na hora de agrupar os acidentes do mês, o código somava as colunas de categorias, como 'acidentes em reta' e 'acidentes em curva'. Só que a soma dos tipos de acidente é exatamente igual ao total de acidentes daquele mês. O modelo não estava prevendo o futuro; estava apenas fazendo uma conta de soma com informações que já continham a resposta.  
> O segundo problema foi o **viés de sobrevivência**. A base só continha meses e trechos onde já tinha acontecido pelo menos um acidente. Esquecia-se de que em mais de 70% das rodovias brasileiras não acontece nenhum acidente durante o mês.  
> Tomamos então a decisão correta para o rigor acadêmico: descartar essas premissas e reconstruir a base de dados do zero."

### 3. Design e Elementos Visuais
- **Layout:** Comparação visual ("O Erro Encontrado vs. A Explicação Simples").
- **Card Vermelho (A Ilusão da V5):**
  - Destaque numérico: `$R^2 = 0{,}9726$` com aviso *"DESQUALIFICADO POR VAZAMENTO CONTÁBIL"*.
  - Fórmula simples em destaque: $\text{Reta} + \text{Curva} + \text{Cruzamento} = \text{Total de Acidentes}$ (Conta óbvia de somar).
- **Card Amarelo (Viés de Sobrevivência):**
  - Ilustração mostrando que trechos com zero acidentes foram ignorados na versão antiga, distorcendo a realidade do país.

### 4. Imagem Recomendada para o Slide (da Monografia)
* **Figura:** `01_analise_exploratoria/eda8_pie_tracado_original.png` (Figura 13 da monografia — Composição do Traçado da Via)
* **O que mostra:** Gráfico de pizza com a distribuição dos acidentes em Reta (58,4%), Curva (13,5%) e Cruzamento/Outros.
* **Como apontar na fala:** *"O modelo antigo somava as fatias dessa pizza dentro de cada mês. Ao somar os acidentes em reta com os em curva, o resultado dava o próprio total. O algoritmo só precisava somar as parcelas, gerando um acerto artificial."*

---

```
================================================================================
SLIDE 4: A ENGENHARIA DE DADOS V6 — GRADE CARTESIANA E ZEROS REAIS
================================================================================
```

### 1. Objetivo Estratégico
Explicar a solução da engenharia de dados: a criação da matriz completa com zeros reais e tratamento correto das ausências do DNIT.

### 2. Roteiro de Fala (Pitch)
> "Para corrigir essas falhas de forma definitiva, construímos a **Grade Cartesiana da Malha Federal**.  
> Mapeamos todas as rodovias federais em 5.655 trechos padronizados de 10 quilômetros. Em seguida, cruzamos cada trecho com todos os 84 meses do período de 2019 a 2025.  
> O resultado foi uma base sólida de **475.020 observações**.  
> Nessa base, **70,61% dos registros são zeros reais**, ou seja, trechos em que o trânsito fluiu com segurança e ninguém se acidentou naquele mês.  
> Para prever o futuro sem trapacear, criamos defasagens históricas limpas: o histórico de acidentes de 1 mês antes, 2 meses antes, a média dos últimos 3 meses e o mesmo mês do ano anterior.  
> E para os trechos em que o DNIT não tinha medição de asfalto naquele ano, não inventamos dados por média. Usamos o valor sentinela `-1.0` com uma etiqueta indicativa, permitindo que a árvore de decisão aprenda regras separadas para trechos monitorados e não monitorados."

### 3. Design e Elementos Visuais
- **Layout:** Fluxo visual do pipeline em 4 blocos sequenciais:
  - Bloco 1: `5.655 trechos x 84 meses` $\rightarrow$ Matriz com $475.020$ registros.
  - Bloco 2: `Inclusão de 70,61% de zeros` $\rightarrow$ $335.430$ meses seguros representados.
  - Bloco 3: `Defasagens temporais limpas` $\rightarrow$ Lags de 1, 2, 3 e 12 meses para capturar o ritmo da via.
  - Bloco 4: `Sentinela do DNIT (-1.0)` $\rightarrow$ Tratamento correto de trechos sem vistoria.

### 4. Imagem Recomendada para o Slide (da Monografia)
* **Figura Principal:** `06_Modelagem/graficos_modelo_v6/03_discriminacao_zeros_v6.png`
* **Figura Complementar:** `documento_tcc/figuras/figura5_distribuicao_cauda_v6.png` (Figura 5 da monografia — Distribuição de Cauda Longa com Zeros)
* **O que mostra:** O gráfico de barras comparativo: para trechos com zero acidentes reais, a barra verde mostra que a previsão média é de **0,28 acidentes/mês**; para trechos com acidentes reais, a barra vermelha mostra **2,71 acidentes/mês**.
* **Como apontar na fala:** *"Vejam a precisão dessa separação: onde a rodovia teve zero acidentes reais, o modelo previu em média 0,28, mostrando que ele aprendeu a reconhecer uma estrada segura sem gerar falsos alarmes."*

---

```
================================================================================
SLIDE 5: A ARQUITETURA PREDITIVA V6 — VALIDAÇÃO NO TEMPO E MODELO POISSON
================================================================================
```

### 1. Objetivo Estratégico
Apresentar a configuração do Random Forest com perda Poisson e o teste estrito nos anos futuros de 2024 e 2025.

### 2. Roteiro de Fala (Pitch)
> "Como ter certeza de que o modelo aprendeu a antecipar o futuro das estradas? Jamais sorteando dados de forma aleatória, pois isso misturaria dias do mesmo mês entre treino e teste.  
> Fizemos uma **Validação Temporal Rígida**:  
> O ano de 2019 serviu para calcular o histórico inicial; os anos de 2020 a 2023 formaram a base de treino, com mais de 271 mil registros; e testamos o algoritmo nos anos de 2024 e 2025, com quase 136 mil registros futuros que o modelo nunca tinha visto.  
> Além disso, como acidentes são contagens que não podem ser negativas e têm muitos zeros, adotamos o critério de **Poisson** nas árvores de decisão. Ao contrário do erro quadrático comum, o Poisson não tenta nivelar todo mundo pela média e erra muito menos em estradas calmas.  
> Em teste cego no futuro, o Modelo V6 Oficial alcançou um $R^2$ legítimo de **0,8678**, com erro médio de apenas **0,61 acidentes por mês** na malha inteira, e de apenas **0,28** nos trechos pacificados."

### 3. Tabela Comparativa de Benchmarking
| Versão | Tipo de Divisão | Função de Erro | $R^2$ no Teste | Erro Médio (MAE) | Situação |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **V5 (Anterior)** | Temporal (24-25) | Quadrático (MSE) | *0,9726* | *0,4181* | Descartado (Vazamento Contábil) |
| **V6-Ablation** | Temporal (24-25) | Poisson | 0,0381 | 1,4550 | Teste sem Histórico de Tráfego |
| **V6-MSE** | Temporal (24-25) | Quadrático (MSE) | 0,8736 | 0,6219 | Benchmark Tradicional |
| **V6-Oficial** | **Temporal (24-25)** | **Poisson Deviance** | **0,8678** | **0,6137** | **Modelo Definitivo Aprovado** |

### 4. Imagem Recomendada para o Slide (da Monografia)
* **Figura:** `documento_tcc/figuras/figura18_analise_residuos_v6.png` (Figura 18 da monografia — Diagnóstico de Resíduos do Modelo V6)
* **O que mostra:** O painel oficial com dois gráficos: o histograma de erros centralizado com precisão no zero (média de erros residual de +0,002) e a nuvem de dispersão Previsto vs. Real acompanhando a reta diagonal perfeita.
* **Como apontar na fala:** *"Este é o diagnóstico de resíduos oficial do TCC. O histograma mostra que mais de 84% das previsões erram por menos de 1 acidente, e a dispersão se mantém firme na diagonal em praticamente toda a faixa operacional."*

---

```
================================================================================
SLIDE 6: ESTUDO DE CONTROLE — O PAPEL DO TRÁFEGO E DO ASFALTO
================================================================================
```

### 1. Objetivo Estratégico
Explicar a diferença entre o volume de tráfego (que dá o tamanho do risco) e o asfalto (que ajusta o perigo naquele ponto).

### 2. Roteiro de Fala (Pitch)
> "Com esses resultados, a banca poderia perguntar: 'O modelo prevê bem porque o asfalto é determinante ou simplesmente porque decorou quais estradas são movimentadas?'  
> Para responder com rigor, fizemos um **Estudo de Controle (chamado tecnicamente de Ablação)**: tiramos todo o histórico recente de acidentes e treinamos o modelo apenas com os dados de asfalto do DNIT, calendário e localização.  
> O resultado: o $R^2$ despencou de 0,8678 para **0,0381**.  
> *(Contato visual com a banca)*  
> Isso faz todo sentido na prática: o asfalto esburacado, sozinho, não bate carro se não houver veículos passando. Uma rodovia isolada no interior, com pavimento péssimo, pode passar o mês inteiro com zero acidentes porque quase não tem tráfego.  
> O histórico de tráfego define o **tamanho da exposição ao risco** (representando mais de 90% da importância do modelo). E a condição da pista do DNIT entra como o **fator modulador** desse risco. É exatamente esse efeito que veremos no próximo slide."

### 3. Design e Elementos Visuais
- **Destaque Numérico:**
  $$\text{Modelo Completo V6: } R^2 = 0{,}8678 \quad \longleftrightarrow \quad \text{Sem Histórico de Tráfego: } R^2 = 0{,}0381$$

### 4. Imagem Recomendada para o Slide (da Monografia)
* **Figura:** `documento_tcc/figuras/figura19_importancia_variaveis_v6.png` (Figura 19 da monografia — Top 20 Variáveis por Importância Preditiva)
* **O que mostra:** Gráfico de barras horizontais detalhando a hierarquia das variáveis: a média móvel trimestral (29,98%), o lag de 2 meses (21,08%), lag de 1 mês (20,81%) e lag de 12 meses (19,59%), seguidos pelas variáveis de Estado e pelos índices do DNIT (ICM, ICP e ICC somando 2,24%).
* **Como apontar na fala:** *"Observem as quatro primeiras barras azuis: elas concentram 91,46% do peso preditivo porque capturam o volume de tráfego do local. As barras verdes do DNIT aparecem logo abaixo, funcionando como o ajuste fino da segurança."*

---

```
================================================================================
SLIDE 7: A DESCOBERTA EMPÍRICA — O EFEITO PELTZMAN E A LETALIDADE REAL
================================================================================
```

### 1. Objetivo Estratégico
Apresentar a principal descoberta da pesquisa: por que a contagem simples de batidas engana e por que o asfalto ruim é muito mais letal.

### 2. Roteiro de Fala (Pitch)
> "Quando analisamos como a quantidade de acidentes varia conforme o Índice de Pavimento (ICP), encontramos um fenômeno clássico conhecido como **Efeito Peltzman** ou compensação de risco.  
> Em estradas com asfalto impecável, os motoristas sentem mais segurança, aumentam a velocidade para 110 ou 120 km/h e fazem mais ultrapassagens. Isso mantém uma quantidade constante de pequenas batidas traseiras e toques laterais.  
> *(Ênfase na voz)*  
> Porém, quando olhamos para a gravidade das vítimas registradas pela PRF, a realidade é muito diferente:  
> Um acidente em pavimento **Péssimo é 43,4% mais letal do que em pavimento Bom**!  
> A taxa de mortes sobe de 7,32% para 10,50% de todos os acidentes registrados.  
> E o índice de pessoas com lesões graves ou mortes salta de 31,8% no asfalto bom para **41,1% no asfalto péssimo**.  
> O motivo é puramente físico: no asfalto destruído, o motorista desvia bruscamente de crateras e invade a contramão, provocando batidas de frente em alta velocidade relativa, além de capotamentos por estouro de pneu. O recapeamento do DNIT não serve apenas para conforto do motorista: ele é um escudo direto contra mortes."

### 3. Imagem Recomendada para o Slide (da Monografia)
* **Figura Principal:** `documento_tcc/figuras/figura23_efeito_peltzman_v6.png` (Figura 23 da monografia — O Efeito Peltzman e a Severidade Real do DNIT)
* **Figura Complementar (opcional):** `documento_tcc/figuras/figura20_pdp_icp_v6.png` (Curva PDP mostrando estabilização da contagem bruta)
* **O que mostra:** Painel tríplice com:
  1. A frequência de batidas quase estável na curva PDP (Efeito Peltzman).
  2. A taxa de mortalidade saltando de **7,32% (Bom)** para **10,50% (Péssimo)**.
  3. O banner com o cálculo oficial: **+43,4% de letalidade no asfalto destruído**.
* **Como apontar na fala:** *"Olhem para este gráfico comparativo: a quantidade de batidas quase não muda, mas a barra de mortes dispara de 7% para 10,5%. É aqui que a ciência de dados prova onde o dinheiro público salva vidas de verdade."*

---

```
================================================================================
SLIDE 8: OPERACIONALIZAÇÃO — DIRETRIZES PRÁTICAS PARA DNIT E PRF
================================================================================
```

### 1. Objetivo Estratégico
Demonstrar a utilidade prática do modelo para as duas instituições governamentais envolvidas.

### 2. Roteiro de Fala (Pitch)
> "Como essa inteligência se traduz em ação prática para o governo?  
> Para o **DNIT**, o modelo cria uma fila técnica baseada em vidas salvas, e não apenas em pressão política. Simulamos a recuperação de todos os 226 trechos em estado crítico no país:  
> O trecho número um em prioridade de vidas é a **BR-277 no Paraná, entre os km 580 e 590, perto de Cascavel**. Com asfalto péssimo e tráfego intenso de caminhões de safra, recuperar esse trecho pode poupar cerca de **8 vítimas severas ou mortes por ano**. Outros pontos críticos identificados foram a **BR-469 em Foz do Iguaçu** e a **BR-101 no sul da Bahia**.  
> Para a **PRF**, entregamos uma ferramenta de planejamento de policiamento preventivo. O modelo identificou os **5 maiores gargalos do Brasil** — liderados pela BR-101 na Bahia (região de Feira de Santana, com mais de 56 acidentes por mês) e a Rodovia Presidente Dutra em Guarulhos. Juntos, esses cinco pontos somaram mais de 6 mil acidentes e 244 mortes em dois anos.  
> Além disso, o modelo capturou a variação sazonal de **28% entre o início do ano e o pico de dezembro**, oferecendo uma régua objetiva para dimensionar os plantões da Operação Rodovida."

### 3. Imagens Recomendadas para o Slide (da Monografia)
* **Coluna Esquerda (DNIT):** `documento_tcc/figuras/figura22_top3_reducao_vitimas_v6.png` (Figura 22 da monografia — Top 3 Trechos com Maior Redução de Vítimas Severas)
  * Mostra as barras da **BR-277/PR km 580**, **BR-469/PR km 100** e **BR-101/BA km 500**.
* **Coluna Direita (PRF):** `documento_tcc/figuras/figura25_top5_hotspots_prf_v6.png` (Figura 25 da monografia — Top 5 Trechos com Maior Sinistralidade na Malha)
  * Mostra a comparação entre o real e o previsto nos 5 gargalos (Feira de Santana, João Pessoa, Dutra Guarulhos e Recife).
* **Como apontar na fala:** *"À esquerda temos a fila técnica do DNIT: a BR-277 em Cascavel evita 8 vítimas por ano se for recapeada. À direita temos a escala da PRF: em vez de espalhar viaturas a esmo, concentramos as equipes nesses 5 pontos onde ocorrem mais de 40 acidentes todo mês."*

---

```
================================================================================
SLIDE 9: CONCLUSÃO EXECUTIVA E IMPACTO DA PESQUISA
================================================================================
```

### 1. Objetivo Estratégico
Sintetizar as três grandes entregas do TCC, deixando uma impressão final de segurança e maturidade.

### 2. Roteiro de Fala (Pitch)
> "Chegamos ao final deste trabalho com três conclusões claras:  
> Primeira: Provamos que a honestidade científica compensa. Identificamos e removemos as falhas do modelo anterior, construindo a primeira Grade Cartesiana da malha federal com 475 mil registros e mais de 70% de zeros reais.  
> Segunda: Consolidamos o Modelo V6 Oficial com critério de Poisson, atingindo $R^2$ de **0,8678** legítimo e erro de apenas **0,28 acidentes por mês** em trechos seguros em teste cego no futuro.  
> Terceira: Provamos numericamente que o asfalto destruído aumenta a letalidade em 43,4%, fornecendo uma ferramenta matemática para que DNIT e PRF priorizem recursos escassos onde eles mais salvam vidas.  
> Esse projeto mostra como a Ciência da Computação pode ajudar a transformar dados públicos abertos em decisões que protegem a vida nas estradas brasileiras."

### 3. Imagem Recomendada para o Slide (da Monografia)
* **Figura:** `documento_tcc/figuras/figura24_cenario_extremos_v6.png` (Figura 24 da monografia — Resolução de Amplitude Dinâmica)
* **O que mostra:** O gráfico comparando a capacidade do modelo em calibrar o risco de ponta a ponta: BR-317/AC prevendo **0,09 acidentes/mês** (trecho seguro) até a BR-101/RJ Niterói-Manilha prevendo **29,35 acidentes/mês** (gargalo metropolitano), provando que o modelo não achata a realidade.
* **Como apontar na fala:** *"Concluímos com esta imagem: o modelo aprendeu a reconhecer o Brasil real, desde uma estrada tranquila na fronteira do Acre com 0,09 até o caos urbano da Niterói-Manilha com quase 30 acidentes mensais."*

---

```
================================================================================
SLIDE 10: ENCERRAMENTO E ARGÜIÇÃO DA BANCA
================================================================================
```

### 1. Roteiro de Fala (Pitch)
> "Agradeço à banca examinadora pela atenção e pelas contribuições, ao meu orientador Professor Edy Hayashida pela condução e à UNIP pela formação ao longo deste curso.  
> O repositório com todo o código-fonte, dados tratados e relatórios de auditoria está à disposição.  
> Estou pronto para as perguntas da banca."

### 2. Elementos Visuais
- QR Code apontando para o repositório GitHub auditado do projeto.
- Contatos: Victor Hugo Moreira e orientação do Prof. Me. Edy Hayashida.
- Logos institucionais: UNIP, PRF e DNIT.

---

## 3. Roteiro de Defesa: Perguntas Prováveis da Banca e Respostas Claras

### Pergunta 1: *"Por que vocês abandonaram o modelo anterior (V5) que parecia ter R² de 0,97?"*
- **Como responder de forma simples e segura:**  
  *"Professor, na versão V5 o modelo parecia ter quase 98% de acerto, mas quando auditamos as variáveis, descobrimos um vazamento contábil: o código estava somando os tipos de acidentes do mês (como reta, curva e cruzamento), e a soma das partes é o próprio total de acidentes. O modelo estava apenas fazendo uma conta de somar. Além disso, a base antiga só tinha meses em que já tinha acontecido acidente, esquecendo os trechos seguros. Decidimos corrigir isso criando a Grade Cartesiana com mais de 70% de zeros reais e testando em anos futuros. O $R^2$ de 0,8678 da V6 é legítimo e reflete a capacidade real de previsão."*

### Pergunta 2: *"Como o modelo lida com os mais de 70% de trechos sem dados de asfalto do DNIT?"*
- **Como responder de forma simples e segura:**  
  *"Não tentamos inventar dados preenchendo com a média, porque isso criaria asfalto falso no computador. Como usamos Random Forest (árvores de decisão), colocamos o valor sentinela `-1.0` e uma etiqueta indicando se havia ou não vistoria. A árvore de decisão aprende regras separadas com muita facilidade: ela cria ramos específicos para trechos monitorados e ramos para trechos não monitorados, sem distorcer as estradas reais."*

### Pergunta 3: *"Por que vocês usaram a função de Poisson em vez do Erro Quadrático tradicional?"*
- **Como responder de forma simples e segura:**  
  *"O erro quadrático tradicional parte do princípio de que os dados se distribuem de forma simétrica em torno de uma média, como peso ou altura. Mas acidentes em rodovias são números inteiros, não podem ser negativos e a imensa maioria dos trechos registra zero acidentes. A formulação de Poisson foi criada pela estatística especificamente para contagens desse tipo: ela não deixa o modelo prever valores negativos e diminui muito o erro em trechos seguros, fazendo o erro cair para apenas 0,28 acidentes por mês onde a pista é calma."*
