3 MATERIAIS E MÉTODOS


Este capítulo tem como objetivo definir e detalhar todos os materiais e metodologias empregadas neste estudo,  desde a fase de planejamento até a entrega de resultados com a visualização dos resultados em um dashboard.

3.1 Materiais


        Esta seção do projeto apresenta as ferramentas, linguagens e plataformas utilizadas no desenvolvimento do projeto, incluindo o ambiente de programação, bibliotecas de análise de dados, ferramentas de visualização e demais recursos tecnológicos empregados durante o processo de análise e modelagem.
3.1.1 Linguagem Python


A linguagem de programação utilizada para o desenvolvimento deste estudo foi a Python. Trata-se de uma linguagem criada pelo cientista da computação holandês Guido van Rossum, foi oficialmente lançada em 1991. O Python é uma linguagem de programação interpretada de alto nível, no qual seus pilares fundadores são simplicidade sintática, legibilidade e produtividade.
        Ademais, a linguagem busca reduzir a distância entre o raciocínio humano e a lógica computacional. Para isso, entrega um aprendizado facilitado e a manipulação de informações de forma mais direta. Devido a uma sintaxe mais clara, próxima da linguagem humana, e a presença de estruturas de dados de alto nível (VAN ROSSUM, 2026).
        No contexto deste projeto, tal escolha justifica-se pela robustez da linguagem no tratamento de grandes volumes de dados, amplamente adotada em projetos de ciência de dados, aprendizado de máquina e análise estatística, pela qual conta com uma grande quantidade de bibliotecas especializadas disponíveis para manipulação e análise de dados.
        Esta linguagem de programação é caracterizada pela sua capacidade de abstração aliada à eficiência computacional. Em termos práticos, isso significa que tarefas complexas de análise e modelagem podem ser feitas com um número reduzido de linhas de código. Tal característica favorece tanto a reprodutibilidade de experimentos quanto a clareza na implementação dos métodos empregados, e legibilidade do código, aspectos essenciais em pesquisas científicas (MCKINNEY, 2022).
        Além disso, Python possui um ecossistema consolidado de bibliotecas voltadas à manipulação de dados estruturados e não estruturados, visualização de informações e construção de modelos de aprendizado de máquina. Essas bibliotecas funcionam como instrumentos especializados dentro de um laboratório computacional, cada uma voltada para um conjunto específico de tarefas. Dessa forma, a linguagem atua como uma camada integradora capaz de realizar diversas etapas do pipeline de ciência de dados, permitindo a execução de processos como limpeza, transformação e modelagem dos dados de forma estruturada e escalável (MCKINNEY, 2022).
        No presente trabalho, a linguagem Python foi utilizada para implementar rotinas de preparação dos dados, análise exploratória e desenvolvimento de modelos preditivos. A escolha dessa tecnologia se justifica não apenas por ser a mais popular no campo da ciência de dados, mas também pela sua capacidade de integrar diferentes ferramentas analíticas em um fluxo único de processamento, favorecendo a organização metodológica do projeto e a consistência dos resultados obtidos.
3.1.2 Google Colab


O Google Colaboratory (ou Colab) é um ambiente de desenvolvimento, estilo notebook Jupyter, gratuito e executado totalmente na nuvem. O objetivo desse software é democratizar o acesso a ferramentas de Ciência de Dados e a pesquisa em machine learning, permitindo que qualquer pessoa execute códigos em Python no navegador de forma online, sem qualquer instalação ou configuração prévia.
Com isso, a escolha dessa ferramenta se justifica, pois resolve desafios de reprodutibilidade e observabilidade inerentes a projetos de Ciência de Dados. Ao oferecer acesso a recursos de hardware na nuvem de forma gratuita, o que é crucial para o processamento e modelagem de big data. Além disso, a arquitetura pré-configurada garante a execução de bibliotecas utilizadas no projeto como NumPy, Pandas e Scikit-learn.


3.1.3 Bibliotecas


        Primeiramente, a implementação de um pipeline de Ciência de Dados moderno, desde as fases de entendimento do negócio até a implantação, exige o uso robusto de diversas bibliotecas. Com isso, as bibliotecas selecionadas fazem parte da chamada stack tecnológica do projeto que permite manipular grande volume de dados, realizar análises, gerar gráficos e por fim a treinamento de modelos como o random forest. A seguir o detalhamento das bibliotecas utilizadas no projeto e suas justificativas de escolha para esse complexo cenário.

1. Pandas


É uma biblioteca de código aberto, que atua diretamente nas etapas de extração e transformação, se destacando como motor principal de manipulação de dados para todo o pipeline ETL moderno. Desenvolvida por Wes McKinney em 2008 durante sua passagem pela AQR Capital Management (MCKINNEY, 2010). No contexto desse projeto foi utilizado para ler todos os arquivos originais, remover nulos, filtrar vazamentos de dados, criar novas colunas, e executar o "ASOF Join" (merge avançado por intervalo geográfico) integrando os dados da PRF com os dados de infraestrutura do DNIT.

2. NumPy


O NumPy, estabelecido em 2005 por Travis Oliphant através da unificação de bibliotecas ancestrais, é a fundação sobre a qual quase toda a computação científica em Python é construída (OLIPHANT, 2005). Seu objetivo primordial é o fornecimento de objetos de matriz multidimensional e rotinas otimizadas em C para cálculos matemáticos. No contexto desse trabalho atua como uma grande calculadora por trás de todos os cálculos, vital para a criação de novas colunas como, `is_final_semana`, e cálculo eficiente de medidas de assimetria.


3. Scikit-Learn


Scikit-Learn é o pacote central de modelagem de muitos projetos de Ciência de Dados, pois tem como objetivo democratizar o acesso a algoritmos de aprendizado de máquina. Concebida inicialmente por David Cournapeau em 2007 como um projeto do Google Summer of Code e posteriormente amadurecida por pesquisadores do INRIA. Neste projeto, foram empregadas diversas funcionalidades da biblioteca, tais como a função`RandomForestRegressor`, algoritmo preditivo, o `train_test_split`, dividir a base de dados em treino e teste e por fim a avaliação do modelo com `sklearn.metrics`, para calcular R² e MAE. 


4. Seaborn e Matplotlib

        A visualização de dados neste estudo foi estruturada através do ecossistema composto pelas bibliotecas Matplotlib e Seaborn, ferramentas que permitem a transmutação de matrizes numéricas em representações gráficas de alta qualidade.
O Matplotlib, concebido originalmente por John D. Hunter em 2003, é a biblioteca fundamental de plotagem para a linguagem Python, projetada para emular as capacidades gráficas do MATLAB em um ambiente de código aberto (HUNTER, 2007). Complementarmente, utilizou-se o Seaborn, desenvolvido por Michael Waskom (2021), que atua como uma interface de alto nível construída sobre o Matplotlib. O Seaborn introduz estilos visuais modernos, facilitando a exploração de padrões complexos em grandes volumes de dados (WASKOM, 2021).
Neste trabalho, a sinergia entre ambas foi essencial para a Análise Exploratória de Dados. O Matplotlib foi empregado para o controle refinado da estrutura dos gráficos, os eixos e legendas, enquanto o Seaborn foi utilizado para a geração de Matrizes de Correlação, heatmaps, Histogramas de Distribuição e outros, permitindo identificar visualmente a sazonalidade e os pontos críticos de sinistros nas rodovias federais antes da etapa de modelagem preditiva.


5. Holidays


        A biblioteca Holidays desempenha um papel crucial, ao resolver um grande desafio logístico e computacional de identificar datas comemorativas de forma programática. O projeto open source automatiza o cálculo de feriados nacionais, estaduais e municipais, lidando com a complexidade de datas móveis como o Carnaval e a Sexta-Feira Santa, por exemplo. Com isso, foi utilizado na etapa de engenharia de atributos ao criar variáveis dummies como `is_feriado`,`is_final_semana` e outras.


6. Joblib


A ferramenta Joblib foi empregada no projeto através da função joblib.dump para a serialização do modelo Random Forest treinado, exportando-o para o formato .pkl. A escolha do Joblib em detrimento do módulo Pickle nativo do Python justifica-se por sua alta eficiência na manipulação de grandes matrizes NumPy que compõem a estrutura da floresta aleatória. A biblioteca utiliza técnicas de memmapping, que permitem a leitura rápida do modelo em disco para a execução do pipeline de previsões diárias, garantindo que a fase de inferência ocorra com baixo consumo de memória.


7. SciPy


O SciPy é uma biblioteca fundamental para a computação científica em Python, servindo como base para operações matemáticas avançadas. Fundada por Travis Oliphant, Eric Jones e Pearu Peterson em 2001, a biblioteca consolidou décadas de algoritmos numéricos em uma interface de alto desempenho (VIRTANEN et al., 2020).
        Neste estudo, utilizou-se especificamente o submódulo scipy.stats para a realização de diagnósticos estatísticos descritivos durante a Análise Exploratória de Dados. A função scipy.stats.skew foi fundamental para quantificar matematicamente a assimetria dos dados de sinistros da PRF. Essa métrica evidenciou o fenômeno de "Zero-Inflation", onde a grande maioria dos trechos rodoviários apresenta zero acidentes, enquanto poucos pontos críticos concentram a maioria das ocorrências. Com essa avaliação de altíssima assimetria, skewness, serviu de base teórica para descartar modelos de regressão linear simples e validar a escolha do Random Forest, algoritmo que, por natureza, é adaptado a distribuições não-normais e assimetrias severas.
3.1.4 Random Forest


        Primeiramente, o modelo de machine learning empregado neste estudo é o Random Forest. A escolha por esta arquitetura, detalhada conceitualmente no Capítulo 2.2.5, fundamenta-se nas características estatísticas complexas do dataset, que exigem um algoritmo robusto a distribuições não-normais e valores extremos.
        Ademais, três características do dataset ajudaram a sustentar esse modelo como o único ideal para a análise e o descarte de outros como Regressão Linear Múltipla, Algoritmos Baseados em Distância, Redes Neurais Artificiais e Boosting que são a alta assimetria, grande número de valores extremos e característica de zero inflados.
A análise exploratória do dataset revelou dados extremamente heterogêneos, caracterizados por uma assimetria positiva severa, pela análise da distribuição estatística dos dados, a variável alvo do modelo, ‘quantidade_acidentes’ (agrupada em buckets espaciais de 10 km por mês), demonstrou uma média de 2,22 acidentes por trecho, enquanto isso a mediana ficou em apenas 1,0. Essa diferença mostra uma distribuição assimétrica confirmada pelo índice de Skewness de 5,6871. Comportamento também é visto em outras variáveis do dataset como de feridos apresentou um Skewness de 10,36 e mortos de 9,68.
A seguir a Tabela 3, em que demonstra que os dados não seguem uma distribuição normal e sim uma cauda longa. Além disso, como a mediana é menor que a média, sabemos que em muitos casos, nada aconteceu, não houve nenhuma morte, caracterizando um cenário de zero inflados com a média sendo “puxada” pelos valores extremos.
Média: Representa o valor aritmético médio das ocorrências.
Mediana: Representa o valor central exato da distribuição, separando os 50% menores dos 50% maiores. 
Skewness: Métrica que mostra a inclinação da distribuição dos dados. 
Kurtosis: Mede o peso das caudas e o grau de achatamento da distribuição em relação à Curva de Gauss (que possui curtose = 3). 
% de Outliers (IQR): Indica a proporção de registros que ultrapassam o limite superior do Intervalo Interquartil. 
Tabela 3 - Análise distribuição estatística dos dados

Variável
	Média
	Mediana
	Skewness
	Kurtosis
	% Outlier IQR
	mortos
	0.083
	0.0
	9.6802
	432.9515
	7.12
	feridos_graves
	0.276
	0.0
	5.8208
	130.3877
	22.44
	feridos_leves
	0.868
	1.0
	9.0764
	285.6457
	4.32
	feridos
	1.144
	1.0
	10.3618
	312.7316
	44.15
	veiculos
	1.976
	2.0
	6.9958
	477.6257
	7.0
	

Fonte: Elaborado pelo autor (2026).


        Com isso, é imperativo analisar em detalhes a variável alvo do modelo ‘quantidade_acidentes’. Para tal objetivo foi construído a figura 5 com um painel analítico triplo.
Figura 5 - Distribuição e Cauda da Variável Alvo
  

Fonte: Elaborado pelo autor (2026).
A Figura 5 foi construída para analisar a assimetria, dividindo a análise em três perspectivas complementares:


Histograma + KDE (Kernel Density Estimate):


        O primeiro gráfico ilustra o histograma de distribuição, para mostrar a frequência de ocorrência, já o KDE é a linha suave que passa por cima dele, estimando a função de densidade de probabilidade. Dessa forma mostra a curva que a base de dados possui. O eixo horizontal (X) representa a quantidade de acidentes em um trecho/mês, enquanto o eixo vertical (Y) mostra quantas vezes esse cenário ocorreu na base histórica.
        É possível observar uma clara longa cauda para a direita, isso prova que na maioria dos buckets acontecem apenas 1 acidentes (Mediana = 1), entretanto a linha tracejada vermelha (Média = 2.2) está deslocada para a direita, sendo “puxada” pelos valores extremos.


A Função de Cauda (CCDF - Complementary Cumulative Distribution Function):


        O gráfico central mostra a probabilidade de ocorrer um determinado número de acidentes. Para visualizar eventos raros, o eixo vertical (Y) foi colocado em escala logarítmica (de 10^0 a 10^-5).
         Caso os dados não tivessem uma cauda pesada, a linha azul despencaria verticalmente para zero logo após os primeiros números, pois números maiores não existiriam. Em vez disso, observamos uma escada que se estende longamente para a direita. 
Analisando os percentis da distribuição, temos que o P95 é igual a 7, apenas 5% dos trechos têm mais de 7 acidentes no trecho, mas o P99.9 chega a 28 acidentes, e a cauda se arrasta ainda até o máximo de 64 acidentes. Este gráfico prova que eventos de altíssima sinistralidade, embora raros, são persistentes e estruturais.






Q-Q Plot vs. Normal (Quantile-Quantile Plot):


        O gráfico da direita é uma ferramenta de diagnóstico estatístico, em que ele plota os quantis dos dados reais (pontos azuis) contra os quantis teóricos de uma distribuição Normal/Gaussiana perfeita (linha vermelha diagonal). Se os dados seguissem uma distribuição normal, os pontos azuis seguiam perfeitamente a linha vermelha.
        Com isso, percebe-se que a premissa de normalidade não é atendida, o que invalida o uso de abordagens paramétricas tradicionais, como a Regressão Linear Múltipla. Pois a mesma pressupõe que os resíduos do modelo seguem uma distribuição normal, premissa diretamente refutada pelo Q-Q Plot apresentado na Figura 5, que demonstra o severo desvio da normalidade nos dados de sinistros. A violação desta premissa inválida os intervalos de confiança e os testes de significância do modelo, comprometendo a validade estatística de qualquer inferência gerada, além de que a manutenção dos outliers estruturais 'puxaria' a reta de regressão, forçando o algoritmo a superestimar de forma irrealista o número de acidentes em dias normais e vias seguras (JAMES et al., 2023). 
De forma muito parecida, Redes Neurais Artificiais (RNAs), embora sejam modelos não-lineares, exigem a normalização das variáveis, transformação na mesma escala, para a convergência dos gradientes, conflitando diretamente com o objetivo central deste trabalho, uma vez que a normalização da variável alvo inviabiliza a geração de previsões diretamente interpretáveis na escala real de acidentes para consumo. Adicionalmente, RNAs sofrem severa instabilidade diante de outliers extremos não tratados (GOODFELLOW; BENGIO; COURVILLE, 2016). Uma saída comum na literatura seria eliminar esses outliers, entretanto, neste domínio, os valores extremos não são erros de medição, mas sim a realidade real deste complexo cenário.
        Para algoritmos baseados em distância, como K-Nearest Neighbors (KNN) e Support Vector Machines (SVM), a topologia dos dados também as desqualifica. Por calcularem distâncias geométricas entre os pontos, esses algoritmos são fundamentalmente sensíveis à escala das variáveis, variáveis em escalas maiores dominam o cálculo da distância, distorcendo a avaliação de similaridade entre as observações (JAMES et al., 2013), exigindo obrigatoriamente a normalização dos dados de entrada. Tal exigência conflita diretamente com o objetivo central deste trabalho, pois normalizar as variáveis tornaria as previsões não interpretáveis na escala real, inviabilizando o consumo direto dos resultados nos dashboards sem transformação inversa, introduzindo uma dependência operacional e adicionando uma complexidade fora do padrão para o objetivo do projeto.
Ademais, a presença confirmada da cauda longa afasta a viabilidade de algoritmos de Gradient Boosting. A razão principal reside em que modelos de Boosting constroem árvores rasas de forma sequencial, onde cada nova árvore tenta corrigir os resíduos da árvore anterior (HASTIE; TIBSHIRANI; FRIEDMAN, 2009), exigindo calibração extensiva de hiperparâmetros para lidar com a alta variância introduzida pelos valores extremos. Por fim, sua natureza sequencial amplifica o impacto dos resíduos extremos em distribuições de cauda longa, onde os poucos pontos de altíssima sinistralidade dominam o processo de otimização do gradiente a cada iteração, comprometendo a capacidade de generalização do modelo para os trechos de sinistralidade rotineira que compõem a maioria da base.
        Portanto, faz-se necessário analisar especificamente os benefícios inerentes da escolha da arquitetura do Random Forest. O argumento central da sua escolha reside na invariância à escala, por operar através de divisões binárias baseadas em comparações de valor, o algoritmo é matematicamente insensível a transformações de escala das variáveis, eliminando a necessidade de normalização da variável alvo e permitindo que as previsões sejam geradas e exportadas diretamente na escala real de acidentes para consumo, satisfazendo o RF-1 do projeto (JAMES et al., 2023). 
Em segundo lugar, tal invariância à escala implica diretamente na robustez a outliers estruturais, preservando os valores extremos que representam a realidade crítica do cenário viário, satisfazendo o RNF-2. Em terceiro lugar, o uso do Bagging e da 'sabedoria da multidão' (BREIMAN, 2001) garante resistência ao overfitting e maior capacidade de generalização do modelo. Por último, o Random Forest não tem como base relações lineares entre as variáveis, permitindo a modelagem de relações não lineares e a captura automática de interações complexas entre os atributos, conforme abordado nos fundamentos teóricos dos Capítulos 2.2.3 e 2.2.5. Portanto, essa soma de características técnicas e o projeto é focado em análises preditivas e não na comparação de modelos, o Random Forest é modelo ideal para reduzir a incerteza estatística no cenário complexo e não-normal dos sinistros viários, oferecendo uma relação custo benefício ótima para o objetivo.
3.1.5 PowerBI

        Para a construção de dashboards para apoiar a análise exploratória e visualização dos resultados foi utilizado o PowerBI, que é uma plataforma de inteligência de negócios desenvolvida pela Microsoft (2026), ele foi incorporado à stack tecnológica do projeto para operacionalizar a camada de visualização e tomada de decisão. Ferramenta de mercado consolidada, seu papel no pipeline vai além da mera geração de relatórios estáticos, ele funciona como a ponte de consumo entre os modelos preditivos complexos desenvolvidos em Python e os gestores públicos da PRF e do DNIT, que são os profissionais que demandam respostas rápidas, mas não necessariamente dominam linhas de código.


A escolha da plataforma justifica-se por três fatores práticos:


Em primeiro lugar, pela integração nativa com arquivos estruturados `.csv` e `.parquet` exportados diretamente das rotinas de modelagem. O modelo utilizado o Random Forest gera estimativas na escala original dos dados, eliminando a necessidade de transformações inversas e permitindo que o Power BI consuma a variável resposta diretamente. 
Em segundo lugar, a flexibilidade da linguagem DAX, Data Analysis Expressions, viabiliza o cálculo de medidas agregadas em tempo real. É possível calcular taxas de severidade, variações percentuais e médias móveis espaciais à medida que o usuário interage com os filtros de tela. 
Por fim, a capacidade de renderização geoespacial da ferramenta permite mapear os 139.590 buckets de 10 km ao longo da malha rodoviária federal. O resultado é uma interface interativa dividida em painéis temáticos: diagnósticos históricos descritivos, mapas de calor de risco espacial, matrizes de simulação de cenários operacionais e rankings de priorização de investimento em infraestrutura para o DNIT.


