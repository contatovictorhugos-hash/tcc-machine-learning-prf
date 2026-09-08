4.4 Avaliação do Impacto da Infraestrutura Rodoviária (DNIT)
Relatórios históricos de trânsito costumam concentrar a responsabilidade dos sinistros quase que exclusivamente na conduta do motorista. Excesso de velocidade, ingestão de álcool e desatenção ao volante encabeçam as estatísticas tradicionais. Essa perspectiva, contudo, oculta a contribuição silenciosa da infraestrutura física, a pista dita as condições de aderência, a estabilidade dinâmica do veículo e a margem de correção em situações de emergência, informações que são chave para analisar a complexidade do cenário. Para mensurar a influência direta da malha pavimentada na ocorrência de sinistros, desenvolveu-se um pipeline de interpretabilidade composto pelo diagnóstico de registros sentinela, análise de valores SHAP, Shapley Additive Explanations, simulação, Ceteris Paribus, (Teste A/B) e Gráficos de Dependência Parcial (PDP).
4.4.1 Diagnóstico do Sentinela -1.0 (Vias Não Monitoradas pelo DNIT)
A integração entre os registros de ocorrências da PRF e as campanhas de monitoramento do DNIT expôs uma disparidade na cobertura dos dados. Dos 213.187 registros agregados em intervalos espaciais de 10 km, apenas 37.798 (17,7%) possuem levantamentos contínuos de pavimento e conservação. Os 82,3% restantes da malha federal não contam com dados de inspeção sistemática nas bases oficiais.
        Diante desse cenário, a imputação estatística clássica e mais utilizada que é a de utilizar a média ou mediana distorce a variância real das medições e mascara a heterogeneidade das rodovias. A solução adotada consistiu em atribuir o valor sentinela `-1.0` a todos os segmentos desprovidos de medição do DNIT nas variáveis contínuas: Índice de Condição do Pavimento `icp`, Índice de Condição da Conservação `icc` e Índice de Condição da Manutenção `icm`. 
Árvores de decisão lidam com valores sentinelas de forma categórica. O Random Forest não calcula distâncias euclidianas, ele cria pontos de corte lineares. O algoritmo estabeleceu nós de divisão dedicados exclusivamente à condição `icp <= -0.95`. Dessa forma, os trechos não monitorados foram segregados em um grupo analítico independente, sem interferir na calibração dos índices contínuos (que variam de 0 a 100 na amostra real).
        O modelo aprendeu que a ausência de vistoria atua como uma variável indicativa de gestão. Rodovias sem monitoramento regular pelo DNIT tendem a coincidir com trechos de tráfego secundário ou de baixo investimento em sinalização e conservação. Para esse grupo sentinela `-1.0`, o modelo convergiu para uma previsão base média de 2,42 acidentes/mês por trecho de 10km.


4.4.2 Análise SHAP e Teste A/B Ceteris Paribus (BR-381 KM 340-350 / MG)
A contribuição marginal dos atributos de infraestrutura foi mensurada por meio dos valores SHAP, fundamentados na Teoria dos Jogos Cooperativos. A avaliação foi processada sobre a partição de dados que dispõe de medição real do DNIT. A Figura 20 a seguir ilustra a hierarquia das variáveis de engenharia rodoviária.
        Entre as 15 variáveis originárias do DNIT, o Índice de Condição do Pavimento `icp` exerceu a maior força preditiva, com SHAP médio de 0,02956. Seguem-se o `icc` de 0,01543 e o `icm` de 0,01288. Variáveis operacionais de conservação deficiências no sistema de drenagem de 0,00407, sinalização horizontal desgastada 0,00356 e falta de roçada lateral com 0,00317 apresentaram contribuições individuais complementares, demonstrando que o algoritmo capta diferentes facetas da deterioração física da via.
Figura 20 – Importância média SHAP das variáveis de infraestrutura do DNIT por grupo.
  

        Fonte: Elaborado pelo autor (2026).
Para comprovar a causalidade da qualidade do asfalto de forma isolada, estruturou-se um experimento, Ceteris Paribus também conhecido simplesmente como Teste A/B. Fixaram-se rígidamente todas as variáveis exógenas e comportamentais em um segmento notoriamente crítico da malha nacional: a BR-381, entre os quilômetros 340 e 350, localizado em Minas Gerais. O trecho que apresenta traçado sinuoso classificado como, curva acentuada combinada com declive, e pista simples.
        O experimento congelou o cenário em condições normais: dia útil sem feriado e céu claro, a fim de normalizar as variáveis comportamentais e temporais, para se estudar exclusivamente o impacto da qualidade de infraestrutura. Alterou-se estritamente o valor do `icp` e seus respectivos descritores de conservação:
   1. Via Perfeita ICP = 90, sem trincas, drenagem limpa: 2,22 acidentes/mês;
   2. Via Boa ICP = 70, pavimento regular, sinalização íntegra: 2,21 acidentes/mês;
   3. Via Regular ICP = 50, presença de trincamento isolado: 2,27 acidentes/mês;
   4. Via Ruim ICP = 25, remendos e panelas superficiais: 2,33 acidentes/mês;
   5. Via Péssima ICP = 5, desagregação severa, buracos e drenagem obstruída: 2,38 acidentes/mês.


Os resultados desse experimento comparativo estão consolidados na Figura 21.


Figura 21 — Simulação Ceteris Paribus (Teste A/B) do impacto do pavimento na BR-381 (KM 340–350/MG).
  

Fonte: Elaborado pelo autor (2026).
A degradação completa da superfície eleva a previsão de sinistros em +7,2%, 2,22 para 2,38. Como os fatores humanos, climáticos e de calendário permaneceram imutáveis, a elevação decorre estritamente da resposta mecânica da via. O asfalto esburacado e a sinalização apagada reduzem a margem de manobra do condutor, transformando pequenas falhas operacionais em colisões graves.


4.4.3 Não Linearidade no PDP e Retorno sobre Investimento
Através da pesquisa realizada anteriormente, foi possível inferir que a relação entre o estado de conservação do pavimento e o volume de acidentes não é linear. O Gráfico de Dependência Parcial, conhecido academicamente como Partial Dependence Plot, apresentado na Figura 22 abaixo, descreve o comportamento do modelo ao longo de toda a faixa do ICP de 0 a 100. A curva exibe uma inflexão nítida em formato de cotovelo.






Figura 22 – Gráfico de Dependência Parcial (PDP): impacto isolado do ICP na previsão de acidentes.
  

Fonte: Elaborado pelo autor (2026).
A recuperação de uma rodovia em estado crítico, de ICP = 0, para um nível intermediário de trafegabilidade, ICP = 50, reduz a estimativa em -0,120 acidentes/mês por trecho de 10 km. Em contraste, o investimento adicional para elevar essa mesma pista do padrão regular, ICP = 50, para um patamar de excelência, ICP = 100, proporciona uma redução residual de apenas -0,080 acidentes/mês.
        Esse comportamento reflete a complexa dinâmica do risco viário brasileiro. A eliminação de buracos profundos, depressões e pistas escorregadias remove armadilhas que causam perda imediata de controle veicular. Após restabelecer as condições mínimas de atrito e nivelamento, ganhos marginais de acabamento asfáltico geram impactos proporcionalmente menores na contenção de colisões.
        Esse achado traz implicações diretas para a gestão orçamentária do DNIT e da ANTT. Sob a ótica da destinação dos recursos públicos, a métrica de Retorno sobre o Investimento em Manutenção (ROMI) para preservação de vidas indica que o orçamento deve priorizar a eliminação de defeitos críticos em trechos degradados da malha antes de financiar recapeamentos de padrão elevado em vias que já operam em nível funcional aceitável.




4.4.4 Showcase de Otimização Global Argmax (BR-40 KM 110-120)
Para demonstrar a utilidade prática do modelo na alocação de recursos públicos, executou-se uma varredura exaustiva sobre os 4.072 trechos monitorados de 10 km que compõem o conjunto de dados utilizados no estudo. O objetivo consistiu em identificar em qual coordenada geográfica uma intervenção estrutural de engenharia promoveria a maior redução absoluta no volume de acidentes.
        O algoritmo isolou as interferências sazonais atribuindo a mediana histórica de cada segmento para dias úteis de tempo seco. Em seguida, simulou o ganho decorrente da restauração da pista de um patamar degradado de, ICP = 5, para uma condição pós-obra satisfatória para ICP = 70.
        O ponto de maior sensibilidade à manutenção identificado na malha nacional está localizado na BR-40, no trecho compreendido entre os quilômetros 110 e 120. O segmento caracteriza-se por pista reta com tráfego denso e histórico de pavimentação comprometida. A Figura 23 ilustra a simulação comparativa e o detalhamento do ganho operacional obtido nesse trecho.
Figura 23 — Simulação de intervenção do DNIT no ponto crítico da BR-40 (KM 110–120).
  

Fonte: Elaborado pelo autor (2026).
Os resultados da simulação evidenciam a magnitude do impacto da infraestrutura:


Condição Péssima (ICP = 5): Estimativa de 4,430 acidentes/mês;
Intervenção do DNIT (ICP = 70): A previsão cai drasticamente para 3,675 acidentes/mês.


A intervenção viária proporciona uma redução absoluta de -0,755 acidentes/mês por trecho de 10 km, correspondendo a uma queda relativa imediata de -17,04% na sinistralidade no trecho. Em termos acumulados, essa manutenção estrutural representa aproximadamente 9 acidentes graves a menos por ano em um único segmento de dez quilômetros.
Essa evidência empírica corrobora a tese central da pesquisa, o estado físico da rodovia dita a probabilidade de ocorrência de sinistros. Como todos os fatores comportamentais e meteorológicos foram rigorosamente retirados, a retração de 17% decorre exclusivamente pela melhora do pavimento. O modelo deixa de ser meramente descritivo e assume caráter prescritivo, entregando ao DNIT uma ferramenta orientada por dados para maximizar a eficiência social do gasto público.


