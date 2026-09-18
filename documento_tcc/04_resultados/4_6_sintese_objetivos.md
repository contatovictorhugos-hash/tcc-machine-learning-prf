## 4.6 Síntese dos Resultados Face aos Objetivos da Pesquisa

Os experimentos apresentados ao longo deste capítulo permitem verificar, de forma direta, se o trabalho entregou o que foi proposto no Capítulo 1. Cada objetivo é confrontado a seguir com as evidências empíricas correspondentes.

### 4.6.1 Objetivo Geral

O objetivo geral consistia em analisar o comportamento histórico dos sinistros viários em rodovias federais para desenvolver um modelo preditivo capaz de estimar o volume de acidentes em períodos subsequentes. Esse objetivo foi atingido com a homologação do Modelo V6 Oficial, um Random Forest com critério de Poisson Deviance treinado sobre a grade cartesiana espaço-temporal completa de 475.020 observações (5.655 trechos físicos de 10 km ao longo de 84 meses).

Conforme detalhado na Seção 4.1, o modelo foi submetido a um teste cego estritamente cronológico, reservando o biênio 2024-2025 (135.720 registros) como horizonte de validação fora do período de treino. Os resultados obtidos nesse teste superaram os parâmetros mínimos de aceitação definidos no RNF-4: o coeficiente de determinação $R^2$ alcançou 0,8678 (meta: $R^2 > 0,70$), o erro médio absoluto ficou em 0,6137 acidentes por trecho/mês (meta: MAE $< 1,0$) e a deviance explicada $D^2$ atingiu 0,7964. O diagnóstico de resíduos apresentado na Seção 4.2 reforça essa conclusão: nos segmentos pacificados, com zero acidentes no período, o erro médio recuou para apenas 0,2889 acidentes por mês, o que confirma a capacidade do modelo de discriminar trechos seguros de pontos críticos de sinistralidade.

### 4.6.2 Objetivos Específicos

Os objetivos específicos definidos no Capítulo 1 foram três: (i) pré-processar e integrar as bases da PRF e do DNIT para gerar insights de apoio à gestão pública; (ii) disponibilizar um modelo de machine learning que auxilie na alocação preventiva de recursos nos locais de maior incidência; e (iii) construir dashboards no Power BI para visualização dos resultados e suporte à decisão operacional. Cada um deles é confrontado a seguir com os resultados obtidos.

**a) Pré-processamento e integração multi-fonte (PRF + DNIT)**

O pipeline de ETL processou mais de 450 mil boletins de ocorrência da PRF e os integrou aos levantamentos de condição estrutural do DNIT por meio de junções intervalares espaciais, conforme descrito no Capítulo 3. A construção da grade cartesiana com 70,61% de zeros reais, validada na comparação entre arquiteturas da Seção 4.1, resolveu o viés de sobrevivência que afetava as versões anteriores e garantiu uma representação fidedigna da totalidade da malha federal, incluindo trechos e meses sem nenhum registro de sinistro.

**b) Modelo preditivo e insights para alocação de recursos**

O cumprimento desse objetivo se sustenta em três frentes de resultados apresentadas neste capítulo.

Na Seção 4.4, a aplicação de Gráficos de Dependência Parcial (PDP) e testes contrafactuais *ceteris paribus* expôs a mecânica interna do modelo e identificou dois fenômenos relevantes para a política de manutenção rodoviária: o Efeito Peltzman e o Paradoxo da Severidade. O asfalto em estado péssimo (ICP $\le$ 20) apresentou taxa de mortalidade de 10,50%, frente a 7,32% em pavimento bom, uma letalidade 43,4% superior provocada por invasões da contramão para desviar de buracos e valas. As simulações contrafactuais identificaram na BR-277/PR (km 580 a 590, em Cascavel) a intervenção prioritária nacional para salvamento de vidas, com potencial de evitar cerca de 8 vítimas severas por ano após recapeamento, seguida pela BR-469/PR (km 100) e pela BR-101/BA (km 500). Esses resultados fornecem ao DNIT uma fila técnica de priorização baseada em preservação da vida, e não em critérios exclusivamente econômicos de desgaste veicular.

Na Seção 4.5, as simulações operacionais comprovaram a resolução dinâmica do modelo ao longo da malha federal. O algoritmo estimou 0,09 acidentes por mês para o trecho seguro da BR-317/AC (km 0 a 10) e 29,35 acidentes por mês para o gargalo metropolitano da BR-101/RJ (km 310 a 320), sem sofrer a compressão de variância típica do erro quadrático. O mapeamento dos cinco maiores hotspots federais (liderados pela BR-101/BA em Feira de Santana com 56,42 acidentes reais por mês) e a identificação da oscilação sazonal de +28% entre fevereiro e dezembro fornecem subsídios concretos para a programação preventiva da PRF, permitindo antecipar o reforço de plantões e o posicionamento de viaturas nos períodos e trechos de maior concentração de risco.

Ainda na Seção 4.3, a análise de importância das variáveis confirmou que o histórico recente do trecho (lags temporais e média móvel) concentra 91,46% do poder preditivo do modelo, seguido pela localização regional e pelos índices de condição física do pavimento (ICP, ICC e ICM), que juntos respondem por 2,24%. Esse achado orienta diretamente a priorização de coleta de dados pelo poder público: a manutenção do registro contínuo de ocorrências pela PRF e a ampliação da cobertura de vistorias pelo DNIT são condições de entrada para que o sistema preditivo opere com máxima precisão.

**c) Disponibilização em dashboards de Business Intelligence**

Em conformidade com os requisitos RF-7, RNF-1 e RNF-2, as predições do Modelo V6 foram geradas na escala original das contagens, sem transformações que distorcessem a interpretação policial e sem poda dos valores extremos de alta gravidade. Esses dados foram exportados para dashboards interativos no Microsoft Power BI, que entregam aos gestores públicos mapas de criticidade viária, simuladores de cenários de tráfego e rankings de intervenção asfáltica. A plataforma materializa o terceiro objetivo específico ao disponibilizar os resultados do modelo e da análise descritiva em formato visual e acessível, voltado ao suporte direto de decisões de planejamento operacional e investimento preventivo.
