1 INTRODUÇÃO




Os acidentes de trânsito em rodovias federais brasileiras são um problema histórico e de grande magnitude para a saúde pública e para as finanças do governo. De acordo com dados extraídos do Anuário da Polícia Rodoviária Federal (POLÍCIA RODOVIÁRIA FEDERAL, 2025), foram contabilizados 73.156 sinistros nas rodovias federais no ano de 2024, resultando em 6.160 mortes e 84.526 feridos. Esses números evidenciam não apenas o impacto humano, mas também a pressão sobre os serviços de emergência e o sistema de saúde pública.
Além do alto custo humano evidenciado, os acidentes de trânsito causam impactos econômicos expressivos para a sociedade brasileira. Uma pesquisa realizada pelo Instituto de Pesquisa Econômica Aplicada estima que os acidentes de trânsito no Brasil geram perdas anuais em torno de R$50 bilhões, considerando despesas médicas, custos hospitalares, perda de produtividade, danos materiais e gastos administrativos relacionados ao atendimento e investigação dos sinistros (IPEA, 2020). Em análise mais detalhada sobre rodovias federais, um levantamento da Confederação Nacional do Transporte aponta que os acidentes nessas vias geraram um custo aproximado de R$12,9 bilhões em 2022, reforçando o impacto financeiro da falta de segurança viária para o orçamento público e para a economia nacional (CNT, 2023). Esses valores gritantes evidenciam que a redução dos acidentes não representa apenas uma questão de preservação da vida, mas também uma estratégia de eficiência econômica e sustentabilidade das políticas públicas.
        Atualmente o setor público brasileiro flutua hoje sobre um oceano de dados que cresce a cada dia, gerado pelos órgãos estatais. A transformação digital no setor público surge como solução para transformar dados em informações estratégicas. Apesar disso ainda esbarra em uma cultura de fragmentação e em processos estritamente analógicos. Como aponta o Laboratório de Inovação e Dados (ÍRIS, 2022), a administração governamental ainda enfrenta o desafio de ultrapassar a barreira da burocracia documental para atingir a maturidade analítica, agindo frequentemente apenas quando o incêndio já começou. Essa postura reativa impede que o Estado alcance a maturidade analítica necessária para converter dados brutos em inteligência estratégica. Portanto, mais do que uma atualização tecnológica, a superação desse cenário exige uma ruptura com o modelo descritivo em favor de uma cultura orientada a dados, capaz de explorar o potencial preditivo da máquina estatal.
1.1 Motivação


Diante da gravidade desse cenário e da complexidade dos fatores que influenciam os acidentes de trânsito, passar de uma análise meramente descritiva para uma abordagem preditiva estatística torna-se imperativo. O volume crescente de dados abertos disponibilizados por órgãos como a Polícia Rodoviária Federal possibilita que a Ciência de Dados avance no diagnóstico retrospectivo para a antecipação de riscos e planejamento preventivo, evitando o incêndio repentino. Pesquisas na área de engenharia de transportes demonstram que modelos de previsão são utilizados para avaliar a segurança de trechos rodoviários, identificar locais críticos e apoiar o planejamento de intervenções e investimentos em infraestrutura, contribuindo diretamente para a administração estratégica da segurança viária (BOFFO, 2011). Nesse cenário, surge a pergunta central deste estudo: como a aplicação de arquiteturas de Machine Learning em dados históricos pode embasar o planejamento operacional e a alocação preventiva de recursos, reduzindo a ocorrência e a gravidade dos acidentes em rodovias federais brasileiras?
A transição para uma abordagem preditiva do tema, proposta neste estudo, representa uma mudança clara de patamar e fundamenta-se na cultura Data-Driven, que consiste em utilizar o dado como o ativo central para a estratégia organizacional. Ao utilizar o modelo Random Forest para estimar o volume de acidentes e identificar trechos críticos antes que os acidentes ocorram, o sistema deixa de ser apenas um baú de dados e torna-se uma ferramenta de antecipação operacional. Para o Poder Público, essa migração do reativo para o preditivo significa a capacidade de alocar viaturas, equipes médicas e recursos de manutenção de forma orientada a dados, reduzindo gastos, devido à eficácia do investimento, e salvando valiosas vidas.


1.2 Objetivo Geral


O objetivo geral deste Trabalho de Conclusão de Curso consiste em analisar o comportamento histórico dos sinistros viários em rodovias federais brasileiras para o desenvolvimento de um modelo preditivo robusto fundamentado em arquiteturas de machine learning. A pesquisa busca estimar o volume de acidentes para períodos subsequentes, visando validar a eficácia de modelos de alta performance na redução da incerteza estatística inerente aos microdados governamentais, que muitas vezes parecem peças de um quebra-cabeça espalhadas, demonstrando a aplicabilidade da Ciência de Dados na transformação de dados esparsos em um mapa estratégico navegável.
1.3 Objetivo Específico


Como objetivos específicos, temos a disponibilização de um modelo de machine learning e insights para auxiliar o poder público no investimento assertivo e  a alocação de recursos preventivos nos locais com maior número de incidentes como meio para diminuir a quantidade de vítimas. Para isso ocorre o pré-processamento extenso da base da Polícia Rodoviária Federal (2019-2025) por meio de técnicas de transformação de dados e integração multi-fonte, através do enriquecimento com a base do Departamento Nacional de Infraestrutura de Transportes. Ademais, pretende-se construir dashboards no Power BI para a visualização dos resultados do modelo e da análise descritiva com o objetivo de fornecer insights estratégicos voltados ao suporte à decisão no planejamento operacional e na alocação preventiva de recursos. 
1.4 Metodologia


A metodologia deste estudo fundamenta-se no framework CRISP-DM, Cross Industry Standard Process for Data Mining, que servirá de norte para todos os processos executados, estruturado em um pipeline de Ciência de Dados que integra processos de ETL, Extraction, Transformation, Loading. A fase de preparação dos dados inicia-se com a extração de registros brutos dos portais de dados abertos da PRF e do DNIT (2019-2025). Na etapa de transformação, procede-se à limpeza dos dados, tratamento de valores nulos e engenharia de atributos, visando converter observações brutas em informação estruturada.
        A etapa de modelagem está fortemente ligada à aplicação do algoritmo Random Forest, utilizando a técnica de validação cruzada para assegurar o desempenho e a capacidade de generalização do modelo. Por fim, na fase de implantação, os dados processados e as predições são integrados ao Power BI, resultando em dashboards estratégicos que permitem a visualização intuitiva e o suporte à decisão operacional, materializando o conceito de gestão Data-Driven no setor público.
        Portanto, a pesquisa possui uma finalidade exploratória, pois na fase de análise exploratória do framework que orienta o trabalho, busca compreender a relação das variáveis do dataset, dinâmica temporal dos sinistros, identificar os meses de maior incidência e detectar padrões sazonais de acidentes. Por fim, o estudo se torna explicativo ao desenvolver dashboards para fornecer insights estratégicos dos dados, e preditivo com a realização do modelo de dados, ao realizar estimativas do número de acidentes.
1.5 Organização do Texto


EM CONSTRUÇÃO
        No capítulo 2 será apresentado a revisão de literatura ...
O capítulo 3 visa demonstrar ....
O capítulo 4 tem como objetivo ....
O capítulo 5 apresenta...
O capítulo 6 contém ...
O capítulo 8 tem como conteúdo as conclusões...
Por fim temos um capítulo reservado para as referências bibliográficas.
________________


