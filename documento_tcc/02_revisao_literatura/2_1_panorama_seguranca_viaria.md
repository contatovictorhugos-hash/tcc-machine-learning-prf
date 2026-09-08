2 REVISÃO DA LITERATURA


Este capítulo apresenta o embasamento teórico necessário para a compreensão dos acidentes de trânsito no Brasil e as tecnologias de Ciência de Dados aplicadas neste presente projeto. A fundamentação está baseada na análise histórica da segurança do trânsito nacional e na evolução das técnicas de inteligência artificial voltadas à modelagem preditiva.
2.1 Panorama da segurança viária no Brasil


A segurança de trânsito no Brasil é um campo crítico que envolve variáveis humanas, veiculares e de infraestrutura, apesar de ser um tema extremamente importante para milhões de brasileiros, o panorama atual da malha rodoviária brasileira é alarmante. Segundo a 27ª edição da pesquisa CNT de Rodovias o país dispõe de cerca de 1,5 milhão de quilômetros de estradas e vias, mas apenas 12,4% desse total é pavimentado, e da minoria que é pavimentada 67,5% da extensão avaliada foi classificada como “regular”, “ruim” ou “péssima”, e apenas 32,5% como “boa” ou “ótima”.
Além disso, o país está ficando para trás na já velha corrida da pavimentação das estradas, em uma análise da cobertura de estradas pavimentadas em comparação com a área total do país, o Brasil conta com 25,1 km pavimentados por mil km² de área, países como China e Estados Unidos possuem 447 km e 437 km de estradas pavimentadas por mil km² de área, respectivamente. Já em comparação com países adjacentes na América latina, o país detém um índice menor do que os verificados no Equador (31,4), na Argentina (42,3) e no Uruguai (43,9), como exemplificado na Figura 1.
Figura 1 - Densidade da malha rodoviária pavimentada por país (valores em km/mil km²) 
  

Fonte: Elaboração CNT, com dados de The CIA World Factbook (2024), Ministério da Infraestrutura (2020) e IBGE (2022).
A persistência desse cenário de insegurança rodoviária está alicerçada em um descompasso do tamanho da malha rodoviária em comparação com o número de motoristas, pois de 2013 a 2023, houve um aumento de modestos 2,2% das rodovias pavimentadas (CNT, 2024), ao contrário do número de motoristas em que no mesmo período foi registrado um crescimento de 46,1% no número total de veículos, sendo que a região Norte registrou a maior alta, com 70,1%, e é a região com o menor índice de km pavimentados por mil km² de área.
        Tais problemas afetam a competitividade econômica do país, o estado crítico das rodovias encarece o transporte ao elevar custos com reparos mecânicos, combustível e atrasos. Esse cenário gera gargalos na distribuição que afetam a eficiência de toda a produção, reduzindo a competitividade do país e intensificando o impacto negativo do Custo Brasil. Ademais, também é possível destacar o desperdício de insumos agrícolas, notadamente grãos, durante o deslocamento rodoviário das zonas produtoras aos portos comerciais, tendo na precariedade da malha viária um de seus fatores determinantes..
Pois o Brasil é essencialmente rodoviário, sendo responsável por aproximadamente 65% do transporte de cargas escoadas e por quase 95% dos passageiros, conforme Boletim Estatístico CNT (2024), consumindo cerca de 15,5% do PIB nacional (ILOS, 2026), muito acima da média dos países desenvolvidos com 8% do PIB nacional.
Tal dependência é exemplificada pelas suas vulnerabilidades que ficou provada em dados após a paralisação de 2018 onde durante a greve dos caminhoneiros, a produção industrial do Brasil caiu 10,9% em apenas um mês, maio de 2018 (IBGE, 2018), provando é o Brasil é extremamente dependente desse sistema falho economicamente que ceifa milhares de vidas todos os anos.
Ademais, a falta de infraestrutura não é apenas uma questão econômica que freia o crescimento do Brasil, é também uma questão de saúde pública, de interesse de milhões de brasileiros. Tendo em vista que, conforme o Anuário da Polícia Rodoviária Federal (2025), o ano de 2024 foi o mais letal desde 2017, em análise da série histórica, houve 6160 mortos e 73.156 acidentes (média de 200 acidentes e 16 mortes por dia), como é mostrado na Figura 2.












Figura 2 – Evolução anual dos sinistros, feridos e mortos em rodovias federais brasileiras (2007–2024).
  

Fonte: Anuário da Polícia Rodoviária Federal (2025).
Além de extremamente letais, as rodovias federais brasileiras apresentam elevados índices de acidentes associados a fatores estruturais passíveis de intervenção, como falta de investimento, planejamento e duplicação de vias. Conforme dados do Anuário Estatístico da Polícia Rodoviária Federal (2025) exemplificados na Figura 3, a maior parte dos acidentes e das mortes ocorre em trechos de pista simples, que registraram 35.287 sinistros e 4.291 mortes no período analisado, superando significativamente os números observados em rodovias de pista dupla ou múltipla. Esses números evidenciam que a predominância de vias não duplicadas e com menor padrão de infraestrutura contribui diretamente para o agravamento da sinistralidade nas rodovias federais brasileiras, reforçando a necessidade de políticas públicas orientadas por dados para melhoria da segurança viária.
Figura 3 – Distribuição de sinistros, feridos e mortos por tipo de pista em rodovias federais brasileiras.  
Fonte: Anuário da Polícia Rodoviária Federal (2025).
Diante do impacto comprovado da infraestrutura viária na ocorrência de sinistros, a utilização de técnicas de Ciência de Dados apresenta-se como ferramenta estratégica para apoiar decisões de investimento público. A partir da integração de bases abertas, como os registros da Polícia Rodoviária Federal, dados de condição da malha do Departamento Nacional de Infraestrutura de Transportes e levantamentos da Confederação Nacional do Transporte, é possível identificar padrões espaciais e temporais de acidentes, estimar o risco associado a diferentes características da via e priorizar trechos onde intervenções como duplicação, melhoria do pavimento ou reforço da sinalização produzirão maior redução de sinistros. Modelos preditivos e análises geoespaciais permitem, por exemplo, calcular a probabilidade de acidentes em função de variáveis como tipo de pista, volume de tráfego, geometria da via e histórico de ocorrências, fornecendo aos órgãos públicos evidências quantitativas para otimizar recursos escassos. Assim, a Ciência de Dados contribui para uma gestão proativa da segurança viária, transformando grandes volumes de dados operacionais em insights capazes de orientar políticas públicas mais eficazes e reduzir a mortalidade nas rodovias federais brasileiras.
