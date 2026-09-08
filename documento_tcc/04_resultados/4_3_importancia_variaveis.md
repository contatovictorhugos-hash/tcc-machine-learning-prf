4.3 Análise de Importância das Variáveis


A Figura 19 logo abaixo apresenta o gráfico das vinte variáveis de maior importância preditiva no Modelo V5, extraído por meio do atributo nativo feature_importances_ do Random Forest, que quantifica a contribuição média de cada variável na redução da impureza das folhas ao longo de todas as árvores da floresta.








Figura 19 – Top 20 variáveis por importância preditiva no Modelo V5.
  

Fonte: Elaborado pelo autor (2026).
        Em análise do gráfico revela uma hierarquia de importância com implicações estratégicas diretas para os órgãos responsáveis pela segurança viária. A variável is_tracado_via_Reta é a que domina o modelo com aproximadamente 35% de importância total, resultado que, embora à primeira vista possa parecer contraintuitivo, é factível devido os achados da análise exploratória apresentada na Figura 13, que demonstrou que 58,4% dos acidentes ocorrem em trechos de pista reta. Esse domínio confirma que a extensão e a representatividade dos trechos em reta na malha rodoviária federal, combinadas com o comportamento de excesso de velocidade que esses trechos tendem a induzir nos condutores, fazem do traçado reto o principal preditor estrutural do volume de acidentes.
Ademais, a variável is_final_semana surge como a segunda mais importante para a previsão do modelo, com aproximadamente 21% de contribuição, evidência quantitativa que corrobora os padrões temporais identificados na análise exploratória da Figura 15 e que valida diretamente a estratégia metodológica de engenharia de atributos aplicada no ETL de dados. O peso expressivo dessa variável no modelo possui implicação operacional imediata, o reforço preventivo do patrulhamento e da disponibilidade de equipes de resgate nas sextas-feiras, sábados e domingos é a intervenção de maior impacto estatístico que a PRF pode implementar com base nos resultados deste modelo.
Já a variável is_tracado_via_Curva, com importância de aproximadamente 10%, confirma o risco desproporcional das curvas em relação à sua extensão na malha, já apontado na análise exploratória. A sequência seguinte, composta pelas variáveis is_tracado_via_Intersecao_de_Vias, is_tracado_via_Declive e is_tracado_via_Aclive, cada uma com importância entre 3% e 7%, reforça que a geometria da via é o conjunto de variáveis de maior poder explicativo do modelo, respondendo conjuntamente por mais de 60% da importância total. Esse resultado possui valor estratégico direto para o DNIT, pois hierarquiza objetivamente os tipos de intervenção de infraestrutura com maior potencial de redução de sinistros, mostrando que as retas tem uma importância devido a sua quantidade na malha e ao comportamento do motorista, já em outras geometrias menos comuns a periculosidade explode, explicando a sua importância para o modelo.
A variável de qual BR o sinistro ocorreu, sendo o identificador da rodovia federal, aparece com importância de aproximadamente 3%, capturando as diferenças estruturais de volume de tráfego, qualidade de infraestrutura e perfil de uso entre as diferentes rodovias da malha federal, já mostrados na análise exploratória. Os índices de condição do pavimento provenientes da base do DNIT, representados pelas variáveis icc, icp e icm, aparecem com importâncias individuais entre 1% e 2%, resultado que, embora modesto em magnitude absoluta, possui relevância metodológica significativa. A presença dessas variáveis entre as vinte mais importantes valida a estratégia de integração multi-fonte adotada no pipeline de ETL e confirma que a condição do pavimento contribui de forma independente e mensurável para a previsão do volume de acidentes, mesmo após controlar pelos efeitos dominantes do traçado e da sazonalidade temporal.
A variável is_feriado, com importância de aproximadamente 1%, apresenta contribuição modesta em magnitude mas consistente com o padrão identificado na Figura 16, que evidenciou maior média de feridos em dias feriados. A baixa correlação linear dessa variável com as demais, discutida no contexto da Figura 17, é exatamente compensada pelo mecanismo de Feature Subspace do Random Forest, que captura o efeito condicional e não-linear dos feriados sobre a sinistralidade de forma que nenhum modelo paramétrico linear seria capaz de detectar. Por fim, as variáveis de tipo de pista, tipo_pista_Simples, tipo_pista_Dupla e tipo_pista_Múltipla, e de uso do solo completam o ranking, confirmando a relevância das características estruturais da via na previsão do volume de sinistros.


