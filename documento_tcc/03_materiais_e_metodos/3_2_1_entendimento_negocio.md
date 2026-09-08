3.2 Métodos


Esta seção do projeto apresenta as metodologias empregadas no desenvolvimento do estudo, baseia-se nas 6 fases do framework CRISP-DM, entendimento do negócio, entendimento dos dados, preparação dos dados, modelagem, avaliação e por fim a implantação.
3.2.1 Entendimento de negócio 


A primeira fase da metodologia CRISP-DM, o Entendimento de Negócio, transcende a mera catalogação de requisitos para situar-se no cerne da Engenharia de Software, não apenas a entrega de uma lista de to-do`s que o sistema deve fazer mas sim a construção de uma ponte entre a dor do usuário, nesse caso a sociedade, e a viabilidade tecnológica. Com o projeto situado no contexto da administração pública, as dores não são apenas do usuário e sim de toda a sociedade, que sofre com inúmeros problemas já apresentados. Portanto, o sucesso desta etapa não é medido pela quantidade de funções documentadas, mas pela precisão com que os objetivos de governança são traduzidos em requisitos de software que, efetivamente, geram valor público e suportam a tomada de decisão orientada a dados.
A seguir o diagrama que detalha os passos realizados durante o desenvolvimento do projeto:
Figura 6 - Diagrama do Projeto
  
Fonte: Elaborado pelo autor (2026).

3.2.1.1 Identificação dos stakeholders


Primeiramente, no contexto da Engenharia de Software, o termo stakeholders se refere a todas as pessoas, grupos ou instituições que possuem interesse direto ou indireto no sistema desenvolvido. Com isso, a identificação desses personagens é muito importante para compreender as necessidades de cada uma das personas. Um ponto extremamente relevante em projetos de ciência de dados aplicados ao setor público é que essa etapa assume papel ainda mais crucial, pois as decisões feitas a partir dos modelos de dados impactam não apenas organizações específicas, mas também a sociedade como um todo, tendo um impacto na casa dos milhões.
Os stakeholders institucionais, representam a persona governamental do projeto, é responsável pelo planejamento estratégico, formulação de políticas públicas e alocação de orçamento. Neste projeto, temos dois, Polícia Rodoviária Federal e o Departamento Nacional de Infraestrutura de Transportes, que inclusive são as duas fontes de dados do estudo. Para eles, o modelo preditivo e as análises realizadas servem como uma ferramenta de Inteligência Estratégica, trazendo base e justificando matematicamente onde os investimentos em infraestrutura (recapeamento, duplicação) trarão o maior Retorno sobre o Investimento (ROI) em termos de vidas salvas e redução de acidentes.
Os stakeholders operacionais, representam os profissionais que atuam na ponta, ou literalmente no asfalto, e que consumirão os insights do modelo no dia a dia através de relatórios. Neste projeto estão incluídos os agentes da PRF, as equipes de resgate médico e as equipes de manutenção viária, para este grupo, a granularidade do modelo é chave. A inteligência preditiva permite abandonar a postura reativa e passar a realizar o patrulhamento preventivo, posicionando viaturas e radares móveis nos trechos de maior risco probabilístico antes que os sinistros ocorram.
Os stakeholders sociais, representam os beneficiários finais do projeto, a sociedade brasileira, que inclui os motoristas, os passageiros e o setor de transporte de cargas. De forma indireta, o Sistema Único de Saúde (SUS), que sofre com a superlotação de leitos, e a economia nacional, que amarga perdas anuais na casa dos R$50 bilhões devido ao famoso "Custo Brasil" gerado por acidentes, como apresentado na seção 2.1. Para a sociedade, o sucesso deste modelo preditivo são a preservação da vida, no aumento da segurança viária e na eficiência logística do país.



3.2.1.2 Requisitos Funcionais e Não Funcionais


        Após a tradução das necessidades da administração pública em um problema estruturado, é necessário a definição do escopo técnico do projeto. Sob a ótica da Engenharia de Software, esta etapa consolida a criação do contrato do desenvolvimento com os stakeholders, através da elicitação de requisitos. Os Requisitos Funcionais (RF) delimitarão as ações, transformações e processamentos que o pipeline de dados deve executar para gerar inteligência estratégica. Em contrapartida, os Requisitos Não Funcionais (RNF) atuarão como as restrições arquiteturais e metodológicas inegociáveis do projeto, garantindo que o rigor científico se aplicada na metodologia e a integridade estatística dos dados rodoviários sejam preservados ao longo de todo o ciclo de vida do modelo preditivo, para que seja assegurada um alto nível de certeza nos resultados do modelo.


Requisitos funcionais: 


* Ingestão e Integração Multi-fonte: O sistema deve ser capaz de extrair dados históricos da PRF (2019-2025) e do DNIT (2019-2025) realizar a junção intervalar, alocando o quilômetro exato do acidente dentro do intervalo [km_inicial, km_final] da rodovia correspondente.


* Transformação e Engenharia de Atributos: O sistema deve ser capaz de realizar a limpeza, tratamento dos dados e a criação de novas variáveis temporárias e binárias.


* Exploração dos dados: O sistema deve ser capaz de explorar os dados para gerar gráficos e insights que auxiliem na tomada de decisões estratégicas.


* Treinamento do modelo preditivo: O sistema deve instanciar e treinar um algoritmo Random Forest Regressor para estimar a quantidade de acidentes por trecho/mês.


* Simulação de cenários: O sistema deve ser capaz de realizar a simulação de predições alterando variáveis exógenas (ex: clima, feriado) e de infraestrutura (ex: qualidade do pavimento).


* Integração com BI: O sistema deve ser capaz de exportar os dados tratados para o PowerBI afim na geração de insights na ferramenta.


* Escala Real: O sistema deve ser capaz de gerar previsões diretamente interpretáveis em escala real para consumo em dashboards por stakeholders não técnicos.


Requisitos não funcionais: 


* Output em escala real: O pipeline de pré-processamento não deve realizar transformações inversas.


* Preservação de outliers: O sistema não deve remover valores extremos, outliers, das variáveis, pois os dados apresentam assimetria extrema representando a realidade crítica do cenário, e não erros de medição.


* Otimização de memória: O sistema deve otimizar os tipos de dados, convertendo float64 para float32 e strings para category, garantindo a execução fluida no ambiente.


* Desempenho mínimo do modelo: O modelo treinado deve atingir um Coeficiente de Determinação (R²) superior a 0.90 e um Erro Médio Absoluto (MAE) inferior a 1.0 acidente por trecho/mês no conjunto de testes.


* Avaliação entre versões: Devem ser gerados pelo menos três versões do modelo Random Forest Regressor com configurações diferentes e comparar o desempenho dos mesmos.






3.2.1.3 Estratégia de Versionamento


        Para o versionamento do código, foi empregada no projeto o Git, com o modelo de ramificação GitFlow, que é um sistema de controle distribuído onde é possível guardar diferentes versões e o histórico do projeto. Esta escolha se justifica pois é necessário manter a integridade do código durante o avanço das 6 etapas do procedimento CRISP-DM, o uso do Git é padrão nos projetos de desenvolvimento de software modernos.
        A ramificação principal main, foi constantemente alimentada com novas versões branchs assim que uma nova etapa do procedimento é concluída. Com isso, como repositório remoto foi escolhido o GitHub, onde no final do projeto será disponibilizado para conferência.


