2.4 Conceitos de engenharia de software


A Engenharia de Software é mais que o mero ato de `codar`, sendo uma disciplina que aplica princípios de engenharia, métodos e boas práticas para o desenvolvimento, manutenção de sistemas complexos de forma sistemática. O surgimento desta área, está historicamente datado pela conferência da OTAN em 1968, ocorreu como uma resposta estrutural à chamada "crise do software", um período em que a complexidade dos sistemas superava a capacidade humana de gerenciá-los pois não existia uma base metodológica sólida, para servir de base para a expansão dos sistemas. Sob a perspectiva de Sommerville (2011), a engenharia de software profissional não se limita apenas aos aspectos técnicos da produção, mas envolve a gestão de projetos, o desenvolvimento de ferramentas, métodos e teorias que sustentam a produção de software de alta qualidade dentro de limites econômicos viáveis.
Entrando na prática, para compreender a natureza da engenharia de software, pode-se fazer um paralelo com o urbanismo. Enquanto a engenharia civil lida com materiais físicos e com a lei da gravidade para erguer edifícios, a engenharia de software lida com a abstração para construir estruturas lógicas que devem suportar fluxos informacionais massivos. O software, neste contexto, está para uma cidade em constante expansão, se não houver um zoneamento rigoroso, arquitetura, um sistema de saneamento funcional, gestão de dados, e inspeções constantes, testes, a cidade entrará em colapso sob o peso de sua própria expansão. Esta disciplina é o esforço humano para impor ordem ao caos enorme de milhões de linhas de código de um sistema complexo.
Assim, a engenharia de software é alçada em princípios como abstração, modularidade e encapsulamento, os quais por sua vez contribuem para a redução da complexidade ao direcionar o desenvolvimento para os aspectos essenciais de um sistema. Esses princípios permitem mitigar a chamada complexidade arbitrária que surge ao longo do ciclo de vida do sistema, facilitando a organização e a manutenção dos sistemas. (PRESSMAN; MAXIM, 2016), enfatiza que o software é um elemento de sistema que é lógico, e não físico, ele não se desgasta no sentido tradicional, mas se deteriora devido a mudanças mal planejadas que introduzem instabilidades. Nesse contexto, a engenharia de software busca estabelecer processos sistemáticos e repetíveis capazes de aumentar a previsibilidade dos resultados, permitindo que o desenvolvimento de sistemas deixe de ser um "artesanato heroico" para se tornar uma produção industrial controlada.
A interdependência entre os componentes de um sistema moderno exige que o engenheiro de software possua uma visão ampla, compreendendo desde a delimitação dos requisitos até a estratégia de implantação e manutenção. Conforme Sommerville (2011), a engenharia de software está centralizada como uma engenharia de cunho econômico, onde a eficácia é medida pela capacidade de entregar valor ao negócio com custos adequados e alta confiabilidade. Um ponto muito importante é a definição do que o sistema deve ser e fazer.


2.4.1 Requisitos funcionais


Os requisitos funcionais são basicamente a especificação de o que o sistema deve fazer, servindo como o contrato primordial, um elo entre a equipe de desenvolvimento e os stakeholders, declarando suas funções, comportamentos e serviços, definindo a utilidade fundamental do produto para o usuário e para o negócio. 
A determinação desses requisitos é uma parte crucial do processo de descoberta e refinamento, onde o responsável deve traduzir necessidades muitas vezes ambíguas dos usuários em especificações técnicas objetivas. Atualmente, este processo é frequentemente documentado através de Casos de Uso ou Histórias de Usuário. Por exemplo, em um sistema de gestão de saúde mental, conforme exemplificado por Sommerville (2011), um requisito funcional seria "o sistema deve permitir que o médico prescreva medicamentos para um paciente", sendo uma ação crítica do sistema.


2.4.2 Requisitos não funcionais


Enquanto os requisitos funcionais tratam de o'que o sistema deve fazer, os requisitos não funcionais (RNFs) definem as restrições, qualidades e padrões de desempenho sob os quais o sistema deve operar (SOMMERVILLE, 2011), sendo o `como`, o sistema deve operar. Eles não descrevem funções ou objetivos específicos do programa, mas sim a forma como essas funções são executadas, abrangendo atributos como usabilidade, segurança, confiabilidade, portabilidade e eficiência.
Em essência, os requisitos não funcionais devem ser objetivos e quantificáveis, o que representa um dos maiores desafios da engenharia de requisitos. Por isso, especificações vagas como "o sistema deve ser rápido", devem ser substituídas por métricas objetivas, como, o sistema deve processar no mínimo 6 transações por segundo. 
Portanto, para ilustrar a importância dos RNFs, considere a analogia do oxigênio em um submarino. A função do submarino é navegar e realizar missões (requisitos funcionais), mas a qualidade do ar e a resistência do casco à pressão (requisitos não funcionais) são o que permitem que a missão seja bem-sucedida e segura. Se o ar acaba, o submarino para de funcionar como um ambiente habitável, independentemente de seus motores estarem operacionais. No software, um sistema que realiza todas as tarefas solicitadas, mas leva 30 segundos para carregar uma página, é um sistema que falhou em seus requisitos de performance, tornando-se inválido.


2.4.3 Metodologias de desenvolvimento


Primeiramente, as metodologias de desenvolvimento de software representam os frameworks de processos adotados pelas organizações para estruturar as atividades técnicas. Historicamente, o campo é dividido entre as metodologias tradicionais, baseadas em planejamento extenso, e as metodologias ágeis, focadas na adaptabilidade e no valor incremental. Em geral, a escolha metodológica é extremamente determinante para o sucesso do projeto e influência em uma série de fatores como o prazo, o custo e a qualidade do produto final.
O modelo tradicional mais conhecido é o Cascata (Waterfall), que tem como base uma sequência rígida de fases onde a saída de uma fase é a entrada obrigatória da próxima. Um paralelo claro para este modelo é o de uma linha de montagem industrial fordista clássica, não se pode instalar o motor de um carro antes que o chassi esteja pronto. No software, isso significa que a implementação só começa após a especificação de requisitos estar totalmente entregue. Embora ofereça previsibilidade em ambientes estáveis, este modelo falha drasticamente em cenários de alta incerteza, conforme evidenciado pelo CHAOS Report (STANDISH GROUP, 2020), que com base em 50.000 projetos demonstrou que projetos baseados no modelo Waterfall têm o dobro de probabilidade de falhar em comparação a projetos gerenciados com metodologias ágeis.
Em contrapartida, as metodologias ágeis, como Scrum e Extreme Programming (XP), baseiam-se em ciclos iterativos e incrementais. Como no livro A Origem das Espécies do naturalista Charles Robert Darwin,  o software "evolui" a cada iteração, adaptando-se às pressões do ambiente (feedback do cliente) e descartando o que não funciona. O Manifesto Ágil prioriza o software em funcionamento sobre a documentação abrangente, permitindo que o cliente participe ativamente de todo o ciclo de vida do projeto. Portanto, as metodologias ágeis buscam mitigar o risco de entregar algo que o cliente não deseja através de entregas frequentes e feedback constante.
        Com isso, os principais diferenciais entre essas duas filosofias, pode ser observado na Tabela 1, destacando como cada uma lida com o planejamento, a documentação e o papel dos envolvidos.


Tabela 1 — Comparativo entre Metodologia Tradicional e Metodologia Ágil.


Critério Comparativo
	Metodologia Tradicional
	Metodologia Ágil
	Planejamento
	Preditivo e Detalhado (Up-front)
	Adaptativo e Contínuo (Just-in-time)
	Requisitos
	Estáticos e formalizados no início
	Dinâmicos e refinados em iterações
	Documentação
	Artefatos exaustivos e burocráticos
	Apenas o necessário para a colaboração
	Feedback
	Ocorre ao final do ciclo de meses/anos
	Ocorre em ciclos de 2 a 4 semanas
	Papel do Cliente
	Fornece requisitos e valida o final
	Coautor e participante ativo do processo
	Fonte: Elaborado pelo autor (2026).

        A transição para o agilismo reflete a necessidade de velocidade e adaptação no mercado contemporâneo, que é caracterizado por ser um ambiente extremamente complexo de constante mudança devido à evolução tecnológica contínua. No entanto, conforme ressaltado por Schwaber e Sutherland (2020), as metodologias ágeis exigem equipes pequenas e altamente motivadas, podendo apresentar desafios em organizações de grande porte acostumadas a processos burocráticos. A eficácia dessas metodologias reside na valorização do fator humano; o sucesso do desenvolvimento depende fundamentalmente da competência comportamental e da colaboração da equipe, e não apenas de ferramentas técnicas, premissa base estabelecida pelo próprio Manifesto Ágil (BECK et al., 2001).
2.4.4 Estratégias de teste

Primeiramente, de acordo com Roger S. Pressman e Bruce Maxim (2016), a etapa de teste de software é o processo de avaliação técnica que tem como objetivo identificar diferenças entre o comportamento esperado de acordo com os requisitos funcionais e não funcionais e o comportamento observado no sistema no ambiente de testes. É uma etapa crítica no processo de desenvolvimento de software e garante qualidade para o sistema, atuando como um filtro que busca descobrir defeitos antes que eles se transformem em bugs em produção. Abaixo os 5 testes mais comuns no processo de desenvolvimento de software:


1. Teste de Unidade (Unit Test): Focado na menor unidade do sistema, geralmente um módulo ou função. O objetivo é validar individualmente cada unidade garantindo que elas funcionem isoladamente.
2. Teste de Integração (Integration Test): Focado em detectar falhas na comunicação entre módulos ou diferentes sistemas que já foram testados individualmente, pelo teste de unidade.
3. Teste de Sistema (System Test): Focado no hardware e seu desempenho da aplicação com um todo, são verificados requisitos de performance, segurança e compatibilidade.
4. Teste de Aceitação (Acceptance Test): Conduzido pelo cliente ou usuário final no ambiente de testes garantindo que o sistema está em conformidade com os requisitos funcionais e não funcionais, com isso está pronto para implantação.
5. Teste de Regressão (Regression Test): Focado no pós implantação, tem como objetivo garantir que as funcionalidades críticas que não foram alteradas, continuem funcionando corretamente.


A estratégia de teste é uma abordagem de melhoria contínua, evoluindo conforme o sistema cresce em complexidade. Um elemento importante que muitas vezes é esquecido é a documentação, que garante que todos os elementos do software foram adequadamente catalogados e estão prontos para o suporte e manutenção. Segundo Sommerville (2011), a validação não se limita à execução de código, mas inclui revisões técnicas eficazes que servem como filtro antes mesmo do início dos testes formais.
2.4.5 Arquitetura de sistemas
A arquitetura de software refere-se à organização estrutural de um sistema e às relações estabelecidas entre seus principais componentes. Em termos gerais, ela define como as partes de um sistema são distribuídas, como se comunicam e quais responsabilidades cada componente assume dentro da aplicação. Diferentemente das decisões de baixo nível, como algoritmos ou estruturas de dados, a arquitetura está relacionada às decisões estruturais que vão afetar o sistema no longo prazo. Conforme discutido por Richards e Ford (2020), a arquitetura de software moderna é composta pela combinação das características do sistema (os chamados ilities, como escalabilidade e disponibilidade), os componentes de software e as decisões de design que guiam a construção do ecossistema tecnológico.
        No contexto de sistemas modernos, a arquitetura tem um papel fundamental na capacidade de evolução do software, por exemplo uma aplicação desenvolvida sem uma organização arquitetural clara tende a acumular complexidade com o passar do tempo, devido a ajustes técnicos, funcionalidades foram do padrão e entre outros, tornando alterações e manutenções progressivamente mais difíceis com o passar do tempo. Segundo Len Bass, Paul Clements e Rick Kazman (2012), decisões arquiteturais influenciam diretamente atributos de qualidade como desempenho, escalabilidade, segurança e manutenção, razão pela qual essas decisões costumam ser tomadas nas fases iniciais do projeto.
        A escolha de uma arquitetura não é arbitrária, em geral, ela está diretamente relacionada aos requisitos não funcionais definidos para o sistema. Por exemplo, aplicações que demandam alta escalabilidade e disponibilidade frequentemente adotam arquiteturas distribuídas, enquanto sistemas mais simples podem se beneficiar de estruturas monolíticas ou em camadas. Como destacam Len Bass, Paul Clements e Rick Kazman (2012), a arquitetura funciona como um meio de comunicação entre os diferentes stakeholders do projeto, permitindo discutir decisões técnicas e avaliar antecipadamente seus impactos sobre a qualidade do sistema.
        Diferentes estilos arquitetônicos podem ser utilizados para estruturar sistemas de software, cada um apresentando características específicas relacionadas à organização dos componentes, comunicação entre módulos e escalabilidade da aplicação. A comparação entre os principais estilos de arquitetura da engenharia de software pode ser observada na Tabela 2.




















Tabela 2 – Comparação entre estilos de arquitetura de software


Estilo
	Definição
	Vantagens
	Desvantagens
	Cenários
	Monolítico
	Única unidade executável (módulos no mesmo processo).
	Desenvolvimento simples; implantação direta.
	Baixa escalabilidade; acoplamento elevado.
	Pequenos sistemas e protótipos.
	Microsserviços
	Serviços independentes que se comunicam via APIs.
	Alta escalabilidade; independência tecnológica.
	Complexidade de rede; alto custo operacional.
	Grandes sistemas distribuídos.
	Camadas
	Organização lógica (Apresentação, Negócio, Dados).
	Organização clara; fácil manutenção.
	Sobrecarga de comunicação entre camadas.
	Sistemas corporativos tradicionais.
	Cliente-Servidor
	Divisão entre solicitantes e processadores de dados.
	Controle centralizado e segurança.
	Dependência total do servidor central.
	Bancos de dados e apps web.
	Serverless
	Funções em nuvem acionadas por eventos.
	Escalabilidade automática; custo por uso.
	Vendor lock-in; controle limitado.
	Processamento sob demanda.
	

Fonte: Adaptado de Len Bass, Paul Clements e Rick Kazman (2012).
________________
