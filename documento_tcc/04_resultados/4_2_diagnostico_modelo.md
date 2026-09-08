4.2 Diagnóstico do Modelo V5


A avaliação da qualidade das estimativas do Modelo V5 envolve o diagnóstico dos erros de previsão e o estudo da contribuição de cada variável explicativa. 
A Figura 18, logo abaixo, apresenta um painel de diagnóstico dos resíduos do Modelo V5, composto pelo histograma de erros e pelo gráfico de dispersão Predito versus Real, no qual são instrumentos fundamentais para avaliar se o modelo comete erros aleatórios ou sistemáticos, e mais importante, em que regiões da distribuição esses erros se concentram.

Figura 18 – Análise de resíduos do Modelo V5: histograma de erros e dispersão predito versus real.


  

Fonte: Elaborado pelo autor (2026).


O primeiro gráfico, o histograma de resíduos, evidencia algo já sabido dado à arquitetura do modelo, apresenta uma distribuição de erros esmagadoramente concentrada em zeros, e uma massa de erros limitada basicamente a -2 e +2 acidentes. Entretanto, pode-se observar uma cauda a direita, com pequenas ocorrências de erros que vão até cerca de 27 acidentes. Advento que faz sentido, dado às características dos dados que apresenta alta assimetria e valores extremos, com isso, naturalmente o modelo tende a subestimar eventos extremos, pois são raridade no dataset. O gráfico evidencia a taxa de sucesso do modelo a estimar o número de acidentes.
Essa leve subestimação nos picos decorre do próprio mecanismo teórico do Random Forest detalhado no item 3.1.4, como a predição final de regressão é a média das predições de todas as árvores individuais, o modelo promove uma suavização estatística natural na cauda extrema da distribuição.
        O gráfico de dispersão Predito vs Real, ajuda a entender esse cenário, uma vez que existe uma densa nuvem de pontos em aderência com a linha diagonal de previsão, apenas após 40 acidentes é possível observar um deslocamento onde o modelo tende a subestimar os eventos extremos, devido a sua raridade na realidade. 
        Trazendo o modelo para o dia a dia das rodovias brasileiras, do ponto de vista operacional, os resultados da análise de resíduos possuem uma relevância estratégica direta para os stakeholders do projeto. O modelo é altamente preciso para os trechos de sinistralidade rotineira, exatamente onde a maior parte das decisões de alocação preventiva de recursos é tomada no dia a dia da PRF, e conservador, ao subestimar, nos trechos de sinistralidade extrema. Essa propriedade é operacionalmente desejável: um modelo que sistematicamente subestima os pontos mais críticos força os gestores a manter uma margem de segurança na alocação de recursos nesses trechos, reduzindo o risco de subinvestimento nas vias de maior letalidade acumulada.
