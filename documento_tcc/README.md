# Estrutura Modular do Documento do TCC (2026)

Este diretório contém a monografia completa do TCC particionada em seções modulares para otimizar o fluxo de edição e revisão com a IA, evitando estouro de contexto e permitindo intervenções focadas.

---

## Mapa das Seções e Arquivos

| Capítulo / Seção | Arquivo Modular | Linhas | Foco de Atualização para V6 |
| :--- | :--- | :---: | :--- |
| **0. Elementos Pré-Textuais** | `00_pre_textual/00_capa_e_folha_rosto.md`<br>`00_pre_textual/01_resumo_abstract.md`<br>`00_pre_textual/02_lista_figuras.md`<br>`00_pre_textual/03_lista_tabelas.md`<br>`00_pre_textual/04_sumario.md` | ~270 | Atualizar resumo com métricas finais da V6 ($R^2=0.8678$, MAE=0.61) e listas de elementos ilustrativos. |
| **1. Introdução** | `01_introducao/1_introducao.md` | 43 | Citar a evolução metodológica das arquiteturas. |
| **2. Revisão da Literatura** | `02_revisao_literatura/2_1_panorama_seguranca_viaria.md`<br>`02_revisao_literatura/2_2_descritiva_a_preditiva_etl_crisp_rf.md`<br>`02_revisao_literatura/2_3_business_intelligence_powerbi.md`<br>`02_revisao_literatura/2_4_engenharia_software.md` | ~252 | Teórico / Estável. Mantém fundamentos de CRISP-DM e Random Forest. |
| **3. Materiais e Métodos** | `03_materiais_e_metodos/3_1_materiais.md`<br>`03_materiais_e_metodos/3_2_1_entendimento_negocio.md`<br>`03_materiais_e_metodos/3_2_2_entendimento_dados.md`<br>`03_materiais_e_metodos/3_2_3_preparacao_dados.md`<br>`03_materiais_e_metodos/3_2_4_modelagem.md` | ~525 | **ZONA CRÍTICA DE ATUALIZAÇÃO:**<br>- `3_2_3`: Lags temporais, Grade Cartesiana de zeros reais e integração DNIT.<br>- `3_2_4`: Split temporal estrito 2020-2023 / 2024-2025 e perda Poisson. |
| **4. Resultados** | `04_resultados/4_1_avaliacao_comparativa.md`<br>`04_resultados/4_2_diagnostico_modelo.md`<br>`04_resultados/4_3_importancia_variaveis.md`<br>`04_resultados/4_4_impacto_infraestrutura_dnit.md`<br>`04_resultados/4_5_cenarios_taticos_estudo_caso.md` | ~258 | **ZONA CRÍTICA DE ATUALIZAÇÃO:**<br>Substituir dados de V5 pelas métricas e gráficos consolidados da Arquitetura V6. |
| **5. Discussão** | `05_discussao/5_discussao.md` | 55 | Incorporar Efeito Peltzman e discussão de severidade viária. |
| **6. Conclusão** | `06_conclusao/6_conclusao.md` | 145 | Sintetizar contribuições e aprendizados da V6. |
| **7. Referências** | `07_referencias/7_referencias.md` | 49 | Referências bibliográficas ABNT. |

---

## Como Trabalhar nas Revisões

1. Cada arquivo possui entre 15 e 180 linhas, facilitando a edição granular sem sobrecarregar a memória do modelo.
2. A skill **humanizer** (`.agents/skills/humanizer`) está integrada para remover padrões artificiais e clichês de IA na escrita acadêmica.
3. Quando você estiver pronto para compilar o texto completo, podemos concatenar todos os arquivos na ordem numérica exata.
