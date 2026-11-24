"""
11_Apresentacao_e_Dashboards/gerar_base_powerbi_pavimento_colisoes_v6.py
========================================================================
Gera a base analítica pronta para o Power BI unindo 100% dos microdados de
ocorrências da PRF (2019–2025) aos índices de pavimento do DNIT homologados
na Arquitetura V6 (df_model_ready_v6.csv).

Saída:
    11_Apresentacao_e_Dashboards/base_powerbi_pavimento_colisoes_v6.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PRF_CSV = BASE_DIR / "df_prf_consolidado_19-25.csv"
V6_CSV = BASE_DIR / "05_Pre_Modelagem" / "df_model_ready_v6.csv"
OUT_CSV = BASE_DIR / "11_Apresentacao_e_Dashboards" / "base_powerbi_pavimento_colisoes_v6.csv"

FAIXA_LABELS = [
    "Péssimo (0-20)",
    "Ruim (21-40)",
    "Regular (41-60)",
    "Bom (61-80)",
    "Ótimo (81-100)",
]
FAIXA_ORDEM_MAP = {label: idx + 1 for idx, label in enumerate(FAIXA_LABELS)}
FAIXA_ORDEM_MAP["Sem DNIT"] = 99

USO_SOLO_MAP = {
    "Não": "Rural",
    "Sim": "Urbano",
}

GRUPO_COLISAO_MAP = {
    "Colisão traseira": "Mesmo Sentido / Fluxo",
    "Colisão lateral mesmo sentido": "Mesmo Sentido / Fluxo",
    "Engavetamento": "Mesmo Sentido / Fluxo",
    "Colisão frontal": "Sentido Oposto / Frontal",
    "Colisão lateral sentido oposto": "Sentido Oposto / Frontal",
    "Saída de leito carroçável": "Perda de Controle / Saída de Pista",
    "Capotamento": "Perda de Controle / Saída de Pista",
    "Tombamento": "Perda de Controle / Saída de Pista",
    "Colisão com objeto": "Perda de Controle / Saída de Pista",
    "Colisão com objeto estático": "Perda de Controle / Saída de Pista",
    "Colisão com objeto em movimento": "Perda de Controle / Saída de Pista",
    "Colisão transversal": "Transversal / Cruzamento",
    "Atropelamento de Pedestre": "Atropelamento",
    "Atropelamento de Animal": "Atropelamento",
}


def classificar_periodo(ano: int) -> str:
    """Classifica o ano conforme a partição temporal oficial do Modelo V6."""
    if ano == 2019:
        return "2019 (Warm-up)"
    if ano <= 2023:
        return "2020-2023 (Treino)"
    return "2024-2025 (Teste)"


def main() -> None:
    """Executa a reconciliação PRF + DNIT (V6), valida métricas e exporta o CSV."""
    print(f"Lendo PRF: {PRF_CSV}")
    prf_cols = [
        "id",
        "data_inversa",
        "ano_base",
        "uf",
        "br",
        "km",
        "municipio",
        "tipo_pista",
        "tipo_acidente",
        "tracado_via",
        "uso_solo",
        "fase_dia",
        "condicao_metereologica",
        "causa_acidente",
        "mortos",
        "feridos_graves",
        "feridos_leves",
        "pessoas",
        "veiculos",
    ]
    prf = pd.read_csv(PRF_CSV, usecols=prf_cols, low_memory=False)

    prf["ano"] = pd.to_numeric(prf["ano_base"], errors="coerce").astype(int)
    prf["mes"] = pd.to_datetime(prf["data_inversa"], errors="coerce").dt.month.astype(int)
    prf["br"] = pd.to_numeric(prf["br"], errors="coerce").astype(int)
    km_num = pd.to_numeric(prf["km"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    prf["trecho_10km"] = ((km_num // 10) * 10).astype(int)

    print(f"Lendo índices DNIT da grade V6: {V6_CSV}")
    v6_cols = ["ano", "mes", "br", "trecho_10km", "icp", "icc", "icm", "has_dnit_data"]
    v6_dnit = pd.read_csv(V6_CSV, usecols=v6_cols, low_memory=False)

    # Chave V6 exata: (ano, mes, br, trecho_10km) sem UF para evitar perda de trechos limítrofes.
    merged = prf.merge(v6_dnit, on=["ano", "mes", "br", "trecho_10km"], how="left")

    assert len(merged) == 473734, f"Total esperado 473734, obtido {len(merged)}"
    assert merged["has_dnit_data"].isna().sum() == 0, "Há registros PRF sem correspondência na grade V6"
    assert (merged["has_dnit_data"] == 1).sum() == 167409, (
        f"Esperado 167409 registros com DNIT, obtido {(merged['has_dnit_data'] == 1).sum()}"
    )

    has_dnit = merged["has_dnit_data"].astype(int)
    mask_dnit = has_dnit == 1

    icp_clean = merged["icp"].where(mask_dnit, np.nan).astype(float)
    icc_clean = merged["icc"].where(mask_dnit, np.nan).astype(float)
    icm_clean = merged["icm"].where(mask_dnit, np.nan).astype(float)

    faixa_cut = pd.cut(
        icp_clean,
        bins=[-0.001, 20, 40, 60, 80, 100],
        labels=FAIXA_LABELS,
    ).astype("object")
    faixa_icp = faixa_cut.where(mask_dnit, "Sem DNIT")

    mortos = pd.to_numeric(merged["mortos"], errors="coerce").fillna(0).astype(int)
    feridos_graves = pd.to_numeric(merged["feridos_graves"], errors="coerce").fillna(0).astype(int)
    feridos_leves = pd.to_numeric(merged["feridos_leves"], errors="coerce").fillna(0).astype(int)
    pessoas = pd.to_numeric(merged["pessoas"], errors="coerce").fillna(0).astype(int)
    veiculos = pd.to_numeric(merged["veiculos"], errors="coerce").fillna(0).astype(int)
    vitimas_severas = mortos + feridos_graves

    trecho_ini = merged["trecho_10km"]
    trecho_fim = trecho_ini + 10
    trecho_label = (
        "BR-"
        + merged["br"].astype(str)
        + "/"
        + merged["uf"].astype(str)
        + " km "
        + trecho_ini.astype(str)
        + "-"
        + trecho_fim.astype(str)
    )

    out = pd.DataFrame(
        {
            "fk_data": merged["ano"] * 10000 + merged["mes"] * 100 + 1,
            "ano": merged["ano"],
            "mes": merged["mes"],
            "ano_mes": merged["ano"].map("{:04d}".format) + "-" + merged["mes"].map("{:02d}".format),
            "periodo_v6": merged["ano"].map(classificar_periodo),
            "uf": merged["uf"],
            "br": merged["br"],
            "trecho_10km": trecho_ini,
            "trecho_label": trecho_label,
            "municipio": merged["municipio"],
            "has_dnit_data": has_dnit,
            "icp": icp_clean,
            "icc": icc_clean,
            "icm": icm_clean,
            "faixa_icp": faixa_icp,
            "faixa_icp_ordem": faixa_icp.map(FAIXA_ORDEM_MAP).astype(int),
            "tipo_pista": merged["tipo_pista"].fillna("Não Informado"),
            "tipo_acidente": merged["tipo_acidente"].fillna("Não Informado"),
            "grupo_colisao": merged["tipo_acidente"].map(GRUPO_COLISAO_MAP).fillna("Outros"),
            "tracado_via": merged["tracado_via"].fillna("Não Informado"),
            "uso_solo_desc": merged["uso_solo"].map(USO_SOLO_MAP).fillna("Não Informado"),
            "fase_dia": merged["fase_dia"].fillna("Não Informado"),
            "condicao_metereologica": merged["condicao_metereologica"].fillna("Não Informado"),
            "causa_acidente": merged["causa_acidente"].fillna("Não Informado"),
            "qtd_acidentes": 1,
            "mortos": mortos,
            "feridos_graves": feridos_graves,
            "feridos_leves": feridos_leves,
            "vitimas_severas": vitimas_severas,
            "pessoas": pessoas,
            "veiculos": veiculos,
            "acidente_fatal": (mortos > 0).astype(int),
            "acidente_severo": (vitimas_severas > 0).astype(int),
        }
    )

    # Validação rigorosa contra as métricas oficiais do TCC (Seção 4.4.4)
    dnit_only = out[out["has_dnit_data"] == 1]
    resumo_faixa = dnit_only.groupby(["faixa_icp_ordem", "faixa_icp"], observed=False).agg(
        ac=("qtd_acidentes", "sum"),
        mortos=("mortos", "sum"),
        vitimas_severas=("vitimas_severas", "sum"),
    )
    resumo_faixa["mort_pct"] = resumo_faixa["mortos"] / resumo_faixa["ac"] * 100
    resumo_faixa["sev_pct"] = resumo_faixa["vitimas_severas"] / resumo_faixa["ac"] * 100

    pessimo = resumo_faixa.loc[(1, "Péssimo (0-20)")]
    bom = resumo_faixa.loc[(4, "Bom (61-80)")]
    assert int(pessimo["ac"]) == 15472, f"Esperado 15472 em Péssimo, obtido {pessimo['ac']}"
    assert round(float(pessimo["mort_pct"]), 2) == 10.50, f"Mortalidade Péssimo: {pessimo['mort_pct']:.4f}"
    assert round(float(pessimo["sev_pct"]), 1) == 41.1, f"Severas Péssimo: {pessimo['sev_pct']:.4f}"
    assert int(bom["ac"]) == 9653, f"Esperado 9653 em Bom, obtido {bom['ac']}"
    assert round(float(bom["mort_pct"]), 2) == 7.32, f"Mortalidade Bom: {bom['mort_pct']:.4f}"
    assert round(float(bom["sev_pct"]), 1) == 31.8, f"Severas Bom: {bom['sev_pct']:.4f}"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nCSV gerado com sucesso: {OUT_CSV} ({len(out):,} linhas, {len(out.columns)} colunas)")

    print("\n=== RESUMO POR FAIXA DE ICP (has_dnit_data == 1) ===")
    print(resumo_faixa.to_string())

    print("\n=== 1) FAIXA ICP x TIPO DE PISTA ===")
    tp = dnit_only.groupby(["faixa_icp_ordem", "faixa_icp", "tipo_pista"], observed=False).agg(
        ac=("qtd_acidentes", "sum"),
        mortos=("mortos", "sum"),
        sev=("vitimas_severas", "sum"),
    )
    tp["mort_pct"] = (tp["mortos"] / tp["ac"] * 100).round(2)
    tp["sev_pct"] = (tp["sev"] / tp["ac"] * 100).round(1)
    print(tp.to_string())

    print("\n=== 2) FAIXA ICP x GRUPO DE COLISÃO (% dentro da faixa) ===")
    gc = dnit_only.groupby(["faixa_icp_ordem", "faixa_icp", "grupo_colisao"], observed=False).agg(
        ac=("qtd_acidentes", "sum"),
        mortos=("mortos", "sum"),
    )
    gc["share_pct"] = (
        gc["ac"] / gc.groupby(level=[0, 1])["ac"].transform("sum") * 100
    ).round(1)
    gc["mort_pct"] = (gc["mortos"] / gc["ac"] * 100).round(2)
    print(gc.to_string())

    print("\n=== 3) TIPO DE ACIDENTE (TOP 10 EM has_dnit_data == 1) ===")
    ta = (
        dnit_only.groupby("tipo_acidente")
        .agg(ac=("qtd_acidentes", "sum"), mortos=("mortos", "sum"), sev=("vitimas_severas", "sum"))
        .sort_values("ac", ascending=False)
        .head(10)
    )
    ta["mort_pct"] = (ta["mortos"] / ta["ac"] * 100).round(2)
    ta["sev_pct"] = (ta["sev"] / ta["ac"] * 100).round(1)
    print(ta.to_string())

    print("\n=== 4) FAIXA ICP x USO DO SOLO (Rural vs Urbano) ===")
    us = dnit_only.groupby(["faixa_icp_ordem", "faixa_icp", "uso_solo_desc"], observed=False).agg(
        ac=("qtd_acidentes", "sum"),
        mortos=("mortos", "sum"),
    )
    us["mort_pct"] = (us["mortos"] / us["ac"] * 100).round(2)
    print(us.to_string())

    print("\n=== 5) FAIXA ICP x FASE DO DIA (Pleno dia vs Plena Noite) ===")
    fd = (
        dnit_only[dnit_only["fase_dia"].isin(["Pleno dia", "Plena Noite"])]
        .groupby(["faixa_icp_ordem", "faixa_icp", "fase_dia"], observed=False)
        .agg(ac=("qtd_acidentes", "sum"), mortos=("mortos", "sum"))
    )
    fd["mort_pct"] = (fd["mortos"] / fd["ac"] * 100).round(2)
    print(fd.to_string())


if __name__ == "__main__":
    main()
