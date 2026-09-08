"""
04_Integracao_DNIT/build_cartesian_grid_v6.py
=============================================
Construção da Grade Cartesiana Espaço-Temporal Completa (V6 - TCC 2026).

Objetivo Metodológico:
Eliminar o Viés de Sobrevivência (Survivorship Bias) integrando a totalidade
dos trechos federais com todos os 84 meses do período (2019 a 2025).
Segmentos que não registraram sinistros em determinado mês recebem Y = 0.
"""

import os
import csv
from collections import defaultdict

def run_build_cartesian_grid_v6(
    prf_counts_csv="02_Transformacao/prf_aggregated_counts.csv",
    dnit_csv="dnit/df_dnit_processed.csv",
    output_csv="04_Integracao_DNIT/grade_mestre_zeros.csv"
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    prf_path = os.path.join(base_dir, prf_counts_csv)
    dnit_path = os.path.join(base_dir, dnit_csv)
    out_path = os.path.join(base_dir, output_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    print("=" * 70)
    print("  CONSTRUÇÃO DA GRADE CARTESIANA COM ZEROS REAIS — V6 (TCC 2026)")
    print("=" * 70)
    
    # 1. Coleta do Universo de Trechos e UFs
    trechos_info = {} # (br, trecho_10km) -> {'uf_counter': defaultdict(int)}
    
    # A partir da PRF
    print(f"Mapeando trechos da PRF de: {prf_path}...")
    prf_data = {}
    with open(prf_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ano = int(row['ano'])
            mes = int(row['mes'])
            br = int(row['br'])
            trecho = int(row['trecho_10km'])
            uf = row['uf']
            
            key_trecho = (br, trecho)
            if key_trecho not in trechos_info:
                trechos_info[key_trecho] = defaultdict(int)
            if uf:
                trechos_info[key_trecho][uf] += 1
                
            prf_data[(ano, mes, br, trecho)] = {
                'qtd': int(row['quantidade_acidentes']),
                'mortos': int(row['total_mortos']),
                'fg': int(row['total_feridos_graves']),
                'veic': int(row['total_veiculos'])
            }
            
    # A partir do DNIT
    print(f"Mapeando trechos adicionais do DNIT de: {dnit_path}...")
    with open(dnit_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            br = int(row['br'])
            uf = row['uf']
            k_ini = float(row['km_inicial'])
            k_fim = float(row['km_final'])
            
            t_start = int(k_ini // 10) * 10
            t_end = int(k_fim // 10) * 10
            
            for t in range(t_start, t_end + 10, 10):
                key_trecho = (br, t)
                if key_trecho not in trechos_info:
                    trechos_info[key_trecho] = defaultdict(int)
                if uf:
                    trechos_info[key_trecho][uf] += 1

    trechos_ordenados = sorted(trechos_info.keys())
    print(f"\nTotal de Trechos Únicos de 10km Mapeados: {len(trechos_ordenados):,}")
    
    # Define a UF estática do trecho
    trecho_uf_map = {}
    for t_key, ufs in trechos_info.items():
        trecho_uf_map[t_key] = max(ufs.items(), key=lambda x: x[1])[0] if ufs else "BR"

    # 2. Geração da Grade Cartesiana Completa: Trechos x 84 Meses
    anos = list(range(2019, 2026)) # 2019 a 2025 (7 anos)
    meses = list(range(1, 13))       # 1 a 12
    total_esperado = len(trechos_ordenados) * len(anos) * len(meses)
    print(f"Total de Cenários Espaço-Temporais Esperados: {total_esperado:,}")
    
    zeros_adicionados = 0
    positivos_mantidos = 0
    
    output_headers = ['ano', 'mes', 'br', 'trecho_10km', 'uf', 'quantidade_acidentes', 'total_mortos', 'total_feridos_graves', 'total_veiculos', 'is_zero_real']
    
    print(f"\nGravando grade mestre com Zeros Reais em: {out_path}...")
    with open(out_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(output_headers)
        
        for br, trecho in trechos_ordenados:
            uf = trecho_uf_map.get((br, trecho), "BR")
            for ano in anos:
                for mes in meses:
                    key = (ano, mes, br, trecho)
                    if key in prf_data:
                        d = prf_data[key]
                        writer.writerow([ano, mes, br, trecho, uf, d['qtd'], d['mortos'], d['fg'], d['veic'], 0])
                        positivos_mantidos += 1
                    else:
                        writer.writerow([ano, mes, br, trecho, uf, 0, 0, 0, 0, 1])
                        zeros_adicionados += 1
                        
    print("\n--- RELATÓRIO DA GRADE CARTESIANA V6 ---")
    print(f"Observações com Acidentes (Y >= 1) : {positivos_mantidos:,} ({positivos_mantidos/total_esperado*100:.1f}%)")
    print(f"Observações Seguras (Y = 0)         : {zeros_adicionados:,} ({zeros_adicionados/total_esperado*100:.1f}%)")
    print(f"Total da Matriz V6                 : {total_esperado:,} linhas")
    print(f"Arquivo salvo em: {out_path}")
    print("=" * 70)
    return True

if __name__ == "__main__":
    run_build_cartesian_grid_v6()
