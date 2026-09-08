"""
01_Extracao/etl_dnit_v6.py
==========================
Pipeline de ETL da Base DNIT para a Arquitetura V6 (TCC 2026).

Ações Implementadas:
1. Leitura de 'dinit_combined_complete.csv'.
2. Remoção de cabeçalhos duplicados e linhas vazias.
3. Recuperação do ano de 2023 (as 38.529 linhas com Ano em branco).
4. Padronização da chave 'br' (extração numérica de 'Rodovia', ex: 'BR-116' -> 116).
5. Padronização e ordenação de limites espaciais ('km_inicial', 'km_final').
6. Renomeação para padrão minúsculo: 'icp' (era IP), 'icc' (era IC), 'icm' (era ICM).
7. Aplicação da Regra de Ouro do Sentinela -1.0 em dados ausentes (sem backfill artificial).
8. Exportação para 'dnit/df_dnit_processed.csv'.
"""

import os
import csv
import re

def run_dnit_etl_v6(input_csv="dinit_combined_complete.csv", output_csv="dnit/df_dnit_processed.csv"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    candidates_in = [
        os.path.join(base_dir, input_csv),
        input_csv
    ]
    in_path = next((p for p in candidates_in if os.path.exists(p)), candidates_in[0])
    
    out_path = os.path.join(base_dir, output_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    print("=" * 70)
    print("  ETL BASE DNIT — V6 (TCC 2026)")
    print("=" * 70)
    print(f"Lendo base bruta de: {in_path}")
    
    if not os.path.exists(in_path):
        print(f"ERRO: Arquivo {in_path} não encontrado!")
        return False
    
    output_headers = ['ano', 'uf', 'br', 'km_inicial', 'km_final', 'icp', 'icc', 'icm']
    
    stats = {
        'total_lidas': 0,
        'validas': 0,
        'cabecalhos_pulados': 0,
        'linhas_vazias': 0,
        'ano_2023_recuperado': 0,
        'br_invalida': 0,
        'km_invalido': 0,
        'por_ano': {}
    }
    
    with open(in_path, 'r', encoding='utf-8', errors='replace') as fp_in, \
         open(out_path, 'w', encoding='utf-8', newline='') as fp_out:
        
        reader = csv.reader(fp_in)
        writer = csv.writer(fp_out)
        
        header = next(reader)
        col_map = {col.strip(): i for i, col in enumerate(header)}
        
        writer.writerow(output_headers)
        
        for row_num, row in enumerate(reader, start=2):
            stats['total_lidas'] += 1
            
            if not row or not any(x.strip() for x in row):
                stats['linhas_vazias'] += 1
                continue
                
            if row[0].strip() == 'UF' or (len(row) > 2 and row[2].strip() == 'Rodovia'):
                stats['cabecalhos_pulados'] += 1
                continue
                
            uf = row[col_map.get('UF', 0)].strip()
            ano_raw = row[col_map.get('Ano', 1)].strip()
            rodovia_raw = row[col_map.get('Rodovia', 2)].strip()
            km_ini_raw = row[col_map.get('Km_Inicial', 3)].strip()
            km_fim_raw = row[col_map.get('Km_Final', 4)].strip()
            ip_raw = row[col_map.get('IP', 5)].strip()
            ic_raw = row[col_map.get('IC', 6)].strip()
            icm_raw = row[col_map.get('ICM', 7)].strip()
            
            if not ano_raw:
                ano = 2023
                stats['ano_2023_recuperado'] += 1
            else:
                try:
                    ano = int(float(ano_raw))
                except ValueError:
                    ano = 2023
                    stats['ano_2023_recuperado'] += 1
            
            br_match = re.findall(r'\d+', rodovia_raw)
            if not br_match:
                stats['br_invalida'] += 1
                continue
            br = int(br_match[0])
            
            try:
                km_ini = float(km_ini_raw.replace(',', '.'))
                km_fim = float(km_fim_raw.replace(',', '.'))
            except ValueError:
                stats['km_invalido'] += 1
                continue
                
            if km_ini > km_fim:
                km_ini, km_fim = km_fim, km_ini
                
            def parse_index(val):
                if not val or val == '':
                    return -1.0
                try:
                    v = float(val.replace(',', '.'))
                    return v if v >= 0 else -1.0
                except ValueError:
                    return -1.0
                    
            icp = parse_index(ip_raw)
            icc = parse_index(ic_raw)
            icm = parse_index(icm_raw)
            
            writer.writerow([ano, uf, br, km_ini, km_fim, icp, icc, icm])
            stats['validas'] += 1
            stats['por_ano'][ano] = stats['por_ano'].get(ano, 0) + 1
            
    print("\n--- RELATÓRIO DO PROCESSAMENTO DNIT V6 ---")
    print(f"Linhas Lidas Brutas     : {stats['total_lidas']:,}")
    print(f"Linhas Válidas Gravadas : {stats['validas']:,}")
    print(f"Ano 2023 Recuperado     : {stats['ano_2023_recuperado']:,} linhas")
    print(f"Cabeçalhos Pulados      : {stats['cabecalhos_pulados']}")
    print(f"Linhas Vazias Puladas   : {stats['linhas_vazias']}")
    print(f"BRs Inválidas Puladas   : {stats['br_invalida']}")
    print(f"KMs Inválidos Pulados   : {stats['km_invalido']}")
    print("\nDistribuição por Ano:")
    for yr in sorted(stats['por_ano'].keys()):
        print(f"  - Ano {yr}: {stats['por_ano'][yr]:,} medições")
    print(f"\nArquivo salvo com sucesso em: {out_path}")
    print("=" * 70)
    return True

if __name__ == "__main__":
    run_dnit_etl_v6()
