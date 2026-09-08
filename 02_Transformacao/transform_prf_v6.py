"""
02_Transformacao/transform_prf_v6.py
====================================
Pipeline de Transformação e Agregação da PRF para a Arquitetura V6 (TCC 2026).

Ações Implementadas:
1. Ingestão de 'df_prf_consolidado_19-25.csv' (473.734 ocorrências).
2. Limpeza e validação de chaves essenciais ('data_inversa', 'br', 'km').
3. Extração das chaves de segmentação: 'ano', 'mes' e 'trecho_10km'.
4. Agrupamento estrito por (ano, mes, br, trecho_10km) para calcular 'quantidade_acidentes'.
5. Eliminação de todas as dummies de contexto ('is_*'), erradicando o Target Leakage.
6. Exportação para '02_Transformacao/prf_aggregated_counts.csv'.
"""

import os
import csv
from collections import defaultdict

def run_prf_transform_v6(input_csv="df_prf_consolidado_19-25.csv", output_csv="02_Transformacao/prf_aggregated_counts.csv"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    candidates_in = [
        os.path.join(base_dir, input_csv),
        input_csv
    ]
    in_path = next((p for p in candidates_in if os.path.exists(p)), candidates_in[0])
    out_path = os.path.join(base_dir, output_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    print("=" * 70)
    print("  ETL BASE PRF — V6 (TCC 2026)")
    print("=" * 70)
    print(f"Lendo ocorrências brutas da PRF de: {in_path}")
    
    if not os.path.exists(in_path):
        print(f"ERRO: Arquivo {in_path} não encontrado!")
        return False
        
    counts = defaultdict(lambda: {
        'quantidade_acidentes': 0,
        'total_mortos': 0,
        'total_feridos_graves': 0,
        'total_veiculos': 0,
        'uf_counter': defaultdict(int)
    })
    
    total_lidos = 0
    linhas_validas = 0
    nulos_descartados = 0
    
    with open(in_path, 'r', encoding='utf-8', errors='replace') as fp_in:
        reader = csv.reader(fp_in)
        header = next(reader)
        col_map = {col.strip(): i for i, col in enumerate(header)}
        
        data_idx = col_map.get('data_inversa')
        br_idx = col_map.get('br')
        km_idx = col_map.get('km')
        uf_idx = col_map.get('uf')
        mortos_idx = col_map.get('mortos')
        fg_idx = col_map.get('feridos_graves')
        veic_idx = col_map.get('veiculos')
        
        for row in reader:
            total_lidos += 1
            
            # Validação das chaves essenciais
            if not row or len(row) <= max(data_idx, br_idx, km_idx):
                nulos_descartados += 1
                continue
                
            data_str = row[data_idx].strip()
            br_str = row[br_idx].strip()
            km_str = row[km_idx].strip()
            uf_str = row[uf_idx].strip() if uf_idx is not None and uf_idx < len(row) else ""
            
            if not data_str or not br_str or not km_str:
                nulos_descartados += 1
                continue
                
            try:
                # Extrai ano e mês de YYYY-MM-DD
                ano = int(data_str[:4])
                mes = int(data_str[5:7])
                br = int(br_str)
                km = float(km_str.replace(',', '.'))
                trecho_10km = int(km // 10) * 10
            except (ValueError, IndexError):
                nulos_descartados += 1
                continue
                
            def to_int(val_idx):
                if val_idx is not None and val_idx < len(row):
                    try: return int(float(row[val_idx].replace(',', '.')))
                    except: return 0
                return 0
                
            mortos = to_int(mortos_idx)
            fg = to_int(fg_idx)
            veic = to_int(veic_idx)
            
            key = (ano, mes, br, trecho_10km)
            bucket = counts[key]
            bucket['quantidade_acidentes'] += 1
            bucket['total_mortos'] += mortos
            bucket['total_feridos_graves'] += fg
            bucket['total_veiculos'] += veic
            if uf_str:
                bucket['uf_counter'][uf_str] += 1
                
            linhas_validas += 1
            
    print(f"Total de Ocorrências Lidas : {total_lidos:,}")
    print(f"Ocorrências Válidas        : {linhas_validas:,}")
    print(f"Ocorrências com Chave Nula : {nulos_descartados:,}")
    print(f"Buckets Espaço-Temporais   : {len(counts):,} (com acidentes)")
    
    # Grava agregação
    output_headers = ['ano', 'mes', 'br', 'trecho_10km', 'uf', 'quantidade_acidentes', 'total_mortos', 'total_feridos_graves', 'total_veiculos']
    with open(out_path, 'w', encoding='utf-8', newline='') as fp_out:
        writer = csv.writer(fp_out)
        writer.writerow(output_headers)
        
        for (ano, mes, br, trecho_10km), data in sorted(counts.items()):
            # UF predominante do bucket
            uf_pred = max(data['uf_counter'].items(), key=lambda x: x[1])[0] if data['uf_counter'] else ""
            writer.writerow([
                ano, mes, br, trecho_10km, uf_pred,
                data['quantidade_acidentes'],
                data['total_mortos'],
                data['total_feridos_graves'],
                data['total_veiculos']
            ])
            
    print(f"\nArquivo consolidado salvo em: {out_path}")
    print("=" * 70)
    return True

if __name__ == "__main__":
    run_prf_transform_v6()
