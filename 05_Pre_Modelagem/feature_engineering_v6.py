"""
05_Pre_Modelagem/feature_engineering_v6.py
=========================================
Engenharia de Atributos Pré-Evento e Sem Vazamento (V6 - TCC 2026).

Ações Implementadas:
1. Leitura da Grade Cartesiana Mestre com Zeros Reais ('grade_mestre_zeros.csv').
2. Geração de variáveis exógenas de Calendário e Sazonalidade:
   - 'dias_no_mes'
   - 'dias_fim_semana' (sábados e domingos reais do calendário)
   - 'dias_feriado' (feriados nacionais oficiais do mês)
   - 'estacao_ano' (Verão, Outono, Inverno, Primavera)
3. Agregação espacial das notas de pavimento do DNIT (ICP, ICC, ICM):
   - Média dos segmentos do DNIT contidos no bucket de 10km naquele ano.
   - Sentinela -1.0 em ausências + flag binária 'has_dnit_data'.
4. Engenharia de Lags Temporais puros (sem vazamento entre trechos):
   - 'acidentes_lag_1m'
   - 'acidentes_lag_2m'
   - 'acidentes_media_movel_3m'
   - 'acidentes_lag_12m' (sazonalidade anual do próprio trecho)
5. Exportação do dataset definitivo 'df_model_ready_v6.csv'.
"""

import os
import csv
import calendar
from datetime import date
from collections import defaultdict

def get_holidays_count(ano, mes):
    # Feriados fixos
    fixed = [(1, 1), (21, 4), (1, 5), (7, 9), (12, 10), (2, 11), (15, 11), (25, 12)]
    if ano >= 2024:
        fixed.append((20, 11)) # Consciência Negra
        
    # Carnaval, Sexta-feira Santa, Corpus Christi (datas móveis 2019-2025)
    moveis = {
        2019: [(3, 4), (3, 5), (4, 19), (6, 20)],
        2020: [(2, 24), (2, 25), (4, 10), (6, 11)],
        2021: [(2, 15), (2, 16), (4, 2), (6, 3)],
        2022: [(2, 28), (3, 1), (4, 15), (6, 16)],
        2023: [(2, 20), (2, 21), (4, 7), (6, 8)],
        2024: [(2, 12), (2, 13), (3, 29), (5, 30)],
        2025: [(3, 3), (3, 4), (4, 18), (6, 19)]
    }
    
    count = 0
    for d, m in fixed:
        if m == mes:
            count += 1
    for m, d in moveis.get(ano, []):
        if m == mes:
            count += 1
    return count

def get_weekend_days(ano, mes):
    cal = calendar.monthcalendar(ano, mes)
    # sábado = cal[w][5], domingo = cal[w][6]
    fds = sum(1 for w in cal if w[5] > 0) + sum(1 for w in cal if w[6] > 0)
    return fds

def get_season(mes):
    if mes in [12, 1, 2]: return 'Verao'
    elif mes in [3, 4, 5]: return 'Outono'
    elif mes in [6, 7, 8]: return 'Inverno'
    else: return 'Primavera'

def run_feature_engineering_v6():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grid_csv = os.path.join(base_dir, "04_Integracao_DNIT", "grade_mestre_zeros.csv")
    dnit_csv = os.path.join(base_dir, "dnit", "df_dnit_processed.csv")
    out_csv = os.path.join(base_dir, "05_Pre_Modelagem", "df_model_ready_v6.csv")
    out_csv_root = os.path.join(base_dir, "df_model_ready_v6.csv")
    
    print("=" * 70)
    print("  ENGENHARIA DE ATRIBUTOS PRÉ-EVENTO — V6 (TCC 2026)")
    print("=" * 70)
    
    # 1. Carregar Medições do DNIT em Dicionário Espacial
    # dnit_map: (ano, br) -> list of (km_ini, km_fim, icp, icc, icm)
    print(f"Carregando e indexando infraestrutura DNIT de: {dnit_csv}...")
    dnit_index = defaultdict(list)
    with open(dnit_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ano = int(row['ano'])
            br = int(row['br'])
            k_ini = float(row['km_inicial'])
            k_fim = float(row['km_final'])
            icp = float(row['icp'])
            icc = float(row['icc'])
            icm = float(row['icm'])
            dnit_index[(ano, br)].append((k_ini, k_fim, icp, icc, icm))
            
    print(f"Pares (Ano, BR) com medições DNIT: {len(dnit_index):,}")
    
    # Função para calcular média DNIT no bucket [t, t+10)
    def query_dnit(ano, br, trecho):
        segments = dnit_index.get((ano, br), [])
        if not segments:
            return -1.0, -1.0, -1.0, 0
            
        t_ini = float(trecho)
        t_fim = t_ini + 10.0
        
        icp_vals, icc_vals, icm_vals = [], [], []
        for k_ini, k_fim, icp, icc, icm in segments:
            if k_ini < t_fim and k_fim > t_ini:
                if icp >= 0: icp_vals.append(icp)
                if icc >= 0: icc_vals.append(icc)
                if icm >= 0: icm_vals.append(icm)
                
        if icp_vals:
            avg_icp = round(sum(icp_vals) / len(icp_vals), 2)
            avg_icc = round(sum(icc_vals) / len(icc_vals), 2)
            avg_icm = round(sum(icm_vals) / len(icm_vals), 2)
            return avg_icp, avg_icc, avg_icm, 1
        else:
            return -1.0, -1.0, -1.0, 0

    # 2. Carregar Grade Mestre e Agrupar por Trecho para cálculo dos Lags
    print(f"Carregando grade com Zeros Reais de: {grid_csv}...")
    
    # Estrutura: trechos_data[(br, trecho)] = list of rows ordenada por tempo
    trechos_series = defaultdict(list)
    
    with open(grid_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ano = int(row['ano'])
            mes = int(row['mes'])
            br = int(row['br'])
            trecho = int(row['trecho_10km'])
            uf = row['uf']
            qtd = int(row['quantidade_acidentes'])
            mortos = int(row['total_mortos'])
            fg = int(row['total_feridos_graves'])
            veic = int(row['total_veiculos'])
            is_zero = int(row['is_zero_real'])
            
            trechos_series[(br, trecho)].append({
                'ano': ano, 'mes': mes, 'br': br, 'trecho_10km': trecho,
                'uf': uf, 'quantidade_acidentes': qtd,
                'total_mortos': mortos, 'total_feridos_graves': fg,
                'total_veiculos': veic, 'is_zero_real': is_zero
            })
            
    print(f"Trechos a processar para Lags Temporais: {len(trechos_series):,}")
    
    # 3. Processar Lags e Atributos Exógenos
    print("Computando Lags Históricos e Atributos de Calendário/DNIT...")
    
    output_headers = [
        'ano', 'mes', 'br', 'trecho_10km', 'uf',
        'dias_no_mes', 'dias_fim_semana', 'dias_feriado', 'estacao_ano',
        'icp', 'icc', 'icm', 'has_dnit_data',
        'acidentes_lag_1m', 'acidentes_lag_2m', 'acidentes_media_movel_3m', 'acidentes_lag_12m',
        'total_mortos', 'total_feridos_graves', 'total_veiculos', 'is_zero_real',
        'quantidade_acidentes' # TARGET Y
    ]
    
    total_linhas_processadas = 0
    linhas_com_dnit = 0
    
    with open(out_csv, 'w', encoding='utf-8', newline='') as f_out, \
         open(out_csv_root, 'w', encoding='utf-8', newline='') as f_root:
         
        writer_out = csv.writer(f_out)
        writer_root = csv.writer(f_root)
        
        writer_out.writerow(output_headers)
        writer_root.writerow(output_headers)
        
        for (br, trecho), timeline in trechos_series.items():
            # Ordena estritamente por tempo cronológico (ano, mes)
            timeline.sort(key=lambda x: (x['ano'], x['mes']))
            
            # Histórico de acidentes da série deste trecho
            qtd_history = [item['quantidade_acidentes'] for item in timeline]
            
            for idx, item in enumerate(timeline):
                ano = item['ano']
                mes = item['mes']
                uf = item['uf']
                qtd = item['quantidade_acidentes']
                mortos = item['total_mortos']
                fg = item['total_feridos_graves']
                veic = item['total_veiculos']
                is_zero = item['is_zero_real']
                
                # Calendário
                _, num_dias = calendar.monthrange(ano, mes)
                fds = get_weekend_days(ano, mes)
                feriados = get_holidays_count(ano, mes)
                estacao = get_season(mes)
                
                # DNIT
                icp, icc, icm, has_dnit = query_dnit(ano, br, trecho)
                if has_dnit:
                    linhas_com_dnit += 1
                    
                # Lags Temporais do próprio trecho
                lag_1m = qtd_history[idx - 1] if idx >= 1 else 0
                lag_2m = qtd_history[idx - 2] if idx >= 2 else 0
                
                if idx >= 3:
                    mm_3m = round((qtd_history[idx-1] + qtd_history[idx-2] + qtd_history[idx-3]) / 3.0, 2)
                elif idx == 2:
                    mm_3m = round((qtd_history[idx-1] + qtd_history[idx-2]) / 2.0, 2)
                elif idx == 1:
                    mm_3m = float(qtd_history[idx-1])
                else:
                    mm_3m = 0.0
                    
                lag_12m = qtd_history[idx - 12] if idx >= 12 else 0
                
                row_data = [
                    ano, mes, br, trecho, uf,
                    num_dias, fds, feriados, estacao,
                    icp, icc, icm, has_dnit,
                    lag_1m, lag_2m, mm_3m, lag_12m,
                    mortos, fg, veic, is_zero,
                    qtd # TARGET Y
                ]
                
                writer_out.writerow(row_data)
                writer_root.writerow(row_data)
                total_linhas_processadas += 1
                
    print("\n--- RESUMO DO DATASET V6 FINAL ---")
    print(f"Total de Linhas Processadas : {total_linhas_processadas:,}")
    print(f"Trechos com Cobertura DNIT  : {linhas_com_dnit:,} ({linhas_com_dnit/total_linhas_processadas*100:.1f}%)")
    print(f"Arquivo V6 gerado em        : {out_csv}")
    print(f"Cópia na raiz gerada em     : {out_csv_root}")
    print("=" * 70)
    return True

if __name__ == "__main__":
    run_feature_engineering_v6()
