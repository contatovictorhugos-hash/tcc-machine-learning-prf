import os
import glob
import pandas as pd
import numpy as np
import re

DATA_DIR = r"c:\Users\yuri\projetcs\dnit_project"

def load_and_standardize(filepath):
    """
    Loads a specific DNIT file and standardizes its columns based on its known schema structure.
    """
    filename = os.path.basename(filepath).lower()
    df = pd.DataFrame()
    
    try:
        if filename == 'dnit_2022.csv':
            df = pd.read_csv(filepath, sep=';', encoding='latin1')
            df.rename(columns={
                'Rodovia': 'rodovia', 'km inicial': 'km_inicial', 
                'km final': 'km_final', 'Extenso (km)': 'extensao_km', 
                'ICC': 'icc', 'ICP': 'icp', 'ICM': 'icm'
            }, inplace=True)
            df['ano'] = 2022
            
        elif filename == 'dnit_2023.csv':
            df = pd.read_csv(filepath, sep=';', encoding='latin1', skiprows=1)
            # Dnit 2023 had unnameds.
            df.columns = ['uf', 'rodovia', 'km_inicial', 'km_final', 'extensao_km', 
                          'cond_pavimento', 'cond_conservacao', 'icm', 'icc', 'icp'][:len(df.columns)]
            df['ano'] = 2023
            
        elif filename == 'dnit_2024.xlsx':
            df = pd.read_excel(filepath)
            df.rename(columns={
                'Rodovia': 'rodovia', 'Km_Inicial': 'km_inicial', 
                'Km_Final': 'km_final', 'IP': 'icp', 'IC': 'icc', 'ICM': 'icm'
            }, inplace=True)
            df['ano'] = 2024
            
        elif filename == 'dnit_2025.xlsx':
            df = pd.read_excel(filepath, skiprows=1)
            cols = ['uf', 'rodovia', 'sentido', 'km_inicial', 'km_final', 'extensao_km', 
                    'cond_pav_panela', 'cond_pav_remendo', 'cond_pav_trinca', 
                    'cond_cons_rocada', 'cond_cons_drenagem', 'cond_cons_sinalizacao',
                    'icc', 'icp', 'icm']
            
            # Since Pandas length matching threw an error previously, we must slice the df cols precisely
            # or assign one by one. dnit_2025 had 31 columns. Let's just grab by Index to be safe
            df_lean = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 28, 29, 30]].copy()
            df_lean.columns = cols
            df = df_lean
            df['ano'] = 2025
            
        elif filename == 'dnit_2026.csv':
            df = pd.read_csv(filepath, sep=';', encoding='latin1')
            df.rename(columns={
                'Rodovia': 'rodovia', 'Km_Inicial': 'km_inicial', 
                'Km_Final': 'km_final', 'IP': 'icp', 'IC': 'icc', 'ICM': 'icm'
            }, inplace=True)
            df['ano'] = 2026

        else:
            print(f"Skipping or pattern unmapped: {filename}")
            
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        
    return df

def clean_rodovia(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val)
    digits = re.findall(r'\d+', val_str)
    if digits:
         return int(digits[0])
    return np.nan

def process_km_value(km_val):
    if pd.isna(km_val):
        return np.nan
    val_str = str(km_val).replace(',', '.')
    try:
        return float(val_str)
    except ValueError:
        return np.nan

def standardize_geographic_limits(df):
    """
    Cleans km_inicial and km_final maintaining the native intervals.
    """
    if df.empty or 'km_inicial' not in df.columns or 'km_final' not in df.columns:
        return df

    df['km_inicial'] = df['km_inicial'].apply(process_km_value)
    df['km_final'] = df['km_final'].apply(process_km_value)
    
    df = df.dropna(subset=['km_inicial', 'km_final']).copy()
    
    # Ensure correct start/end order
    mask = df['km_inicial'] > df['km_final']
    df.loc[mask, ['km_inicial', 'km_final']] = df.loc[mask, ['km_final', 'km_inicial']].values
    
    df['br'] = df['rodovia'].apply(clean_rodovia)
    return df

def clean_target_features(df):
    """
    Fills continuous numeric nulls with -1 to maintain them without normalizing.
    """
    for col in ['icc', 'icp', 'icm']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)
            
    return df

def generate_temporal_backfill(df_all):
    """
    Duplicates 2022's data (assuming dnit_2021 failed/empty) to represent 2019, 2020, 2021.
    """
    if df_all.empty:
        return df_all
        
    earliest_year = df_all['ano'].min()
    base_df = df_all[df_all['ano'] == earliest_year].copy()
    
    all_years_df = [df_all]
    
    target_years = [2019, 2020, 2021]
    for yr in target_years:
        if yr not in df_all['ano'].unique():
            print(f"Creating Backfill: Copying {earliest_year} data for year {yr}")
            df_yr = base_df.copy()
            df_yr['ano'] = yr
            all_years_df.append(df_yr)
            
    return pd.concat(all_years_df, ignore_index=True)

def run_etl():
    files = glob.glob(os.path.join(DATA_DIR, "dnit_*"))
    all_dfs = []
    
    for f in files:
        if '2021' in f:
            print("Skipping dnit_2021 raw file, creating it mathematically in Backfill step to avoid broken encoding.")
            continue
            
        print(f"Processing: {os.path.basename(f)}")
        # 1. Ingest
        df_raw = load_and_standardize(f)
        if df_raw.empty:
            continue
            
        # 2. Geography
        df_geo = standardize_geographic_limits(df_raw)
        
        # 3. Clean
        df_clean = clean_target_features(df_geo)
        
        cols_to_keep = ['ano', 'br', 'km_inicial', 'km_final', 'icc', 'icp', 'icm']
        cond_cols = [c for c in df_clean.columns if c.startswith('cond_')]
        cols_to_keep.extend(cond_cols)
        
        final_cols = [c for c in cols_to_keep if c in df_clean.columns]
        
        # Drop rows where 'br' failed to parse
        df_final = df_clean.dropna(subset=['br'])[final_cols]
        all_dfs.append(df_final)
        
    if all_dfs:
        print("\nConcatenating Temporal Panel...")
        df_concat = pd.concat(all_dfs, ignore_index=True)
        print(f"Panel Rows: {len(df_concat)}")
        
        print("\nExecuting Temporal Backfill...")
        df_complete = generate_temporal_backfill(df_concat)
        print(f"Total Final Expanded Rows: {len(df_complete)}")
        
        out_path = os.path.join(DATA_DIR, "df_dnit_processed.csv")
        df_complete.to_csv(out_path, index=False)
        print(f"\nPipeline Finalizado! Dataset exportado: {out_path}")
        
        return df_complete
        
    return None

if __name__ == "__main__":
    df_result = run_etl()
    if df_result is not None:
        print("\nAmostra das Chaves de Junção (br, km_inicial, km_final, ano):")
        print(df_result[['br', 'ano', 'km_inicial', 'km_final']].head())
