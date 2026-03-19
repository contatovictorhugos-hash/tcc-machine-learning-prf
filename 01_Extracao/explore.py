import pandas as pd
import glob

files = glob.glob("c:/Users/yuri/projetcs/dnit_project/dnit_*")
for f in files:
    print(f"--- {f} ---")
    try:
        if f.endswith('.csv'):
            df = pd.read_csv(f, nrows=2, sep=None, engine='python')
        else:
            df = pd.read_excel(f, nrows=2)
        print("Columns:", df.columns.tolist())
        if not df.empty:
            print("First row:", df.iloc[0].to_dict())
    except Exception as e:
        print("Error reading file:", e)
    print("\n")
