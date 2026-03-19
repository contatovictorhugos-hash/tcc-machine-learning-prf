import pandas as pd

def explore_csv(filepath):
    try:
        # Just sample to explore unique elements
        df = pd.read_csv(filepath, sep=',', encoding='utf-8')
        print("Successfully read with sep=',' and utf-8")
        
        print("\n--- INFO ---")
        df.info()

        print("\n--- UNIQUE TRACADO_VIA ---")
        if 'tracado_via' in df.columns:
            print(df['tracado_via'].value_counts(dropna=False))
        else:
            print("Column tracado_via not found")

    except Exception as e:
        print("Failed to read:", e)

if __name__ == "__main__":
    explore_csv("df_prf_consolidado_19-25.csv")
