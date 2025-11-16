import argparse
import pandas as pd

def preprocess(input_path, output_path):
    # Detectar dataset
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    print("Dataset cargado. Filas y columnas:", df.shape)
    print(df.head())

    # limpieza simple 
    
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    df = df.fillna("Unknown")
    df = df.drop(columns=["Sales_Volume"])


    # guardar dataset limpio 
    df.to_csv(output_path, index=False)
    print("Procesado guardado en:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    preprocess(args.input, args.output)
