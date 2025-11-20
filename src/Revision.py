import argparse
import pandas as pd

def revisar(input_path):
    # Cargar dataset
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    print("Dataset cargado. Filas y columnas:", df.shape)

    # Revisar nulos
    hay_nulos = df.isnull().any().any()

    # Revisar duplicados
    hay_duplicados = df.duplicated().any()

    # Revisar vacíos 
    hay_vacios = (df.astype(str).apply(lambda x: x.str.strip() == "")).any().any()

    print("\n--- RESULTADOS ---")
    
    print("¿Hay nulos?:", "SI" if hay_nulos else "NO")
    print("¿Hay duplicados?:", "SI" if hay_duplicados else "NO")
    print("¿Hay vacíos?:", "SI" if hay_vacios else "NO")

    #el rsultado mostro que no habia ninguno de los tres entonces en preprocess solo eliminamos la columna innecesaria

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    revisar(args.input)