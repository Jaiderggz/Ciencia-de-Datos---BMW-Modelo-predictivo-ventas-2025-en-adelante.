import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
import numpy as np

df = pd.read_csv("data/processed/processed.csv")

X = df.drop("Sales_Classification", axis=1)
y = df["Sales_Classification"]

y_numeric = y.map({"Low": 0, "High": 1})

print("Distribución de clases (numérica):")
print(y_numeric.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_numeric, test_size=0.30, random_state=42, stratify=y_numeric
)

#  Preprocesamiento de datos

categorical_cols = ["Model", "Region", "Color", "Fuel_Type", "Transmission"]
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ],
    remainder="passthrough"
)


#  Pipeline SMOTE + UnderSampling + XGBoost

resampler = SMOTEENN(random_state=42)  
num_low = (y_train == 0).sum()
num_high = (y_train == 1).sum()
scale_pos_weight = num_low / num_high
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

model_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("resample", resampler),
    ("classifier", XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=scale_pos_weight
    ))
])


#  Entrenar modelo, es normal que demore dos o mas minutos

print("\nEntrenando modelo...")
model_pipeline.fit(X_train, y_train)

# Evaluacion del modelo

y_proba = model_pipeline.predict_proba(X_test)[:, 1]

threshold = 0.5
y_pred = (y_proba >= threshold).astype(int)

print("\n=== ACCURACY DEL MODELO ===")
print(round(accuracy_score(y_test, y_pred), 4))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=["Low", "High"]))


# 8. Guardar modelo
os.makedirs("models", exist_ok=True)
joblib.dump(model_pipeline, "models/model_bmw_balanced.joblib")
print("\nModelo guardado correctamente.")
