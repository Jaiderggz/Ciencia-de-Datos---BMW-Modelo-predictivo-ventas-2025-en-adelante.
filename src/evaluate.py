import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Cargar modelo 
model_pipeline = joblib.load("models/model_bmw_balanced.joblib")

# Cargar datos de test
X_test = pd.read_csv("data/processed/processed.csv")

# Extraer Y real
y_test = X_test["Sales_Classification"].map({"Low": 0, "High": 1})

# Eliminar la real de X para no contaminar
X_test = X_test.drop("Sales_Classification", axis=1)

# Predicciones
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

best_threshold = 0.52
y_pred = (y_proba >= best_threshold).astype(int)

# Evaluación
print("=== ACCURACY ===")
print(round(accuracy_score(y_test, y_pred), 4))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=["Low", "High"]))

# Guardar predicciones + y real
X_test["Real_Class"] = y_test
X_test["Predicted_Class"] = y_pred
X_test["Probability_High"] = y_proba

X_test.to_csv("data/test_predictions.csv", index=False)
print("\nPredicciones guardadas en 'data/test_predictions.csv'.")
