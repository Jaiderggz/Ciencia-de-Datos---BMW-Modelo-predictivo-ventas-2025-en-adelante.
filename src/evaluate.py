import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# ======================================================
# 1. Cargar modelo guardado
# ======================================================
model_pipeline = joblib.load("models/model_bmw_balanced.joblib")

# ======================================================
# 2. Cargar datos de test
# ======================================================
X_test = pd.read_csv("data/processed/processed.csv")

y_test = X_test["Sales_Classification"].map({"Low": 0, "High": 1})
X_test = X_test.drop("Sales_Classification", axis=1)

# ======================================================
# 3. Predicciones
# ======================================================
# Probabilidad de High
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Usar el umbral óptimo del entrenamiento
best_threshold = 0.52
y_pred = (y_proba >= best_threshold).astype(int)

# ======================================================
# 4. Evaluación
# ======================================================
print("=== ACCURACY ===")
print(round(accuracy_score(y_test, y_pred), 4))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=["Low", "High"]))

# ======================================================
# 5. Guardar predicciones (opcional)
# ======================================================
X_test["Predicted_Class"] = y_pred
X_test.to_csv("data/test_predictions.csv", index=False)
print("\nPredicciones guardadas en 'data/test_predictions.csv'.")
