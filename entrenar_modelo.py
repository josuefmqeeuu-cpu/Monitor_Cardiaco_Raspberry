import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Entrenando modelo de IA para Arritmias...")

# 1. Generar datos sinteticos (Simulamos pacientes)
# Clase 0: Normal (BPM 60-100, Voltaje estable)
bpm_normal = np.random.uniform(60, 100, 1000)
volt_normal = np.random.uniform(0.8, 1.5, 1000)
X_normal = np.column_stack((bpm_normal, volt_normal))
y_normal = np.zeros(1000)

# Clase 1: Arritmia/Taquicardia (BPM > 100 o < 60, Voltaje inestable)
bpm_arr = np.concatenate([np.random.uniform(101, 180, 500), np.random.uniform(30, 59, 500)])
volt_arr = np.random.uniform(0.2, 2.5, 1000)
X_arr = np.column_stack((bpm_arr, volt_arr))
y_arr = np.ones(1000)

# Juntar todo
X = np.concatenate((X_normal, X_arr))
y = np.concatenate((y_normal, y_arr))

# 2. Entrenar
clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(X, y)

# 3. Guardar
joblib.dump(clf, 'cerebro_ecg.joblib')
print("LISTO: Archivo 'cerebro_ecg.joblib' creado exitosamente.")
