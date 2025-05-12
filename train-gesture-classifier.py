"""
Script para entrenar modelo de clasificación de gestos
"""

import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Para escala media=0, std=1
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset
try:
    X = np.load('data/rps_dataset.npy')
    y = np.load('data/rps_labels.npy')
    assert len(X) > 0, "El dataset está vacío"
    assert len(X) == len(y), "Datos y etiquetas no coinciden"
except Exception as e:
    print("Error: No se encontraron los archivos del dataset. Ejecuta primero record-dataset.py")
    print(f"Error: {str(e)}")
    exit()

# Preprocesamiento

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)


# Normalización
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encoding para las etiquetas
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Construir modelo
'''
model = Sequential([
    #El dropout y la regularización L2 ayudan a prevenir el sobreajuste.
    Dense(64, activation='relu', input_shape=(42,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

'''
model = Sequential([
    # Capa 1 con regularización L2 (lambda=0.001)
    Dense(32, activation='relu', 
          input_shape=(42,), 
          kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    
    # Capa 2 con regularización
    Dense(16, activation='relu', 
          kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    
    # Capa de salida
    Dense(3, activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])

# Definir EarlyStopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',   # Métrica a monitorear
    patience=5,           # Épocas sin mejora antes de detener
    restore_best_weights=True  # Restaura los mejores pesos encontrados
)

# Entrenamiento con EarlyStopping
history = model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stop]
)

# Evaluación
def evaluate_model():
    """
    Evalúa el modelo entrenado utilizando los datos de prueba.
    
    Calcula y muestra:
    - Pérdida, accuracy, precisión y recall.
    - Matriz de confusión visual.
    - Reporte de clasificación detallado (precisión, recall, F1-score por clase).
    
    Utiliza las variables globales `model`, `X_test`, `y_test`, y `y_test_cat`.
    """
    class_names = ['Piedra', 'Papel', 'Tijeras']

    # Generar predicciones
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    # Obtener métricas
    test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test_cat)
    print(f"\nMétricas: ")
    print(f"- Pérdida: {test_loss:.4f}")
    print(f"- Accuracy: {test_acc*100:.2f}%")
    print(f"- Precisión (por clase): {test_prec*100:.2f}%")
    print(f"- Recall: {test_rec*100:.2f}%")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.show()
    
    # Reporte de clasificación
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=class_names))

evaluate_model()

# Visualización del entrenamiento
def plot_training():
    """
    Genera gráficos de la precisión y pérdida durante el entrenamiento.
    
    Muestra dos subplots:
    1. Precisión en entrenamiento vs. validación por época.
    2. Pérdida en entrenamiento vs. validación por época.
    
    Utiliza el historial guardado en `history.history`.
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión durante Entrenamiento')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida durante Entrenamiento')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training()

# Crear directorios si no existen
os.makedirs('models', exist_ok=True)

# Guardar modelo y escalador
model.save('models/rps_model.h5')
joblib.dump(scaler, 'models/rps_scaler.pkl')

print("\nModelo y escalador guardados en la carpeta 'models'")