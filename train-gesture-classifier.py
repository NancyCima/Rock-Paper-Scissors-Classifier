"""
Script para entrenar modelo de clasificación de gestos
"""

import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Para escala media=0, std=1
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset
try:
    X = np.load('data/rps_dataset_300.npy')
    y = np.load('data/rps_labels_300.npy')
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
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)


# Normalización
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# One-hot encoding para las etiquetas
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
y_val_cat = to_categorical(y_val)

# Construimos el modelo
# Este es un modelo simple, condos capas de 32 y 16 neuronas respectivamente. 
# Tiene activación de relu, y regularización y dropout para evitar el overfitting.
# La capa de salida tiene tres neuronas, representando las 3 clases
# y una activación de softmax.
model = Sequential([
    Input(shape=(42,)),
    GaussianNoise(0.01),  # Simula variaciones en landmarks

    # Capa 1 con regularización L2 (lambda=0.01)
    Dense(32, activation='relu',  
          kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    
    # Capa 2 con regularización
    Dense(16, activation='relu', 
          kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    
    # Capa de salida
    Dense(3, activation='softmax')
])

# Compilamos
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
            )

# Definimos EarlyStopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',   # Métrica a monitorear
    patience=3,           # Épocas sin mejora antes de detener
    restore_best_weights=True  # Restaura los mejores pesos encontrados
)

# Le daremos un peso un poco mayor a la clase "Tijera" ya que
# esta clase tiende a no precedirse correctamente
class_weights = {0: 1.0, 1: 1.0, 2: 1.5}

# Entrenamiento con EarlyStopping
history = model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val_cat),
    callbacks=[early_stop],
    class_weight=class_weights
)

# Evaluación
def evaluate_model():
    """
    Evalúa el modelo entrenado utilizando los datos de prueba.
    
    Calcula y muestra:
    - Pérdida, accuracy
    - Matriz de confusión visual.
    - Reporte de clasificación detallado (precisión, recall, F1-score por clase).
    
    Utiliza las variables globales `model`, `X_test`, `y_test`, y `y_test_cat`.
    """
    class_names = ['Piedra', 'Papel', 'Tijeras']

    # Generamos predicciones
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    # Obtenems métricas
    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    print(f"\nMétricas: ")
    print(f"- Pérdida: {test_loss:.4f}")
    print(f"- Accuracy: {test_acc*100:.2f}%")
    
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

# Creamos directorios si no existen
os.makedirs('models', exist_ok=True)

# Guardamos modelo y escalador
model.save('models/rps_model.keras')
joblib.dump(scaler, 'models/rps_scaler.pkl')

print("\nModelo y escalador guardados en la carpeta 'models'")