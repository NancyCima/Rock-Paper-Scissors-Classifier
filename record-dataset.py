"""
Grabación del dataset:
Script para capturar y almacenar landmarks de gestos de mano (piedra, papel, tijeras)
- Carga un dataset existente o crea uno nuevo
- Usar teclas 0, 1, 2 para grabar muestras (0: piedra, 1: papel, 2: tijeras)
- Presionar 'q' para salir y guardar
"""
# Configuración inicial para silenciar logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silencia TensorFlow
os.environ['GLOG_minloglevel'] = '3'      # Silencia MediaPipe
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Silencia absl

# Importaciones
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuramos el directorio de datos
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# Obtenemos las rutas de los datsets si existen
dataset_path = os.path.join(DATA_DIR, 'rps_dataset_300.npy')
labels_path = os.path.join(DATA_DIR, 'rps_labels_300.npy')

# Pedimos el modo de operación al usuario
print("\nOpciones:")
print("1. Crear nuevo dataset")
print("2. Continuar con dataset existente")
choice = input("Seleccione una opción (1 o 2): ")

if choice == '2' and os.path.exists(dataset_path) and os.path.exists(labels_path):
    # Cargamos datos existentes
    dataset = np.load(dataset_path).tolist()
    labels = np.load(labels_path).tolist()
    print(f"\nDataset cargado: {len(dataset)} muestras existentes\n")
else:
    if choice == '2' and (not os.path.exists(dataset_path) or not os.path.exists(labels_path)):
        print(f"\nNo hay datasets existentes en {dataset_path} o en {labels_path}")
    # Inicializamos nuevos datasets
    dataset = []
    labels = []
    print("\nCreando nuevo dataset\n")

total_samples = len(dataset)  # Contador total de muestras
new_samples = 0 # Contador de muestras nuevas

# Configuración de MediaPipe Hand Landmarker
base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task',
    delegate=python.BaseOptions.Delegate.CPU  # Evita warnings de GPU
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE  # Modo de ejecución
)
detector = vision.HandLandmarker.create_from_options(options)

# Configuramos la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Copia del frame original para procesar
    display_frame = frame.copy()

    # Convertimos el frame a formato MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detección de landmarks
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # Extraemos coordenadas normalizadas (x,y)
        landmarks = [lm.x for lm in hand_landmarks] + [lm.y for lm in hand_landmarks]
        
        # Mostramos las clases y la cantidad de muestras en pantalla
        info_text = [
            "0: Piedra | 1: Papel | 2: Tijeras",
            f"Muestras totales: {total_samples + new_samples}", # Muestra el total acumulado
            f"Nuevas muestras: {new_samples}" # Muestra solo las de esta sesión
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(display_frame, text, (10, 30 + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Mostramos el frame para la carga de muestras
    cv2.imshow('Dataset Collector', display_frame)
    
    # Manejo de teclas
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key in [ord('0'), ord('1'), ord('2')]:
        if detection_result.hand_landmarks:
            label = int(chr(key))
            labels.append(label)
            dataset.append(landmarks)
            new_samples += 1 
            print(f"\nMuestra {total_samples + new_samples}: {label}")

# Guardamos dataset combinado
if dataset:
    final_dataset = np.array(dataset)
    final_labels = np.array(labels)
    
    # Guardamos los datsets
    np.save(dataset_path, final_dataset)
    np.save(labels_path, final_labels)
    print(f"\nDataset guardado en {DATA_DIR}:")

    # Cantidad de datos totales
    print(f"- Muestras totales: {len(final_dataset)}")
    # Cantidad de datos por clase
    class_names = {0: 'Piedra', 1: 'Papel', 2: 'Tijeras'}
    unique, counts = np.unique(final_labels, return_counts=True)
    distribution = ', '.join([f"{class_names[cls]}: {count}" for cls, count in zip(unique, counts)])
    print(f"- Distribución de clases: {distribution}")
else:
    print("\nNo se guardaron nuevas muestras")

# Liberamos recursos
cap.release()
cv2.destroyAllWindows()