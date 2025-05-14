"""
Script para clasificación en tiempo real de gestos:
Tomará como entrada la imagen de la cámara web y utilizando MediaPipe detecta los landmarks de la mano.
Luego, clasifica el gesto en "piedra", "papel" o "tijeras" utilizando el modelo entrenado.
Muestra el gesto reconocido en pantalla o "Gesto no detectado" en caso de que la clasificación
se haga con una confianza menor a 0.6
"""

import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from tensorflow.keras.models import load_model

# Cargamos modelo entrenado y el scaler utilizado
model = load_model('models/rps_model.keras')
scaler = joblib.load('models/rps_scaler.pkl') 
class_names = ['Piedra', 'Papel', 'Tijeras']

# Configuramos estilos de dibujo de los landmarks
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Configuramos el detector de MediaPipe
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1,
                                       min_hand_detection_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)

# Definimos las conexiones entre landmarks (MediaPipe Hands)
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# Configuramos la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertimos el frame a formato MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detección de landmarks
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        # Convertimos landmarks a NormalizedLandmarkList
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
            for lm in detection_result.hand_landmarks[0]
        ])
        
        # Dibujamos los landmarks en pantalla
        annotated_image = np.copy(rgb_frame)
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),  # Puntos verdes
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)   # Líneas azules
        )
        
        # Convertimos de vuelta a BGR para OpenCV
        frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        # Extraemos los landmarks para predicción
        landmarks = [lm.x for lm in detection_result.hand_landmarks[0]] + [lm.y for lm in detection_result.hand_landmarks[0]]

        # Validamos los landmarks
        if len(landmarks) != 42:
            continue  # Omitir frame inválido
        
        # Preprocesamos y predecimos
        input_data = scaler.transform(np.array([landmarks]).reshape(1, -1)) #Normalizamos los landmarks antes de predecir
        prediction = model.predict(input_data, verbose=0)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

        # Mostramos resultado
        # Si el mayor valor de confianza es mayor a 0.6, le otorgo esa etiqueta
        if confidence > 0.6:
            label = f"{class_names[class_id]} ({confidence*100:.1f}%)"
            color = (0, 255, 0)  # Verde
        # Si es menor a 0.6, considero que el gesto no se reconocio correctamente.
        else:
            label = "Gesto no detectado"
            color = (0, 0, 255)  # Rojo

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Clasificador de Gestos', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()