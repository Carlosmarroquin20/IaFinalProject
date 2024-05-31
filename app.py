from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import base64
import dlib
import re

app = Flask(__name__)

# Cargar el detector de caras de dlib y el predictor de forma
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Diccionario para traducir emociones al español
emotions_translation = {
    'angry': 'enojado',
    'disgust': 'disgustado',
    'fear': 'miedo',
    'happy': 'feliz',
    'sad': 'triste',
    'surprise': 'sorprendido',
    'neutral': 'neutral'
}

# Función para dibujar puntos faciales en la imagen
def draw_face_points(frame, face):
    landmarks = predictor(frame, face)
    for i in range(36, 48):  # Ojos
        cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (255, 0, 0), -1)
    for i in range(17, 27):  # Cejas
        cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (255, 0, 0), -1)
    for i in range(48, 68):  # Boca
        cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (255, 0, 0), -1)

# Función para detectar emociones en una imagen
def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convertir la imagen a escala de grises
    dlib_faces = detector(gray_frame) # Detectar caras en la imagen

    for face in dlib_faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)  # Dibujar un rectángulo alrededor de la cara
        draw_face_points(frame, face) # Dibujar puntos faciales
        face_roi = frame[y:y + h, x:x + w] # Extraer la región de interés (la cara)
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False) # Analizar la emoción
        dominant_emotion = result[0]['dominant_emotion'] # Obtener la emoción dominante
        emotion_confidence = result[0]['emotion'][dominant_emotion]  # Obtener el nivel de confianza
        emotion_spanish = emotions_translation[dominant_emotion] # Traducir la emoción al español
        label_emotion = f"{emotion_spanish}"
        label_confidence = f"{emotion_confidence:.2f}%"
        cv2.putText(frame, label_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Mostrar la emoción en la imagen
        cv2.putText(frame, label_confidence, (x + w - 100, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Mostrar el nivel de confianza

    return frame

# Función para decodificar una imagen en formato base64
def decode_base64_image(base64_string):
    img_str = re.search(r'base64,(.*)', base64_string).group(1)
    img_bytes = base64.b64decode(img_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

@app.route('/')
def index():
    return render_template('index.html') # Renderizar la plantilla HTML principal

@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    image_data = data['image']
    frame = decode_base64_image(image_data) # Decodificar la imagen recibida
    frame = detect_emotion(frame) # Detectar emociones en la imagen
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': frame_base64}) # Devolver la imagen procesada en formato base64

if __name__ == "__main__":
    app.run(debug=True)
