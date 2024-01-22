import cv2
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.jit import load
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

pad = 20

# Charger le modèle sauvegardé
model_path = 'sauvegarde(1).pt'
model = load(model_path, map_location=torch.device('cpu'))
model.eval()

image_transform = transforms.Compose([
    transforms.Resize((128, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[0.5])
])

# Permet de crop RGB-Camera
def crop_hand_mp(frame): 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0].landmark

        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')

        for landmark in hand_landmarks:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            x_min = min(x_min, x) 
            y_min = min(y_min, y) 
            x_max = max(x_max, x) 
            y_max = max(y_max, y)

        if x_max > x_min and y_max > y_min:
            hand_cropped = frame[y_min-pad:y_max+pad, x_min-pad:x_max+pad]
            return hand_cropped
    return frame

def recognize_gesture(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0)  

    with torch.no_grad():
        output = model(image_tensor)

    _, predicted = torch.max(output.data, 1)
    
    os.remove(image_path)

    return predicted.item()

# Fonction pour capturer une seule image, l'enregistrer et appliquer le modèle
def capture_and_predict_gesture():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        exit()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Erreur: Impossible de capturer une image.")
                exit()

            # Crop la main de l'image
            cropped_frame = crop_hand_mp(frame)

            cropped_frame = cv2.resize(cropped_frame, (frame.shape[1], frame.shape[0]))

            concatenated_frame = cv2.hconcat([frame, cropped_frame])





            # Afficher l'image avec la main recadrée
            cv2.imshow('Capture Real-Life Gesture', concatenated_frame)

            # Enregistrer l'image recadrée
            image_path = os.path.join(os.getcwd(), 'captured_image.jpg')
            cv2.imwrite(image_path, cropped_frame)

            # Appliquer le modèle de reconnaissance de gestes
            predicted_label = recognize_gesture(image_path)
            print(f"Predicted Gesture Label: {predicted_label}")

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    finally:
        # Libérer la capture
        cap.release()
        cv2.destroyAllWindows()

# Appeler la fonction pour démarrer le programme
capture_and_predict_gesture()

