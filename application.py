import cv2
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.jit import load

# Charger le modèle sauvegardé
model_path = 'sauvegarde_cpu.pt'
model = load(model_path, map_location=torch.device('cpu'))
model.eval()

# Définir la transformation de prétraitement des images
image_transform = transforms.Compose([
    transforms.Resize((128, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[0.5])
])

# Fonction pour la reconnaissance des gestes sur une image donnée
def recognize_gesture(image_path):
    # Lire et prétraiter l'image
    image = Image.open(image_path).convert('L')
    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Ajouter une dimension de lot

    # Effectuer l'inférence
    with torch.no_grad():
        output = model(image_tensor)

    # Obtenir l'étiquette prédite
    _, predicted = torch.max(output.data, 1)
    
    # Supprimer l'image une fois que la prédiction est terminée
    os.remove(image_path)

    return predicted.item()

# Fonction pour capturer une seule image, l'enregistrer et appliquer le modèle
def capture_and_predict_gesture():
    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)

    # Vérifier si la capture est réussie
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        exit()

    try:
        # Laisser la caméra ouverte indéfiniment
        while True:
            # Capturer une seule image
            ret, frame = cap.read()

            # Vérifier si la capture est réussie
            if not ret:
                print("Erreur: Impossible de capturer une image.")
                exit()

            # Afficher la fenêtre de la caméra
            cv2.imshow('Capture Real-Life Gesture', frame)

            # Enregistrer l'image dans le dossier du répertoire actuel
            image_path = os.path.join(os.getcwd(), 'captured_image.jpg')
            cv2.imwrite(image_path, frame)

            # Appliquer le modèle pour prédire le label
            predicted_label = recognize_gesture(image_path)
            print(f"Predicted Gesture Label: {predicted_label}")

            # Attendre 10 millisecondes entre chaque image capturée
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    finally:
        # Libérer la capture
        cap.release()
        cv2.destroyAllWindows()

# Appeler la fonction pour démarrer le programme
capture_and_predict_gesture()


