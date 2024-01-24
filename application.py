import cv2
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.jit import load

# Charger le modèle sauvegardé
model_path = 'sauvegarde.pt'
model = load(model_path, map_location=torch.device('cpu'))

print(model)
exit
model.eval()


dic = ["ciseau", "feuille", "pierre"]
dic_ot = ["palm", "one", "fist", "fist_moved",
          "thumb", "index", "ok", "palm_moved", "c", "down"]


image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def recognize_gesture(image_path):

    image = Image.open(image_path)
    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    # image_tensor = image_tensor.to(device)

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
        #
        while True:

            ret, frame = cap.read()

            if not ret:
                print("Erreur: Impossible de capturer une image.")
                exit()

            cv2.imshow('Capture Real-Life Gesture', frame)

            image_path = os.path.join(os.getcwd(), 'captured_image.jpg')
            cv2.imwrite(image_path, frame)

            predicted_label = dic[recognize_gesture(image_path)]

            print(f"Predicted Gesture Label: {predicted_label}")

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    finally:
        # Libérer la capture
        cap.release()
        cv2.destroyAllWindows()


# Appeler la fonction pour démarrer le programme
capture_and_predict_gesture()
