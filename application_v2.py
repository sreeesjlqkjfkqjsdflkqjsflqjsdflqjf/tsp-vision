import cv2
import os
import PIL
import torch
from torchvision import transforms
from torch.jit import load

# Charger le modèle sauvegardé
model_path = 'sauvegarde.pt'
model = load(model_path, map_location=torch.device('cpu'))
model.eval()


liste_labels = ["ciseau", "feuille", "pierre"]
dic_ot = ["palm", "one", "fist", "fist_moved",
          "thumb", "index", "ok", "palm_moved", "c", "down"]


image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main(model):
    was_training = model.training
    model.eval()

    cap = cv2.VideoCapture(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

            cv2.imshow('PIERRE-FEUILLE-CISEAU', frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame)
            img = image_transform(img)
            img = img.unsqueeze(0)
            img = img.to(device)

            with torch.no_grad():
                outputs = model(img)
                _, preds = torch.max(outputs, 1)

                print(f'Prédiction: {liste_labels[preds[0]]}')

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            model.train(mode=was_training)
    finally:
        # Libérer la capture
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(model)
