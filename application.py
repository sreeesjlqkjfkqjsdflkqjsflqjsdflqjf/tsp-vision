import torch

device = torch.device('cpu')
modele = torch.jit.load('sauvegarde.pt', map_location=device)
modele.eval()
modele.forward()
