import torch

modele = torch.jit.load('sauvegarde.pt')
modele.eval()
