# -*- coding: latin-1 -*-
import torch

# Charger le checkpoint
checkpoint = torch.load('checkpoint6uslim.pth', map_location='cpu')

# Extraire le dictionnaire d'état
state_dict = checkpoint['model_state_dict']

# Afficher les caractéristiques de chaque paramètre
for key, tensor in state_dict.items():
    print(f"{key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")