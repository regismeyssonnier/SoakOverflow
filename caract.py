# -*- coding: latin-1 -*-
import torch

# Charger le checkpoint
checkpoint = torch.load('checkpoint6uslim.pth', map_location='cpu')

# Extraire le dictionnaire d'�tat
state_dict = checkpoint['model_state_dict']

# Afficher les caract�ristiques de chaque param�tre
for key, tensor in state_dict.items():
    print(f"{key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")