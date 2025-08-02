# -*- coding: latin-1 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def export_torch_weights_to_numpy_python_file(model_class, checkpoint_path, output_py='weights_numpy.py'):
    # 1. Instancier le modèle
    model = model_class()
    
    # 2. Charger les poids du checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 3. Ouvrir le fichier pour écrire les poids
    with open(output_py, 'w') as f:
        f.write("import numpy as np\n\n")
        for name, param in model.named_parameters():
            np_array = param.detach().cpu().numpy()
            # Format avec 3 chiffres après la virgule
            np_repr = np.array2string(
                np_array,
                separator=', ',
                precision=3,
                floatmode='fixed',
                threshold=np.inf  # pour ne pas tronquer
            )
            safe_name = name.replace('.', '_')  # e.g., conv1.weight -> conv1_weight
            f.write(f"{safe_name} = np.array({np_repr}, dtype=np.float32)\n\n")


    print(f"[o] Weights exported to {output_py}")

class PolicyNet(nn.Module):
	def __init__(self, num_players=10, num_actions=5):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(83, 8, 3, padding=1), nn.ReLU(),
			nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
			nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
		)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(16, num_players * num_actions)
		self.num_players = num_players
		self.num_actions = num_actions

	def forward(self, x):
		x = self.pool(self.conv(x)).view(x.size(0), -1)
		return self.fc(x).view(-1, self.num_players, self.num_actions)


export_torch_weights_to_numpy_python_file(
    model_class=PolicyNet,
    checkpoint_path='checkpoint6uslim.pth',
    output_py='weights_numpy.py'
)
