# -*- coding: latin-1 -*-
import torch
import numpy as np
from conv2d import *
import torch.nn as nn
import torch.nn.functional as F

def encode_weights_to_unicode_string(weights, offset=12.0, divider=2048.0):
	weights = np.clip(weights, -offset, offset)
	s = np.round(divider * (weights + offset)).astype(np.uint16)
	bytes_be = bytearray()
	s_flat = s.flatten()

	for val in s_flat:
		bytes_be.append((val >> 8) & 0xFF)
		bytes_be.append(val & 0xFF)
	unicode_str = bytes_be.decode('utf-16-be')
	return unicode_str

def decode_unicode_string_to_weightsO(unicode_str, offset=12.0, divider=2048.0, shape=None):
	bytes_be = unicode_str.encode('utf-16-be')
	arr_uint16 = np.frombuffer(bytes_be, dtype=np.uint16)
	weights = arr_uint16.astype(np.float32) / divider - offset
	if shape is not None:
		weights = weights.reshape(shape)
	return weights

import numpy as np

def decode_unicode_string_to_weights(unicode_str, offset=12.0, divider=2048.0, shape=None):
	# Étape 1 : reconstruire la chaîne binaire 'weights_bytes' comme en C++ wstring -> string
	weights_bytes = bytearray()
	for c in unicode_str:
		val = ord(c)
		weights_bytes.append((val >> 8) & 0xFF)  # octet haut
		weights_bytes.append(val & 0xFF)         # octet bas

	# Étape 2 : lire les poids 2 octets par 2 octets, big-endian
	size = len(weights_bytes) // 2
	output = []
	for i in range(size):
		s1 = weights_bytes[2*i]
		s2 = weights_bytes[2*i + 1]
		s = (s1 << 8) + s2
		val = (s / divider) - offset
		output.append(val)

	# Étape 3 : si shape précisé, reshape en numpy array
	if shape is not None:
		import numpy as np
		output = np.array(output, dtype=np.float32).reshape(shape)
	else:
		output = list(output)

	return output


def load_weights_from_wstring_py(w0, size, offset=12.0, divider=2048.0):
	# Étape 1 : construction de la string binaire 'weights' (comme en C++ wstring -> string)
	weights_bytes = bytearray()
	for c in w0:
		val = ord(c)
		weights_bytes.append((val >> 8) & 0xFF)  # c >> 8
		weights_bytes.append(val & 0xFF)         # c & 255

	# Étape 2 : lecture 2 octets à 2 octets + conversion en float
	output = []
	for i in range(size):
		s1 = weights_bytes[2*i]
		s2 = weights_bytes[2*i + 1]
		s = (s1 << 8) + s2  # big-endian reconstitué comme en C++
		val = (s / divider) - offset
		output.append(val)
	return output


def export_torch_weights_to_unicode_python_file(model_class, checkpoint_path, output_py='weights_unicode.py'):
	model = model_class()
	checkpoint = torch.load(checkpoint_path, map_location='cpu')
	model.load_state_dict(checkpoint['model_state_dict'])

	with open(output_py, 'w', encoding='utf-8') as f:
		f.write("# Weights encoded as UTF-16BE unicode strings with offset=12, divider=2048\n\n")
		for name, param in model.named_parameters():
			np_array = param.detach().cpu().numpy().astype(np.float32)
			encoded_str = encode_weights_to_unicode_string(np_array)
			safe_name = name.replace('.', '_')
			shape_str = str(np_array.shape)

			# Write shape and encoded string
			f.write(f"{safe_name}_shape = {shape_str}\n")
			f.write(f"{safe_name} = '''{encoded_str}'''\n\n")

	print(f"[o] Weights exported as compressed unicode strings to {output_py}")

# Exemple modèle PyTorch simplifié
class PolicyNet(nn.Module):
	def __init__(self, num_players=10, num_actions=5):
		super().__init__()
		self.conv1 = nn.Conv2d(83, 8, 3, padding=1)
		self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
		self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
		self.relu = nn.ReLU()
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(16, num_players * num_actions)
		self.num_players = num_players
		self.num_actions = num_actions

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.pool(x).view(x.size(0), -1)
		return self.fc(x).view(-1, self.num_players, self.num_actions)

# Fonction pour vérifier les poids
def check_weights_export_and_importO(model_class, checkpoint_path):
	# Charger modèle PyTorch original
	model = model_class()
	checkpoint = torch.load(checkpoint_path, map_location='cpu')
	model.load_state_dict(checkpoint['model_state_dict'])

	model2 = PolicyNet_Numpy(num_players=10, num_actions=5)
	load_pytorch_weights_into_numpy_model('checkpoint6uslim.pth', model2)

	# Exporter
	#export_torch_weights_to_unicode_python_file(model_class, checkpoint_path, output_py='weights_unicode.py')

	# Charger le fichier poids exporté comme module Python
	import weights_unicode

	max_diff = 0.0
	for name, param in model.named_parameters():
		safe_name = name.replace('.', '_')
		orig = param.detach().cpu().numpy()
		shape = getattr(weights_unicode, f"{safe_name}_shape")
		encoded_str = getattr(weights_unicode, safe_name)

		decoded = decode_unicode_string_to_weights(encoded_str, shape=shape)

		diff = np.max(np.abs(orig - decoded))
		print(f"{name}: max abs diff = {diff:.6f}")
		max_diff = max(max_diff, diff)

		# Affichage exemple sur un petit extrait
		print(f"Orig {name} sample:\n{orig.flatten()[:5]}")
		print(f"Decoded {name} sample:\n{decoded.flatten()[:5]}")

	print(f"\nMax absolute difference across all params: {max_diff:.6f}")

import torch
import numpy as np

def check_weights_export_and_import(model_class, checkpoint_path):
	# Charger modèle PyTorch original
	model = model_class()
	checkpoint = torch.load(checkpoint_path, map_location='cpu')
	model.load_state_dict(checkpoint['model_state_dict'])

	# Charger modèle numpy (ton modèle alternatif avec poids chargés)
	model2 = PolicyNet_Numpy(num_players=10, num_actions=5)
	load_pytorch_weights_into_numpy_model(checkpoint_path, model2)  # Assure-toi que cette fonction existe
	   
	max_diff_models = 0.0
	print("Comparaison entre modèle PyTorch original et modèle numpy chargé:")
	# Attention ici, les noms de params peuvent être différents ou model2 pas torch.Tensor
	# On suppose que model2 a des attributs numpy arrays nommés comme model (adapté)
	for name, param in model.named_parameters():
		orig = param.detach().cpu().numpy()

		# Récupérer le même poids dans model2 (attention à l'attribut exact)
		# Exemple si model2.conv1.weight numpy array
		attr = model2
		for attr_part in name.split('.'):
			attr = getattr(attr, attr_part)

		numpy_weight = attr
		diff = np.max(np.abs(orig - numpy_weight))
		print(f"{name}: max abs diff = {diff:.6f}")
		max_diff_models = max(max_diff_models, diff)

		print(f"Orig {name} sample:\n{orig.flatten()[:5]}")
		print(f"Numpy {name} sample:\n{numpy_weight.flatten()[:5]}")

	print(f"\nMax absolute difference (original vs numpy model): {max_diff_models:.6f}")



if __name__ == "__main__":
	checkpoint_path = 'checkpoint6uslim.pth'  # adapte ton chemin
	check_weights_export_and_import(PolicyNet, checkpoint_path)
