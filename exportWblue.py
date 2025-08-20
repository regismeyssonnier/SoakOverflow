# -*- coding: latin-1 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def encode_weights_to_unicode_string(weights, offset=12.0, divider=2048.0):
	"""
	Encode un tableau numpy float en cha�ne unicode compress�e.
	"""
	print("min max:", weights.min(), weights.max())
	weights = np.clip(weights, -offset, offset)
	s = np.round(divider * (weights + offset)).astype(np.uint16)
	bytes_be = bytearray()
	s_flat = s.flatten()

	for val in s_flat:
		bytes_be.append((val >> 8) & 0xFF)
		bytes_be.append(val & 0xFF)
	unicode_str = bytes_be.decode('utf-16-be')
	return unicode_str


def decode_unicode_string_to_weights(unicode_str, offset=12.0, divider=2048.0, shape=None):
	"""
	D�code la cha�ne unicode compress�e en poids float numpy.
	"""
	bytes_be = unicode_str.encode('utf-16-be')
	arr_uint16 = np.frombuffer(bytes_be, dtype=np.uint16)
	weights = arr_uint16.astype(np.float32) / divider - offset
	if shape is not None:
		weights = weights.reshape(shape)

	# Affichage debug
	#print(f"Poids d�compress�s (shape={weights.shape}):")
	#print(weights)
	return weights


def export_weights_decompressed_to_txt(decompressed_weights_dict, output_txt='weights_decompressed.txt'):
	"""
	Enregistre les poids d�compress�s dans un fichier texte lisible.
	"""
	with open(output_txt, 'w', encoding='utf-8') as f:
		for name, weights in decompressed_weights_dict.items():
			f.write(f"### {name} (shape={weights.shape}) ###\n")
			weights_str = np.array2string(weights, precision=6, floatmode='fixed', max_line_width=120)
			f.write(weights_str + '\n\n')

	print(f"[o] Poids d�compress�s enregistr�s dans {output_txt}")


def export_torch_weights_to_unicode_python_file(model_class, checkpoint_path,
											   output_py='weights_unicode.py',
											   output_txt='weights_decompressed.txt'):
	"""
	Exporte les poids compress�s dans un fichier Python
	et enregistre les poids d�compress�s dans un fichier texte lisible.
	"""
	model = model_class()
	checkpoint = torch.load(checkpoint_path, map_location='cpu')
	model.load_state_dict(checkpoint['model_state_dict'])

	decompressed_weights_dict = {}

	with open(output_py, 'w', encoding='utf-8') as f:
		f.write("# Weights encoded as UTF-16BE unicode strings with offset=12, divider=2048\n\n")

		for name, param in model.named_parameters():
			np_array = param.detach().cpu().numpy().astype(np.float32)
			encoded_str = encode_weights_to_unicode_string(np_array)
			safe_name = name.replace('.', '_')
			shape_str = str(np_array.shape)

			f.write(f"{safe_name}_shapeb = {shape_str}\n")
			f.write(f"{safe_name}b = '''{encoded_str}'''\n\n")

			decompressed = decode_unicode_string_to_weights(encoded_str, shape=np_array.shape)
			decompressed_weights_dict[safe_name] = decompressed

	export_weights_decompressed_to_txt(decompressed_weights_dict, output_txt)

	print(f"[o] Poids compress�s export�s dans {output_py}")


# Exemple d�utilisation
if __name__ == "__main__":
	class ConvDQN(nn.Module):
		def __init__(self, in_channels=11, num_actions=5):
			super().__init__()
			# Petit réseau conv (adapte si besoin)
			self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
			self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
			self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
			self.pool  = nn.AdaptiveAvgPool2d(1)   # -> (B,64,1,1)
			self.fc1   = nn.Linear(16, 64)
			self.fc2   = nn.Linear(64, num_actions)

			nn.init.uniform_(self.fc2.weight, -0.1, 0.1)
			nn.init.constant_(self.fc2.bias, 0.0)

		def forward(self, x):
			# x: (B,C,H,W)
			x = F.relu(self.conv1(x))
			x = F.relu(self.conv2(x))
			x = F.relu(self.conv3(x))
			x = self.pool(x).view(x.size(0), -1)  # (B,64)
			x = F.relu(self.fc1(x))
			x = self.fc2(x)                    # (B, NUM_ACTIONS) = Q(s,·)
			return x


	# Remplace 'checkpoint6uslim.pth' par le chemin de ton checkpoint
	export_torch_weights_to_unicode_python_file(
		model_class=ConvDQN,
		checkpoint_path='checkpoint6uslim_dqnoneblue.pth',
		output_py='weights_unicodeblue.py',
		output_txt='weights_decompressed.txt'
	)
