# -*- coding: latin-1 -*-
import numpy as np
import torch

class Conv2D_Numpy:
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='valid', dilation=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.padding = padding
        self.use_bias = bias

        kh, kw = self.kernel_size
        # dilation effect on kernel size
        self.kh_dil = self.dilation[0] * (kh - 1) + 1
        self.kw_dil = self.dilation[1] * (kw - 1) + 1

        fan_in = in_channels * kh * kw
        scale = np.sqrt(1. / fan_in)
        self.weight = np.random.uniform(-scale, scale,
                                        (out_channels, in_channels, kh, kw))
        self.bias = np.random.uniform(-scale, scale, out_channels) if bias else None

    def pad_input(self, x, pad_top, pad_bottom, pad_left, pad_right):
        return np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

    def im2col(self, x, out_h, out_w):
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation

        # Calculate the size of each patch with dilation
        kh_dil = self.kh_dil
        kw_dil = self.kw_dil

        cols = np.zeros((N, C, kh, kw, out_h, out_w), dtype=x.dtype)

        for y in range(kh):
            y_max = y * dh + sh * out_h
            for x_ in range(kw):
                x_max = x_ * dw + sw * out_w
                cols[:, :, y, x_, :, :] = x[:, :, y * dh:y_max:sh, x_ * dw:x_max:sw]

        # Rearrange so that each patch is flattened in last dim
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)  # shape (N*out_h*out_w, C*kh*kw)
        return cols

    def forward(self, x):
        N, C, H, W = x.shape
        sh, sw = self.stride

        # Compute output size and padding
        if self.padding == 'same':
            out_h = int(np.ceil(H / sh))
            out_w = int(np.ceil(W / sw))

            pad_h = max((out_h - 1) * sh + self.kh_dil - H, 0)
            pad_w = max((out_w - 1) * sw + self.kw_dil - W, 0)

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            x_padded = self.pad_input(x, pad_top, pad_bottom, pad_left, pad_right)
        elif self.padding == 'valid':
            out_h = (H - self.kh_dil) // sh + 1
            out_w = (W - self.kw_dil) // sw + 1
            x_padded = x
        else:
            raise ValueError("Only 'same' or 'valid' padding supported")

        # Extract patches
        col = self.im2col(x_padded, out_h, out_w)  # (N*out_h*out_w, C*kh*kw)
        # Reshape weights to (out_channels, C*kh*kw)
        weight_col = self.weight.reshape(self.out_channels, -1)  # (out_channels, C*kh*kw)

        # Matrix multiplication + bias
        out = col @ weight_col.T  # shape (N*out_h*out_w, out_channels)
        if self.use_bias:
            out += self.bias

        # Reshape output to (N, out_channels, out_h, out_w)
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

        return out


   

def adaptive_avg_pool2d_numpy(x, output_size):
    """
    Simule torch.nn.AdaptiveAvgPool2d en NumPy.
    
    Args:
        x: Tensor NumPy de forme (N, C, H_in, W_in)
        output_size: int ou tuple (H_out, W_out)
    
    Returns:
        Tensor NumPy de forme (N, C, H_out, W_out)
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    H_out, W_out = output_size

    N, C, H_in, W_in = x.shape
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

    for i in range(H_out):
        h_start = int(np.floor(i * H_in / H_out))
        h_end = int(np.ceil((i + 1) * H_in / H_out))
        for j in range(W_out):
            w_start = int(np.floor(j * W_in / W_out))
            w_end = int(np.ceil((j + 1) * W_in / W_out))
            patch = x[:, :, h_start:h_end, w_start:w_end]
            out[:, :, i, j] = patch.mean(axis=(2, 3))
    
    return out

import numpy as np

class BatchNorm1d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.gamma = np.ones(num_features) if affine else None
        self.beta = np.zeros(num_features) if affine else None

        self.running_mean = np.zeros(num_features) if track_running_stats else None
        self.running_var = np.ones(num_features) if track_running_stats else None

        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x):
        x = np.asarray(x)
        if x.shape[-1] != self.num_features:
            raise ValueError(f"Expected last dimension to be {self.num_features}, got {x.shape[-1]}")

        # Calcul mean et var sur tous les axes sauf le dernier (feature)
        axes = tuple(i for i in range(x.ndim - 1))  # ex: (0,) pour 2D, (0,1) pour 3D

        mean = np.mean(x, axis=axes, keepdims=True)  # shape compatible pour broadcast
        var = np.var(x, axis=axes, ddof=0, keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + self.eps)

        if self.affine:
            # gamma et beta doivent être broadcastables sur x_hat
            # gamma, beta ont shape (C,), on reshape en (1, 1, ..., C) selon ndim de x
            shape = [1] * x.ndim
            shape[-1] = self.num_features
            gamma = self.gamma.reshape(shape)
            beta = self.beta.reshape(shape)
            x_hat = x_hat * gamma + beta

        return x_hat


    def __call__(self, x):
        return self.forward(x)


import numpy as np

class BatchNorm2d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Paramètres appris (gamma et beta)
        self.gamma = np.ones((1, num_features, 1, 1)) if affine else None
        self.beta = np.zeros((1, num_features, 1, 1)) if affine else None

        # Moyenne et variance courantes (estimées pendant l'entraînement)
        self.running_mean = np.zeros((1, num_features, 1, 1)) if track_running_stats else None
        self.running_var = np.ones((1, num_features, 1, 1)) if track_running_stats else None

        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.num_features:
            raise ValueError(f"Expected input of shape (N, {self.num_features}, H, W), got {x.shape}")

        # Moyenne et variance sur (N, H, W) pour chaque canal C
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)  # shape (1, C, 1, 1)
        var = np.var(x, axis=(0, 2, 3), ddof=0, keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + self.eps)

        if self.affine:
            # reshape pour broadcast (C,) → (1, C, 1, 1)
            gamma = self.gamma.reshape(1, -1, 1, 1)
            beta = self.beta.reshape(1, -1, 1, 1)
            x_hat = x_hat * gamma + beta

        return x_hat


    def __call__(self, x):
        return self.forward(x)


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias_enabled = bias
        
        # Initialisation uniforme U(-k, k) avec k = 1 / sqrt(in_features)
        k = 1 / np.sqrt(in_features)
        self.weight = np.random.uniform(-k, k, size=(out_features, in_features)).astype(np.float32)
        if bias:
            self.bias = np.random.uniform(-k, k, size=(out_features,)).astype(np.float32)
        else:
            self.bias = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x shape: (..., in_features)
        y = x @ self.weight.T  # shape: (..., out_features)
        if self.bias_enabled:
            y += self.bias
        return y

import numpy as np

class PolicyNet_Numpy:
    def __init__(self, num_players=5, num_actions=5):
        self.num_players = num_players
        self.num_actions = num_actions

        self.conv1 = Conv2D_Numpy(in_channels=93, out_channels=8, kernel_size=3, padding='same')
        self.conv2 = Conv2D_Numpy(in_channels=8, out_channels=16, kernel_size=3, padding='same')
        self.conv3 = Conv2D_Numpy(in_channels=16, out_channels=16, kernel_size=3, padding='same')

        self.fc1 = Linear(in_features=16, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=128)
        self.fc3 = Linear(in_features=128, out_features=num_players * num_actions)

    def relu(self, x):
        #return np.maximum(0, x)
        return np.tanh(x)

    def eval(self):
        pass

    def forward(self, x):
        x = self.relu(self.conv1.forward(x))
        x = self.relu(self.conv2.forward(x))
        x = self.relu(self.conv3.forward(x))

        x = adaptive_avg_pool2d_numpy(x, output_size=1)  # shape: (B, 16, 1, 1)
        x = x.reshape(x.shape[0], -1)  # shape: (B, 16)

        x = self.relu(self.fc1.forward(x))  # shape: (B, 64)
        x = self.relu(self.fc2.forward(x))
        x = self.fc3.forward(x)             # shape: (B, num_players * num_actions)

        return x.reshape(-1, self.num_players, self.num_actions)


class PolicyNet_NumpyBc:
    def __init__(self, num_players=5, num_actions=5):
        self.num_players = num_players
        self.num_actions = num_actions

        self.conv1 = Conv2D_Numpy(in_channels=93, out_channels=8, kernel_size=3, padding='same')
        self.bn1 = BatchNorm2d(8)

        self.conv2 = Conv2D_Numpy(in_channels=8, out_channels=16, kernel_size=3, padding='same')
        self.bn2 = BatchNorm2d(16)

        self.conv3 = Conv2D_Numpy(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.bn3 = BatchNorm2d(16)

        self.fc1 = Linear(in_features=16, out_features=64)
        self.bn_fc1 = BatchNorm1d(64)

        self.fc2 = Linear(in_features=64, out_features=128)
        self.bn_fc2 = BatchNorm1d(128)

        self.fc3 = Linear(in_features=128, out_features=num_players * num_actions)

    def relu(self, x):
        return np.maximum(0, x)

    def eval(self):
        pass

    def train(self):
        pass

    def forward(self, x):
        x = self.relu(self.bn1.forward(self.conv1.forward(x)))
        x = self.relu(self.bn2.forward(self.conv2.forward(x)))
        x = self.relu(self.bn3.forward(self.conv3.forward(x)))

        x = adaptive_avg_pool2d_numpy(x, output_size=1)  # shape: (B, 16, 1, 1)
        x = x.reshape(x.shape[0], -1)  # shape: (B, 16)

        x = self.relu(self.bn_fc1.forward(self.fc1.forward(x)))  # shape: (B, 64)
        x = self.relu(self.bn_fc2.forward(self.fc2.forward(x)))  # shape: (B, 128)
        x = self.fc3.forward(x)                                   # shape: (B, num_players * num_actions)

        return x.reshape(-1, self.num_players, self.num_actions)


    import torch

def load_pytorch_weights_into_numpy_model(pytorch_path, numpy_model):
    checkpoint = torch.load(pytorch_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    # Helper pour copier les poids
    def copy_conv_weights(conv_numpy, conv_torch_prefix):
        conv_numpy.weight = state_dict[f'{conv_torch_prefix}.weight'].cpu().numpy()
        if conv_numpy.use_bias:
            conv_numpy.bias = state_dict[f'{conv_torch_prefix}.bias'].cpu().numpy()

    def copy_linear_weights(linear_numpy, linear_torch_prefix):
        linear_numpy.weight = state_dict[f'{linear_torch_prefix}.weight'].cpu().numpy()
        if linear_numpy.bias_enabled:
            linear_numpy.bias = state_dict[f'{linear_torch_prefix}.bias'].cpu().numpy()

    copy_conv_weights(numpy_model.conv1, 'conv1')
    copy_conv_weights(numpy_model.conv2, 'conv2')
    copy_conv_weights(numpy_model.conv3, 'conv3')
    copy_linear_weights(numpy_model.fc, 'fc')

import numpy as np

def softmax(x, axis=-1):
    # Soustrait le max pour �viter l'overflow num�rique
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# 2. Multinomial sampling
def multinomial_numpy(probs):
    """
    Simule torch.multinomial(probs, num_samples=1).squeeze(1)
    Args:
        probs: np.ndarray of shape (batch, num_classes)
    Returns:
        np.ndarray of shape (batch,)
    """
    batch_size, num_classes = probs.shape
    samples = np.array([
        np.random.choice(num_classes, p=probs[i])
        for i in range(batch_size)
    ])
    return samples

