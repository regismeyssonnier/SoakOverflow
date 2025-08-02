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
    def __init__(self, num_players=10, num_actions=5):
        self.num_players = num_players
        self.num_actions = num_actions

        self.conv1 = Conv2D_Numpy(in_channels=83, out_channels=8, kernel_size=3, padding='same')
        self.conv2 = Conv2D_Numpy(in_channels=8, out_channels=16, kernel_size=3, padding='same')
        self.conv3 = Conv2D_Numpy(in_channels=16, out_channels=16, kernel_size=3, padding='same')

        self.fc = Linear(in_features=16, out_features=num_players * num_actions)

    def relu(self, x):
        return np.maximum(0, x)

    def eval(self):
        pass

    def forward(self, x):
        """
        x: numpy array of shape (batch_size, 83, H, W)
        returns: numpy array of shape (batch_size, num_players, num_actions)
        """
        x = self.relu(self.conv1.forward(x))
        x = self.relu(self.conv2.forward(x))
        x = self.relu(self.conv3.forward(x))

        x = adaptive_avg_pool2d_numpy(x, output_size=1)  # shape: (B, C, 1, 1)
        x = x.reshape(x.shape[0], -1)  # shape: (B, 16)
        x = self.fc.forward(x)         # shape: (B, num_players * num_actions)
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

