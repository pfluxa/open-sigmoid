from tqdm.utils import Comparable
import torch
import torch.nn as nn
# import torch.nn.functional as F
from scipy.optimize import minimize, Bounds
import numpy as np
from gekko import GEKKO


from sigmoid.nn.convolutional import MBConvBlock, TransposedMBConvBlock, CBAM1D

def find_valid_parameters(input_dim, output_dim):

    m = GEKKO()             # create GEKKO model
    m.options.SOLVER=1  # APOPT is an MINLP solver

    # optional solver settings with APOPT
    m.solver_options = ['minlp_maximum_iterations 50000', \
                        # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol 1000', \
                        # treat minlp as nlp
                        'minlp_as_nlp 0', \
                        # nlp sub-problem max iterations
                        'nlp_maximum_iterations 5000', \
                        # 1 = depth first, 2 = breadth first
                        'minlp_branch_method 1', \
                        # maximum deviation from whole number
                        'minlp_integer_tol 0.001', \
                        # covergence tolerance
                        'minlp_gap_tol 0.001']

    d = 1.0
    n = 16*input_dim - 15
    k = m.Var(value=3, integer=True, lb=1, ub=19)      # define new variable, initial value=1
    s = m.Var(value=2, integer=True, lb=1, ub=16)      # define new variable, initial value=1
    p = m.Var(value=0, integer=True, lb=0, ub=64)
    # (L_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation}
    #                        \times (\text{kernel\_size} - 1) + \text{output\_padding} + 1
    m.Equations([(n - 1) * s - 2 * p + d * (k - 1) + 1 == output_dim])
    # m.Obj(p)
    m.solve(disp=False)     # solve
    # print('Results')
    # print('n: ' + str(n))
    # print('o: ' + str(output_dim))
    # print('k: ' + str(k.value))
    # print('s: ' + str(s.value))
    # print('p: ' + str(p.value))
    # print('Objective: ' + str(m.options.objfcnval))
    return int(k.value[0]), int(s.value[0]), int(p.value[0])


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class MBConvBlock1D(nn.Module):
    """1D Mobile Inverted Bottleneck Convolution (MBConv) Block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock1D, self).__init__()
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        hidden_dim = in_channels * expand_ratio

        # Expansion phase
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
                # nn.BatchNorm1d(hidden_dim),
                # CBAM1D(hidden_dim, expand_ratio, kernel_size=1),
                Swish()
            )
        else:
            self.expand = nn.Identity()

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden_dim, bias=False),
            # nn.BatchNorm1d(hidden_dim),
            CBAM1D(hidden_dim, expand_ratio, kernel_size=kernel_size),
            # Swish()
        )

        # Squeeze-and-Excitation (SE) block
        if se_ratio is not None:
            squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(hidden_dim, squeezed_channels, kernel_size=1),
                Swish(),
                # CBAM1D(squeezed_channels, int(1./se_ratio), kernel_size=kernel_size),
                nn.Conv1d(squeezed_channels, hidden_dim, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.se = nn.Identity()

        # Pointwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False),
            CBAM1D(hidden_dim, out_channels, 1)
            # nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x) * x  # Apply SE block
        x = self.pointwise(x)

        if self.use_residual:
            x += residual  # Residual connection
        return x

class EfficientNet1D(nn.Module):
    """EfficientNet Model for 1D inputs"""
    def __init__(self, input_channels: int, output_channels: int, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2):

        super(EfficientNet1D, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, int(32 * width_mult), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(int(32 * width_mult)),
            CBAM1D(int(32 * width_mult), 1, kernel_size=3),
            # Swish()
        )

        # Define MBConv blocks (EfficientNet-B0 configuration)
        self.blocks = nn.Sequential(
            # block 0
            MBConvBlock(int(32 * width_mult), int(16 * width_mult), kernel_size=3, stride=1, expand_ratio=1),
            # block 1
            MBConvBlock(int(16 * width_mult), int(24 * width_mult), kernel_size=3, stride=2, expand_ratio=6),
            MBConvBlock(int(24 * width_mult), int(24 * width_mult), kernel_size=3, stride=1, expand_ratio=6),
            # block 2
            MBConvBlock(int(24 * width_mult), int(40 * width_mult), kernel_size=5, stride=2, expand_ratio=6),
            MBConvBlock(int(40 * width_mult), int(40 * width_mult), kernel_size=5, stride=1, expand_ratio=6),
            # block 3
            MBConvBlock(int(40 * width_mult), int(80 * width_mult), kernel_size=3, stride=2, expand_ratio=6),
            MBConvBlock(int(80 * width_mult), int(80 * width_mult), kernel_size=3, stride=1, expand_ratio=6),
            MBConvBlock(int(80 * width_mult), int(80 * width_mult), kernel_size=3, stride=1, expand_ratio=6),
            # block 4
            # MBConvBlock( int(80 * width_mult), int(112 * width_mult), kernel_size=5, stride=1, expand_ratio=6),
            # MBConvBlock(int(112 * width_mult), int(112 * width_mult), kernel_size=5, stride=1, expand_ratio=6),
            # MBConvBlock(int(112 * width_mult), int(112 * width_mult), kernel_size=5, stride=1, expand_ratio=6),
            # # block 5
            # MBConvBlock(int(112 * width_mult), int(192 * width_mult), kernel_size=5, stride=2, expand_ratio=6),
            # MBConvBlock(int(192 * width_mult), int(192 * width_mult), kernel_size=5, stride=1, expand_ratio=6),
            # # block 6
            # MBConvBlock(int(192 * width_mult), int(320 * width_mult), kernel_size=3, stride=1, expand_ratio=6)
        )

        self.head = nn.Sequential(
            nn.Conv1d(int(80 * width_mult), output_channels, kernel_size=1, bias=False),
            # nn.Conv1d(int(320 * width_mult), output_channels, kernel_size=1, bias=False),
            #nn.Conv1d(int(320 * width_mult), int(1280 * width_mult), kernel_size=1, bias=False),
            # nn.BatchNorm1d(output_channels),
            # Swish(),
            nn.AdaptiveAvgPool1d(1)
        )
        #     nn.AdaptiveAvgPool1d(2),
        #     nn.Dropout(dropout_rate),
        #     nn.Flatten(),
        #     nn.Linear(int(1280 * width_mult), output_dim)
        # )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        
        return x


class InverseSwish(nn.Module):
    """Approximate inverse of Swish activation."""
    def forward(self, x):
        # Swish inverse is not straightforward, so we use an identity mapping as a placeholder
        return x

class InverseMBConvBlock1D(nn.Module):
    """Inverse of the 1D MBConv Block."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(InverseMBConvBlock1D, self).__init__()
        hidden_dim = in_channels * expand_ratio

        # Inverse pointwise convolution
        self.inv_pointwise = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            CBAM1D(hidden_dim, expand_ratio, kernel_size=1)
            # InverseSwish()
        )

        # Inverse Squeeze-and-Excitation (SE) block
        if se_ratio is not None:
            squeezed_channels = max(1, int(out_channels * se_ratio))
            self.inv_se = nn.Sequential(
                nn.Conv1d(hidden_dim, squeezed_channels, kernel_size=1),
                CBAM1D(hidden_dim, int(1./se_ratio), kernel_size=1),
                # InverseSwish(),
                nn.Conv1d(squeezed_channels, hidden_dim, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.inv_se = nn.Identity()

        # Inverse depthwise convolution
        self.inv_depthwise = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=hidden_dim, bias=False),
            CBAM1D(hidden_dim, 1, kernel_size=kernel_size),
            nn.BatchNorm1d(hidden_dim),
            # InverseSwish()
        )

        # Inverse expansion phase
        if expand_ratio == 1:
            self.inv_expand = nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
                # InverseSwish()
            )
        else:
            self.inv_expand = nn.Identity()

    def forward(self, x):
        x = self.inv_pointwise(x)
        x = self.inv_se(x) * x  # Apply inverse SE block
        x = self.inv_depthwise(x)
        x = self.inv_expand(x)
        return x

class InverseEfficientNet1D(nn.Module):
    """Inverse of EfficientNet for 1D inputs."""
    def __init__(self, input_channels: int, output_features: int, width_mult=1.0, depth_mult=1.0):
        super(InverseEfficientNet1D, self).__init__()
        self.inv_head = nn.Sequential(
            # nn.Linear(input_features, int(1280 * width_mult) * input_features),
            # nn.Unflatten(1, (int(1280 * width_mult), input_features)),
            nn.ConvTranspose1d(input_channels, int(320 * width_mult), kernel_size=1, bias=False),
            nn.BatchNorm1d(int(320 * width_mult)),
            # InverseSwish()
        )

        # Inverse MBConv blocks (reverse order of EfficientNet)
        self.inv_blocks = nn.Sequential(
            # block 6
            TransposedMBConvBlock(int(320 * width_mult), int(192 * width_mult), kernel_size=3, stride=1, expand_ratio=6),
            # block 5
            TransposedMBConvBlock(int(192 * width_mult), int(192 * width_mult), kernel_size=5, stride=1, expand_ratio=6),
            TransposedMBConvBlock(int(192 * width_mult), int(112 * width_mult), kernel_size=5, stride=2, expand_ratio=6),
            # block 4
            TransposedMBConvBlock(int(112 * width_mult), int(80 * width_mult), kernel_size=5, stride=1, expand_ratio=6),
            TransposedMBConvBlock(int(80 * width_mult), int(80 * width_mult), kernel_size=5, stride=1, expand_ratio=6),
            TransposedMBConvBlock(int(80 * width_mult), int(80 * width_mult), kernel_size=5, stride=1, expand_ratio=6),
            # block 3
            TransposedMBConvBlock(int(80 * width_mult), int(40 * width_mult), kernel_size=3, stride=1, expand_ratio=6),
            TransposedMBConvBlock(int(40 * width_mult), int(40 * width_mult), kernel_size=3, stride=1, expand_ratio=6),
            TransposedMBConvBlock(int(40 * width_mult), int(40 * width_mult), kernel_size=3, stride=2, expand_ratio=6),
            # block 2
            TransposedMBConvBlock(int(40 * width_mult), int(24 * width_mult), kernel_size=5, stride=1, expand_ratio=6),
            TransposedMBConvBlock(int(24 * width_mult), int(24 * width_mult), kernel_size=5, stride=2, expand_ratio=6),
            # block 1
            TransposedMBConvBlock(int(24 * width_mult), int(16 * width_mult), kernel_size=3, stride=1, expand_ratio=6),
            TransposedMBConvBlock(int(16 * width_mult), int(16 * width_mult), kernel_size=3, stride=2, expand_ratio=6),
            # block 0
            TransposedMBConvBlock(int(16 * width_mult), int(32 * width_mult), kernel_size=3, stride=1, expand_ratio=1)
        )

        # skp = find_valid_parameters(input_features, output_features)
        k, s, p = 1, 1, 0
        self.inv_stem = nn.Sequential(
            nn.ConvTranspose1d(int(32 * width_mult), 1, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm1d(1),
            # CBAM1D(out_channels, 1, kernel_size=k)
        )

    def forward(self, x):
        print("x:", x.shape)
        x = self.inv_head(x)
        print("inv head:", x.shape)
        x = self.inv_blocks(x)
        print("inv blocks:", x.shape)
        x = self.inv_stem(x)
        print("inv stem:", x.shape)
        return x
