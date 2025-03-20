""" Contains blocks to build convolutional backbones.
"""
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM1D(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM1D, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention1D(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention1D(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        # print(chan_att.size())
        fp = chan_att * f
        # print(fp.size())
        spat_att = self.spatial_attention(fp)
        # print(spat_att.size())
        fpp = spat_att * fp
        # print(fpp.size())
        return fpp


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention1D, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        # self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))
        self.conv = nn.Conv1d(
            in_channels = 2,
            out_channels = 1,
            kernel_size = kernel_size,
            padding= int((kernel_size-1)/2)
        )
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim = 1)
        conv = self.conv(pool)
        # batchnorm ????????????????????????????????????????????
        conv = conv.repeat(1,x.size()[1],1)
        att = torch.sigmoid(conv)
        return att

    def agg_channel(self, x, pool = "max"):
        b,c,h = x.size()
        x = x.view(b, c, h)
        x = x.permute(0,2,1)
        if pool == "max":
            x = F.max_pool1d(x,c)
        elif pool == "avg":
            x = F.avg_pool1d(x,c)
        x = x.permute(0,2,1)
        x = x.view(b,1,h)

        return x


class ChannelAttention1D(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention1D, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in/ float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )


    def forward(self, x):
        kernel = (x.size()[2],)
        avg_pool = F.avg_pool1d(x, kernel)
        max_pool = F.max_pool1d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2)

        out = sig_pool.repeat(1,1,kernel[0])
        return out

class SEModule(torch.nn.Module):

    def __init__(self,in_channel, ratio=4):

        super(SEModule, self).__init__()
        self.avepool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear1 = torch.nn.Linear(in_channel,in_channel//ratio)
        self.linear2 = torch.nn.Linear(in_channel//ratio,in_channel)
        self.Hardsigmoid = torch.nn.Hardsigmoid(inplace=True)
        self.Relu = torch.nn.ReLU(inplace=True)

    def forward(self,input):

        b,c,_ = input.shape
        x = self.avepool(input)
        x = x.view([b,c])
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.Hardsigmoid(x)
        x = x.view([b,c,1])

        return input*x


class TransposedMBConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=4):

        super(TransposedMBConvBlock, self).__init__()
        # Expansion phase
        expanded_channels = int(in_channels * expand_ratio)
        self.expand_conv = torch.nn.Conv1d(in_channels, expanded_channels,
                                            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(expanded_channels)
        # Depthwise convolution
        self.depthwise_conv = torch.nn.ConvTranspose1d(expanded_channels, expanded_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=kernel_size // 2,
                                              groups=expanded_channels,
                                              bias=False)
        self.bn2 = torch.nn.BatchNorm1d(expanded_channels)
        # Squeeze and Excitation (SE) phase
        # self.se = SEModule(shrinked_channels, se_ratio)
        # Conv. attention
        self.attn = CBAM1D(expanded_channels, se_ratio, kernel_size)
        # Linear Bottleneck
        self.linear_bottleneck = torch.nn.Conv1d(expanded_channels, out_channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0,
                                                 bias=False)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        # Skip connection if input and output channels are the same and stride is 1
        self.use_skip_connection = (stride == 1) and (in_channels == out_channels)
        self.leakyrelu = torch.nn.LeakyReLU(0.02)

    def forward(self, x):

        identity = x
        # Expansion phase
        x = self.leakyrelu(self.bn1(self.expand_conv(x)))
        # Depthwise convolution phase
        x = self.leakyrelu(self.bn2(self.depthwise_conv(x)))
        # print(x.shape)
        # change Squeeze and Excitation phase to Convolutional Attention
        # x = self.se(x)
        x = self.attn(x)
        # Linear Bottleneck phase
        x = self.linear_bottleneck(x)
        x = self.bn3(x) #self.linear_bottleneck(x))
        # Skip connection
        if self.use_skip_connection:
            x = identity + x

        return x


class MBConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=4):

        super(MBConvBlock, self).__init__()
        # Expansion phase
        expanded_channels = int(in_channels * expand_ratio)
        self.expand_conv = torch.nn.Conv1d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(expanded_channels)
        # Depthwise convolution
        self.depthwise_conv = torch.nn.Conv1d(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=expanded_channels, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(expanded_channels)
        # Squeeze and Excitation (SE) phase
        self.se = SEModule(expanded_channels, se_ratio)
        # self.attn = CBAM1D(expanded_channels, se_ratio, kernel_size)
        # Linear Bottleneck
        self.linear_bottleneck = torch.nn.Conv1d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        # Skip connection if input and output channels are the same and stride is 1
        self.use_skip_connection = (stride == 1) and (in_channels == out_channels)
        self.leakyrelu = torch.nn.LeakyReLU(0.02)

    def forward(self, x):

        identity = x
        # Expansion phase
        x = self.leakyrelu(self.bn1(self.expand_conv(x)))
        # Depthwise convolution phase
        x = self.leakyrelu(self.bn2(self.depthwise_conv(x)))
        # Squeeze and Excitation phase
        x = self.se(x)
        # x = self.attn(x)
        # Linear Bottleneck phase
        x = self.bn3(self.linear_bottleneck(x))

        # Skip connection
        if self.use_skip_connection:
            x = identity + x

        return x

def test_CBAM1D():
    # ca = CBAM()
    f = torch.FloatTensor([
        [
            [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]
        ]
    ])
    print('INPUT SIZE TO CBAM1D:', f.size())
    # sa = SpatialAttention(kernel_size = 3)
    # sa(f)
    cbam = CBAM1D(n_channels_in = f.size()[1], reduction_ratio = 2, kernel_size = 3)
    fpp = cbam(f)
    print('OUTPUT SIZE FROM CBAM1D:', fpp.size())
    print(fpp)
    # print(f)
    # print(fp)

if __name__ == "__main__":

    test_CBAM1D()
