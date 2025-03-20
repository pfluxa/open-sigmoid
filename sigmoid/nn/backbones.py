import math

import torch

from sigmoid.nn.convolutional import MBConvBlock, TransposedMBConvBlock


class TransposedEfficientNet1D(torch.nn.Module):
    
    @staticmethod
    def _next_power_of_2(x):
        return 2**(math.floor(math.log(x, 2)))
     
    def __init__(self, in_channels: int, in_features: int, n_blocks: int, out_chan: int, out_feat: int, fc_head: bool = False, fc_out_dim: int = 0):
        
        super(TransposedEfficientNet1D, self).__init__()
        
        self.in_features_ = in_features
        self.in_channels_ = in_channels 
        self.has_fc_ = fc_head
        self.fc_out_dim_ = fc_out_dim
        # Initial stem convolution
        h0 = in_features

        # Building blocks
        self.blocks_ = []
        self.repeat_pattern_ = [2, 2, 3, 3, 2]
        self.repeat_pattern_[0:n_blocks].reverse()
        self.kernel_sizes_ = [3, 5, 3, 5, 5, 3]
        self.kernel_sizes_[0:n_blocks].reverse()
        self.strides_ = [2, 2, 2, 1, 2, 1]
        self.strides_[0:n_blocks].reverse()
        self.factors_ = [1.5, 1.66, 2.0, 1.4, 1.71, 1.66]
        self.factors_[0:n_blocks].reverse()
        
        c_curr = self.in_channels_
        nc_stem = TransposedEfficientNet1D._next_power_of_2(c_curr)
        self.stem = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(c_curr, nc_stem, 
                            kernel_size=3, stride=2, padding=1, 
                            bias=False),
            torch.nn.BatchNorm1d(nc_stem),
            torch.nn.LeakyReLU(0.02)
        )
        
        c_curr = nc_stem
        i = 0
        h = h0
        while i < n_blocks: # and c_curr >= 6:
            
            nr = self.repeat_pattern_[i]
            f = self.factors_[i]
            k = self.kernel_sizes_[i]
            
            s = self.strides_[i]
            h = math.ceil(h * s)
            c_next = math.ceil(c_curr / f)
            # if c_next < 6:
            #    break 
            for j in range(nr):
                # print(c_curr, c_next, k, h)
                self.blocks_.append(
                    TransposedMBConvBlock(c_curr, c_next, 6, k, s)
                )
                if j == 0:
                    s = 1
                    c_curr = c_next
            self.blocks_.append(
                TransposedMBConvBlock(c_curr, c_curr, 6, k, 1) 
            )
            i = i + 1
        
        c_next = c_curr * 2
        self.blocks_.append(
            TransposedMBConvBlock(c_curr, c_next, 1, 3, 1)
        )
        self.blocks = torch.nn.Sequential(*self.blocks_)
        c_curr = c_next
         
        # Head
        c_next = c_curr * 2 # math.ceil(2**(math.log2(c_curr) + 1))
        self.head = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(c_curr, out_chan, 
                kernel_size=3, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm1d(out_chan),
            torch.nn.LeakyReLU(0.02)
        )
        
        if self.has_fc_:
            # global average pooling and classifier
            self.avg_pool = torch.nn.AdaptiveMaxPool1d(1)
            self.fc = torch.nn.Linear(c_curr, self.fc_out_dim_)

    def forward(self, x):
        
        x = self.stem(x)
        print("stem out:", x.shape)
        x = self.blocks(x)
        print("blocks out:", x.shape)
        x = self.head(x)
        if self.has_fc_:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class EfficientNet1D(torch.nn.Module):
    
    def __init__(self, in_channels: int, in_features: int, fc_head: bool = False, fc_out_dim: int = 0):
        
        super(EfficientNet1D, self).__init__()
        
        self.in_features_ = in_features
        self.in_channels_ = in_channels 
        self.has_fc_ = fc_head
        self.fc_out_dim_ = fc_out_dim
        # Initial stem convolution
        h0 = in_features
        nc_stem = int(2**(math.log2(in_channels) + 1))
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, nc_stem, 
                            kernel_size=3, stride=2, padding=1, 
                            bias=False),
            torch.nn.BatchNorm1d(nc_stem),
            torch.nn.LeakyReLU(0.02)
        )

        # Building blocks
        self.blocks_ = []
        self.repeat_pattern_ = [2, 2, 3, 3, 2]
        self.kernel_sizes_ = [3, 5, 3, 5, 5, 3]
        self.strides_ = [2, 2, 2, 1, 2, 1]
        self.factors_ = [1.5, 1.66, 2.0, 1.4, 1.71, 1.66]
        
        c_curr = nc_stem
        c_next = c_curr // 2
        self.blocks_.append(
            MBConvBlock(c_curr, c_next, 1, 3, 1)
        )
        c_curr = c_next
        
        i = 1
        h = h0
        while i < len(self.repeat_pattern_):
            
            nr = self.repeat_pattern_[i]
            f = self.factors_[i]
            k = self.kernel_sizes_[i]
            
            s = self.strides_[i]
            h = int(math.floor(h / s))
            if h <= 2:
                break
            c_next = math.floor(f * c_curr) 
            for j in range(nr):
                # print(c_curr, c_next, k, s)
                self.blocks_.append(
                    MBConvBlock(c_curr, c_next, 6, k, s)
                )
                if j == 0:
                    s = 1
                    c_curr = c_next
            self.blocks_.append(
                MBConvBlock(c_curr, c_curr, 6, k, 1) 
            )
            i = i + 1
        
        self.blocks = torch.nn.Sequential(*self.blocks_)
        self.n_blocks_ = i - 1
        # Head
        self.head = torch.nn.Sequential(
            torch.nn.Conv1d(c_curr, c_curr, 
                kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm1d(c_curr),
            torch.nn.LeakyReLU(0.02)
        )
        if self.has_fc_:
            # global average pooling and classifier
            self.avg_pool = torch.nn.AdaptiveMaxPool1d(1)
            self.fc = torch.nn.Linear(c_curr, self.fc_out_dim_)

        self.inter_nc_ = c_curr
        self.inter_nf_ = h

    def forward(self, x):
        x = self.stem(x)
        print("stem out:", x.shape)
        x = self.blocks(x)
        x = self.head(x)
        print("head out:", x.shape)
        if self.has_fc_:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


if __name__ == '__main__':
    
    from thop import profile
  
    n = 10
    lB = [8] * n
    lC = range(3, 3 + n)
    lH = range(3, 3 + n)
   
    for B, C, H in zip(lB, lC, lH): 
        print('-' * 79)
        
        model1 = EfficientNet1D(C, H, fc_head=False)
        model1.to('cuda')
        
        x_input = torch.randn(B, C, H)
        x_input = x_input.to('cuda')
        print("input dim:", x_input.shape)
        
        y_hat = model1(x_input)
        # print("encoded dim:", y_hat.shape)
        
        # flops, params = profile(model, inputs=(x_input,))
        # print("FLOPs = ", str(flops / 1e6) + '{}'.format("M"))
        # print("params = ", str(params / 1e6) + '{}'.format("M"))
        
        # print("n blocks:", model1.n_blocks_) 
        model2 = TransposedEfficientNet1D(
            model1.inter_nc_, C, model1.n_blocks_, x_input.shape[1], x_input.shape[2])
        model2.to('cuda')
        
        x_input = model2(y_hat)
        print("decoded dim:", x_input.shape)
        print('-' * 79 + '\n')
    
    # flops, params = profile(model2, inputs=(y_hat,))
    # print("FLOPs = ", str(flops / 1e6) + '{}'.format("M"))
    # print("params = ", str(params / 1e6) + '{}'.format("M"))