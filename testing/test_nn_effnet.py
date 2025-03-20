from thop import profile
import numpy
import torch

from sigmoid.nn.efficient_net import EfficientNet1D, InverseEfficientNet1D

if __name__ == '__main__':

    rng = numpy.random.default_rng()
    n = 300
    lc = rng.integers(1, 10, n).tolist()
    lB = [256] * n
    lC = rng.integers(1, 400, n).tolist()
    lH = rng.integers(1, 400, n).tolist()

    for c, B, C, H in zip(lc, lB, lC, lH):
        print('-' * 79)

        model1 = EfficientNet1D(n_channels=C, output_dim=c)
        model1.to('cuda')

        x_input = torch.randn(B, C, H)
        x_input = x_input.to('cuda')
        print("input dim:", x_input.shape)
        y_hat = model1(x_input)
        print("output dim:", y_hat.shape)

        try:
            model2 = InverseEfficientNet1D(in_features=c, out_channels=C, input_dim=c, output_dim=H)
            model2.to('cuda')
            x_input = model2(y_hat)
            print("decoded dim:", x_input.shape)
        except Exception as e:
            print(f"Error creating InverseEfficientNet1D: {e}")
        print('-' * 79 + '\n')

        # flops, params = profile(model1, inputs=(x_input,))
        # print("FLOPs = ", str(flops / 1e6) + '{}'.format("M"))
        # print("params = ", str(params / 1e6) + '{}'.format("M"))

        # flops, params = profile(model2, inputs=(y_hat,))
        # print("FLOPs = ", str(flops / 1e6) + '{}'.format("M"))
        # print("params = ", str(params / 1e6) + '{}'.format("M"))
