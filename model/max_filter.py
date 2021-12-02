import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine.module.normalization import GroupNorm


class MaxFiltering(M.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, tau: int = 2, use_gn: bool = True):
        super().__init__()
        self.conv = M.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm = GroupNorm(32, in_channels) if use_gn else M.identity()
        self.nonlinear = M.ReLU()
        self.max_pool = MaxPool3d(
            kernel_size=(tau + 1, kernel_size, kernel_size), padding=(tau // 2, kernel_size // 2, kernel_size // 2)
        )
        self.margin = tau // 2

    def forward(self, inputs):
        features = []
        for l, x in enumerate(inputs):
            features.append(self.conv(x))

        outputs = []
        for l, x in enumerate(features):
            def func(f): return F.nn.interpolate(
                f, size=x.shape[2:], mode="bilinear")
            feature_3d = []
            for k in range(max(0, l - self.margin), min(len(features), l + self.margin + 1)):
                feature_3d.append(func(features[k]) if k != l else features[k])
            feature_3d = F.stack(feature_3d, axis=2)
            max_pool = self.max_pool(feature_3d)[:, :, min(l, self.margin)]
            output = max_pool + inputs[l]
            outputs.append(self.nonlinear(self.norm(output)))
        return outputs


class MaxPool3d(M.Module):
    """
        A simple implement of MaxPool3d based on MaxPool2d and F.max
    """

    def __init__(self, kernel_size, padding):
        super(MaxPool3d, self).__init__()
        self.n = kernel_size[0]
        self.padding = padding
        self.maxpool2d = M.MaxPool2d(kernel_size[1:], stride=1)

    def forward(self, inputs):
        """
        Arguments:
            inputs (Tensor): with shape [b, c, n, h, w]
        """
        b = inputs.shape[0]
        f = inputs.shape[2]
        outputs = []
        for i in range(b):
            inputs_i = self.maxpool2d(
                F.nn.pad(inputs[i], ((0, 0), *[(x, x) for x in self.padding])))
            overlap_channel_inputs = [inputs_i[:, idx:idx+self.n]
                                      for idx in range(f - self.n + 1 + 2 * self.padding[0])]
            outputs_i = F.concat([x.max(axis=1, keepdims=True)
                                 for x in overlap_channel_inputs], axis=1)
            outputs.append(outputs_i)
        return F.stack(outputs, axis=0)


if __name__ == '__main__':
    # ____________ test MaxPool3d ________________ #
    maxpool = MaxPool3d((2, 3, 3), padding=(1, 1, 1))
    import torch.nn as nn
    maxp = nn.MaxPool3d((2, 3, 3), padding=(1, 1, 1), stride=1)
    import numpy as np
    x = np.arange(2*6*4*16*20).reshape(2, 6, 4, 16, 20)
    x = np.array(x, dtype=np.float32)
    inputs = mge.tensor(x)
    print(inputs.shape)
    a = maxpool(inputs)
    b = maxp(torch.tensor(x))
    print((a.numpy() == b.numpy()).mean())
