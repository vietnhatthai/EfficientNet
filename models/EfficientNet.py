import torch
import torch.nn as nn
from math import ceil

base_model = [
    # epand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple of : (phi_value, resolution, drop_rate)
    "efficientnet-b0": (0, 224, 0.2), # alpha, beta, grama, depth = alpha ** phi
    "efficientnet-b1": (0.5, 240, 0.2),
    "efficientnet-b2": (1, 260, 0.3),
    "efficientnet-b3": (2, 300, 0.3),
    "efficientnet-b4": (3, 380, 0.4),
    "efficientnet-b5": (4, 456, 0.4),
    "efficientnet-b6": (5, 528, 0.5),
    "efficientnet-b7": (6, 600, 0.5),
}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()

        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True) # SiLU <-> Swish

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels,
        kernel_size, 
        stride,
        padding, 
        expand_ratio,
        reduction=4, # squeeze excitation
        survival_prob=0.8, # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = in_channels // reduction

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,
            )
        
        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EffictiveNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EffictiveNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factor(version)
        last_channel = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channel)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channel, num_classes),
        )
    def calculate_factor(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate
    
    def create_features(self, width_factor, depth_factor, last_channel):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, kernel_size=3, stride=2, padding=1)]
        in_channels = channels
        
        for expand_ratio, channels, reapeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels/width_factor)/4)
            layer_repeats  = ceil(reapeats * depth_factor)

            for layer in range(layer_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer==0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size//2, # if k=1:padding=0 k=3:padding=1 k=5:padding=2 
                    )
                )
                
                in_channels = out_channels
            
        features.append(
            CNNBlock(in_channels, last_channel, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.size(0), -1))


if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    version = "efficientnet-b0"
    model = EffictiveNet(version, num_classes=1000)
    model = model.to(device)
    phi, res, drop_rate = phi_values[version] 
    summary(model, (3, res, res))
    x = torch.randn(32, 3, res, res).to(device)
    y = model(x)
    print(y.size()) # (32, 1000)