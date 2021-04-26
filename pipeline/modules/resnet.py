import torch
import torch.nn as nn


class ResNet18(nn.Module):
    """
    ResNet18 Architecture Class
    """
    def __init__(self):
        super(ResNet18, self).__init__()

        self.in_sequence = nn.Sequential(
            # 3x224x224
            nn.Conv2d(3, 64, (7, 7), padding=(3, 3), stride=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=(2, 2)),
        )

        # 64x112x112
        self.layer1 = ResNetLayer(64, 64, 2, downsampling=False, first_layer=True)
        # 64x56x56
        self.layer2 = ResNetLayer(64, 128, 2)
        # 128x28x28
        self.layer3 = ResNetLayer(128, 256, 2)
        # 256x14x14
        self.layer4 = ResNetLayer(256, 512, 2)
        # 512x7x7

        self.out_sequence = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.fc = nn.Linear(512, 1000, bias=True)

    def forward(self, x):
        x = self.in_sequence(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out_sequence(x)
        x = self.fc(x)
        return x

    def predict(self, x):
        y_ = self.forward(x)
        return torch.cat([y_[:, :4], torch.argmax(y_[:, 4:], dim=1).unsqueeze_(0) + 1], dim=1)


class ResNetLayer(nn.Module):
    """
    ResNet Layer Class
    """
    def __init__(self, input_filters, output_filters, num_blocks,
                 downsampling=True, first_layer=False, bottleneck=False):
        super(ResNetLayer, self).__init__()

        self.sequence = nn.Sequential()
        for i in range(num_blocks):
            if i == 0:
                if bottleneck:
                    self.sequence.add_module(f'{i}',
                                             Bottleneck(input_filters, output_filters, downsampling, first_layer))
                else:
                    self.sequence.add_module(f'{i}',
                                             BasicBlock(input_filters, output_filters, downsampling, first_layer))
            else:
                if bottleneck:
                    self.sequence.add_module(f'{i}', Bottleneck(output_filters, output_filters))
                else:
                    self.sequence.add_module(f'{i}', BasicBlock(output_filters, output_filters))

    def forward(self, x):
        x = self.sequence(x)
        return x


class Bottleneck(nn.Module):
    """
    ResNet Bottleneck Class
    """
    def __init__(self, input_filters, output_filters, downsampling=False, first_layer=False):
        super(Bottleneck, self).__init__()

        self.downsampling = downsampling
        self.first_layer = first_layer
        conv_stride = (2, 2) if self.downsampling else (1, 1)

        self.sequential = nn.Sequential(
            nn.Conv2d(input_filters, output_filters//4, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(output_filters//4),
            nn.Conv2d(output_filters//4, output_filters//4, kernel_size=(3, 3), padding=(1, 1), stride=conv_stride, bias=False),
            nn.BatchNorm2d(output_filters//4),
            nn.Conv2d(output_filters//4, output_filters, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(output_filters),
            # nn.Dropout2d(0.1),
        )
        self.relu = nn.ReLU(inplace=True)

        if self.downsampling or self.first_layer:
            ds_stride = (1, 1) if self.first_layer else (2, 2)
            self.downsample = nn.Sequential(
                nn.Conv2d(input_filters, output_filters, kernel_size=(1, 1), stride=ds_stride, bias=False),
                nn.BatchNorm2d(output_filters)
            )

    def forward(self, x):
        initial = x
        x = self.sequential(x)

        if self.downsampling or self.first_layer:
            x += self.downsample(initial)
        else:
            x += initial
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    """
    ResNet Basic Block Class
    """
    def __init__(self, input_filters, output_filters, downsampling=False, first_layer=False):
        super(BasicBlock, self).__init__()

        self.downsampling = downsampling
        self.first_layer = first_layer
        conv_stride = (2, 2) if self.downsampling else (1, 1)

        self.sequential = nn.Sequential(
            nn.Conv2d(input_filters, output_filters, kernel_size=(3, 3), padding=(1, 1), stride=conv_stride, bias=False),
            nn.BatchNorm2d(output_filters),
            # nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(output_filters, output_filters, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(output_filters),
            # nn.Dropout2d(0.1),
        )
        self.relu = nn.ReLU(inplace=True)

        if self.downsampling or self.first_layer:
            ds_stride = (1, 1) if self.first_layer else (2, 2)
            self.downsample = nn.Sequential(
                nn.Conv2d(input_filters, output_filters, kernel_size=(1, 1), stride=ds_stride, bias=False),
                nn.BatchNorm2d(output_filters)
            )

    def forward(self, x):
        initial = x
        x = self.sequential(x)

        if self.downsampling or self.first_layer:
            x += self.downsample(initial)
        else:
            x += initial
        x = self.relu(x)

        return x
