"""Model definitions: a fully-connected baseline and a configurable CNN.

The CNN exposes the knobs used for the controlled experiment (kernel size)
while keeping everything else (depth, number of filters, pooling, stride,
padding strategy, activation) fixed, so a single class can produce every
variant used in the notebook.
"""

import torch.nn as nn


class BaselineMLP(nn.Module):
    """Flatten + Dense baseline, no convolutional layers.

    28x28x1 -> 784 -> 256 -> 128 -> 10
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """Two convolutional blocks (Conv -> ReLU -> MaxPool) + dense head.

    Design rationale (see README / notebook section 3 for full justification):
      - kernel_size: receptive-field size per conv layer (the variable under
        study in the controlled experiment).
      - stride=1 with 'same' padding: preserves spatial resolution inside a
        block, letting MaxPool (not the conv) be the sole source of
        downsampling.
      - 2 conv blocks: enough depth to move from edges (block 1) to
        textures/parts (block 2) on 28x28 inputs without over-parameterizing
        a dataset of this size.
      - filters double (16 -> 32) after each pooling step, the usual
        compensation for halved spatial resolution.
      - Dropout before the final linear layer to curb overfitting, since the
        dense head is the most parameter-heavy part of the network.
    """

    def __init__(
        self,
        num_classes: int = 10,
        kernel_size: int = 3,
        conv_filters=(16, 32),
        use_pooling: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        padding = kernel_size // 2  # 'same' padding for odd kernel sizes, stride=1

        layers = []
        in_channels = 1
        spatial = 28
        for out_channels in conv_filters:
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
            )
            layers.append(nn.ReLU(inplace=True))
            if use_pooling:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                spatial //= 2
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

        flat_features = in_channels * spatial * spatial
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
