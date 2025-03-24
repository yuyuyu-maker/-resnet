import torch
import torch.nn as nn
import torch.nn.functional as F


# 普通卷积结构
class ConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ConvNet, self).__init__()
        # in_channels是输入通道，out_channels是输出通道，kernel_size是卷积核尺寸, stride是步长, padding是填充
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = self.fc2(x)
        return x


# 全连接层结构
class FullyConnectedNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = self.fc3(x)
        print(x.shape)
        return x


# 残差块
class BasicResidualBlock(nn.Module):
    def __init__(self, channels):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x  # 直接跳跃连接
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = self.bn2(self.conv2(out))
        out += identity  # 直接相加
        return F.relu(out)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 只有在通道数不同或者stride > 1时才使用downsample
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # 通过1×1卷积调整尺寸或通道数


        out = F.relu(self.bn1(self.conv1(x)))
        print(f"进行1×1残差连接之前：{out.shape}")
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1):
        super(BottleneckResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        return F.relu(out)

if __name__ == "__main__":
    print("现在我们创建一个简单的多层感知机")
    model = FullyConnectedNet()


    input_tensor_fc = torch.randn(2, 784)
    # 通过模型传递虚拟张量
    output_fc = model(input_tensor_fc)
    # ——————————————————————————————————————————————————————————————————
    # 创建一个卷积神经网络模型实例
    print("现在我们来创建一个卷积神经网络的实例")
    model_cnn= ConvNet()

    # 生成一个虚拟输入张量 (batch_size=1, channels=3, height=32, width=32)
    input_tensor = torch.randn(1, 3, 32, 32)

    # 进行前向传播
    output_cnn = model_cnn(input_tensor)

    # 打印输出形状
    print("输出张量形状:", output_cnn.shape)

    # —————————————————————————————————————————————————————————————————————
    print("现在我们创建三个残差块,大家观察他们尺寸和通道的对比")
    x = torch.randn(1, 64, 32, 32)

    # 直接相加的残差块
    block1 = BasicResidualBlock(64)
    out1 = block1(x)
    print(out1.shape)  # torch.Size([1, 64, 32, 32])
    # 1×1 卷积调整的残差块（in_channels ≠ out_channels 或 stride > 1）

    block2 = ResidualBlock(64, 128, stride=2)
    out2 = block2(x)
    print(out2.shape)  # torch.Size([1, 128, 16, 16])  # stride=2 使特征图缩小

    # Bottleneck 结构（ResNet-50/101）
    block3 = BottleneckResidualBlock(64, 32, 128, stride=2)
    out3 = block3(x)
    print(out3.shape)  # torch.Size([1, 128, 16, 16])  # Bottleneck 结构



