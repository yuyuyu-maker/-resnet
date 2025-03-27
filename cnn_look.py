import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.manual_seed(1)

EPOCH = 2
BATCH_SIZE = 64
LR = 0.001
DOWNLOAD_MNIST = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 数据加载函数
def load_data(batch_size, download=True):
    train_data = torchvision.datasets.MNIST(
        root='D:/data',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=download,
    )
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_data = torchvision.datasets.MNIST(
        root='D:/data',
        train=False,
        transform=torchvision.transforms.ToTensor(),
    )
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
    test_y = test_data.targets[:2000]

    return train_loader, test_x.to(device), test_y.to(device)


# CNN 模型类
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


# 激活图获取钩子函数
activation_map_1 = []
activation_map_2 = []


def get_activation_map_hook(layer_name):
    def hook(module, input, output):
        if layer_name == 'conv1':
            activation_map_1.clear()
            activation_map_1.append(output.detach())
        elif layer_name == 'conv2':
            activation_map_2.clear()
            activation_map_2.append(output.detach())

    return hook


# 训练函数
def train(cnn, train_loader, optimizer, loss_func, epochs, test_x, test_y):
    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for step, (b_x, b_y) in enumerate(train_loader):
            b_x, b_y = b_x.to(device), b_y.to(device)

            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                accuracy = (pred_y == test_y.cpu().numpy()).astype(int).sum() / float(test_y.size(0))
                epoch_accuracy += accuracy

        train_losses.append(epoch_loss / len(train_loader))
        test_accuracies.append(epoch_accuracy / len(train_loader))

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}')

    return train_losses, test_accuracies


# 激活图绘制函数
def plot_activation_map(activation_list, layer_name):
    if not activation_list:
        print(f"激活图为空: {layer_name}")
        return

    activation = activation_list[0]  # 取第一个激活图
    activation = activation.cpu().numpy()

    num_filters = activation.shape[1]  # 计算通道数

    num_cols = 4
    num_rows = (num_filters // num_cols) + (num_filters % num_cols > 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    axes = axes.ravel()

    for i in range(num_filters):
        act_map = activation[0, i]
        axes[i].imshow(act_map, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f"{layer_name} - {i}")

    for j in range(num_filters, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# 保存模型函数
def save_model(cnn, filename):
    torch.save(cnn.state_dict(), filename)


# 加载模型函数
def load_model(cnn, filename):
    cnn.load_state_dict(torch.load(filename, map_location=device))
    cnn.eval()


if __name__ == "__main__":
    # 加载数据
    train_loader, test_x, test_y = load_data(BATCH_SIZE, DOWNLOAD_MNIST)

    # 初始化 CNN 模型、优化器、损失函数
    cnn = CNN().to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    # 注册钩子函数以获取激活图
    cnn.conv1[0].register_forward_hook(get_activation_map_hook('conv1'))
    cnn.conv2[0].register_forward_hook(get_activation_map_hook('conv2'))

    # 训练模型
    train_losses, test_accuracies = train(cnn, train_loader, optimizer, loss_func, EPOCH, test_x, test_y)

    # 绘制激活图
    plot_activation_map(activation_map_1, 'conv1')
    plot_activation_map(activation_map_2, 'conv2')

    # 保存模型
    save_model(cnn, 'mnist_cnn.pth')

    # 加载模型
    load_model(cnn, 'mnist_cnn.pth')

    # 可视化训练损失和测试准确率
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCH), train_losses, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练损失变化')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCH), test_accuracies, label='测试准确率', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.title('测试准确率变化')
    plt.legend()

    plt.tight_layout()
    plt.show()
