# 一、简介



## 1. VAE的输入输出

AE(Auto Encoder)，可用于压缩图像、还原图像、图像降噪等。

VAE(Variational Auto Encoder)，可用于创造前所未见的图像，属于生成式AI。

VAE等auto-encoder架构模型，将真实数据转化为隐变量再重构回预期的“真实数据”，通常输入输出不完全一致，这让模型不是复读机，而是能理解数据并创造新内容的系统。

不一致的原因：

1. **压缩与重构的损耗**

   encoder将高维数据压缩到低位的隐变量空间，会不可避免地丢失一些信息，decoder难以还原。比如丢失图片噪声。

2. **正则化约束的影响**

   KL散度强制隐变量分布接近预设先验分布（如正态分布），这种约束会“牺牲部分细节”来换取平滑规整的隐空间。

3. **模型容量限制**

   若模型容量不足，decoder无法学习到数据全部特征，重构时会出现模糊等偏差。

不一致的意义：

1. **过滤噪声**

   数据压缩到低维空间，过滤不需要的特征，比如修复老照片时照片的污痕。

2. **学习平滑规整的隐空间分布**

   平滑规整的分布，意味着隐空间任意两点之间的插值是有意义的，这表示可以从隐空间采样点，生成模型未训练过的数据。这也是VAE作为**生成模型**的核心能力。

3. **平衡拟合与泛化**

   KL散度相当于在完全拟合的损失函数上加了个惩罚项，提高模型的泛化能力。

例子：

- 输入：一张带轻微斑点的数字 “5” 图片
- 编码器：提取 “5 的轮廓、结构” 等核心特征，隐变量聚焦于这些信息
- 解码器：从隐变量重建出一张清晰的 “5”（斑点被过滤）
- 结果：输出与输入不完全一致，但更接近 “数字 5 的本质形态”



## 2. VAE和AE的比较

VAE（变分自编码器）是 AE（自编码器）的概率增强版本，二者核心区别在于 VAE 引入了**概率建模和隐变量分布约束（归纳偏置）**，使其具备 AE 不具备的生成能力。



从直观角度比较AE和VAE：

  1. **生成新数据的能力（核心优势）**

     * AE 的编码器将输入映射为一个**确定的隐变量 $$z=f(x)$$**，解码器只能根据这个确定的 $$z$$ 重构输入。隐空间可能是 “碎片化” 的 —— 不同样本的  $$z$$ 分布无规律，无法通过采样生成合理的新样本（例如相邻的两个潜空间的点，通过decoder生成的图像是不相似的）。

     * VAE 要求隐变量 $$z$$ 服从一个**预设连续的平滑的先验分布**（如标准正态分布 $$N(0,1)$$），编码器输出的是隐变量分布的参数（均值 $$μ$$ 和方差 $$σ^2$$），解码器从该分布中**随机采样 $z$ ** 进行重构。通过约束隐变量分布接近先验，隐空间变得连续、结构化，允许从先验分布中直接采样生成全新样本（如生成训练数据中未出现的图像）。

       > 例如，用 AE 训练人脸数据，隐空间中的点可能仅对应训练集中的人脸，无法生成 “不存在的人脸”。用 VAE 训练后，从正态分布中随机采样 $z$ ，解码器能生成合理的新人脸，因为隐空间被约束为覆盖所有可能的 “人脸特征组合”。

  2. **概率解释与生成过程建模**

     * AE 是纯确定性模型，无法解释数据生成机制（即 “如何从隐变量生成数据”）。
     * VAE 显式建模了数据生成过程：假设数据  $x$  由隐变量  $z$  生成 $(x\thicksim p_{\theta}(x|z)$ ， $z$ 服从先验分布 $(p(z))$ 。这种概率图模型结构使 VAE 具备**生成模型的完整逻辑**，可用于推断、生成、数据增强等任务。



从数学角度比较：

1. **AE 的目标函数（重构损失）**

   AE 的优化目标是最小化输入 $x$ 与重构 $  \hat{x} $ 的误差（如均方误差或交叉熵）：$\mathcal{L}_{\text{AE}} = \mathbb{E}_{x \sim p_{\text{data}}(x)} \| x - g(f(x)) \|^2$

   - 编码器  $f(x)$  输出确定的 $z$，解码器  $g(z) $ 直接映射回  $\hat{x}$ 。
   - **缺陷**：隐变量 $z$ 的分布无约束，可能出现 “坍缩”（不同 $x$ 对应相同 $z$）或 “不连续”（隐空间结构混乱）。

2. **VAE 的目标函数（证据下界，ELBO）**

   VAE 引入概率建模，目标是最大化数据对数似然  $\log p_\theta(x)$ 的下界（ELBO）： 

   $$
   \mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x|z) \right] - \text{KL}\left( q_\phi(z|x) \| p(z) \right)\\
   = \text{ELBO}
   $$
   物理意义，$\mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)]$ 表示重建项，$KL(q_\phi(z|x)||p_\theta(z))$ 表示正则化项。

   **重建项**：让生成数据贴近真实数据

   * 公式含义：在近似分布 $q (z|x)$ 下，对 “给定 $z$ 时 $x$ 的对数似然” 求期望。
   * 直观意义：要求模型从隐变量 $z$ 生成的 $x$（通过解码器 $p_\theta(x|z)$ ）尽可能接近真实 $x$，类似自编码器的 “重建误差”。
   * 常见形式：若 $x$ 是图像（像素值 0-1），$p_\theta(x|z)$ 常用伯努利分布，此时重建项等价于**二元交叉熵（BCE）**；若 $x$ 是连续值，常用高斯分布，对应**均方误差（MSE）**。

   **正则化项**：让 $q (z|x)$ 贴近先验 $p (z)$

   * 公式含义：KL 散度衡量近似后验 $q (z|x)$ 与先验分布 $p (z)$ 的差异，前面加负号表示 “最小化 KL 散度”。

   * 直观意义：强制隐变量 z 的分布接近一个简单的先验（通常设为标准正态分布 $\mathcal{N}(0,I)$ ），这能保证隐空间的 “连续性”（相似的 $z$ 生成相似的 $x$），避免过拟合，并让隐空间具备插值能力。

   * 计算简化：当 $q (z|x)$ 设为高斯分布 $\mathcal{N}(\mu(x), \sigma^2(x)I)$ 、$p (z)$ 设为 $\mathcal{N}(0,I)$ 时，KL 散度有解析解：

     $\text{KL}\left( q \parallel p \right) = \frac{1}{2} \sum_{d=1}^D \left( \mu_d^2 + \sigma_d^2 - \log \sigma_d^2 - 1 \right)$

     其中 D 是 $z$ 的维度，$\mu(x)$ 和 $\sigma(x)$ 由编码器输出。



# 二、代码



```python
import os

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image

# 设置随机种子，保证结果可复现
torch.manual_seed(666)
if torch.cuda.is_available():
    torch.cuda.manual_seed(666)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换成Tensor
    transforms.Lambda(lambda x: x.view(-1))  # 展平成向量，28×28 -> 784
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

# 数据加载器
batch_size = 128
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 均值和方差的输出层
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出值∈(0, 1)，符合MNIST图像像素值范围
        )

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：将采样过程转换为可导操作
        :param mu:
        :param logvar:
        :return: z = mu + eps * exp(logvar/2), 其中eps ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)  # 标准差
        eps = torch.randn_like(std)  # 从标准正态分布采样
        return mu + eps * std  # 返回采样的z

    def forward(self, x):
        """VAE的前向传播过程"""
        # 编码过程：得到均值和方差
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 解码过程：重建输入
        recon_x = self.decoder(z)

        return recon_x, mu, logvar


# 定义VAE损失函数（负ELBO）
def vae_loss(recon_x, x, mu, logvar):
    """
    计算VAE的损失函数
    :param recon_x: 重建的图像
    :param x: 原始图像
    :param mu: 均值向量
    :param logvar: 对数方差向量
    :return:
    """
    # 重建损失：二进制交叉熵
    recon_loss = nn.functional.binary_cross_entropy(
        recon_x, x, reduction='sum'
    )

    # KL散度损失：衡量q(z|x)与p(z)的差异
    # 解析解：KL(q|p) = 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())

    # 总损失 = 重建损失 + KL损失
    return recon_loss + kl_loss


# 初始化模型、优化器
input_dim = 784  # 28×28
hidden_dim = 400  # 隐层维度
latent_dim = 20  # 潜空间维度

model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 创建保存结果的目录
os.makedirs('vae_results', exist_ok=True)


# 训练模型
def train(epochs=10):
    model.train()  # 设置为训练模式
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)  # 将数据移到设备上

            # 前向传播
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)

            # 反向传播和优化
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item()

            # 打印训练进度
            if batch_idx % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item() / len(data):.4f}')

        # 计算平均损失
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}')

        # 每个epoch保存一次重建结果
        with torch.no_grad():
            # 取测试集前32个样本进行重建
            sample = next(iter(test_loader))[0][:32].to(device)
            recon, _, _ = model(sample)

            # 拼接原始图像和重建图像
            comparison = torch.cat([sample.view(-1, 1, 28, 28),
                                    recon.view(-1, 1, 28, 28)])
            save_image(comparison.cpu(),
                       f'vae_results/reconstruction_{epoch + 1}.png',
                       nrow=8)

    # 保存模型
    torch.save(model.state_dict(), 'vae_mnist.pth')
    print("模型已保存为'vae_mnist.pth'")


# 推理：生成新的手写数字
def generate(num_samples=32):
    model.load_state_dict(torch.load('vae_mnist.pth', map_location=device))
    model.eval()    # 设置为评估模式

    with torch.no_grad():
        # 从先验分布p(z) = N(0, 1)中采样
        z = torch.randn(num_samples, latent_dim).to(device)

        # 解码生成图像
        generated = model.decoder(z)

        # 保存生成的图像
        save_image(generated.view(num_samples, 1, 28, 28).cpu(),
                   'vae_results/generated_samples.png', nrow=8)
        print("生成的图像已保存为 'vae_results/generated_samples.png'")

        # 显示生成的图像
        plt.figure(figsize=(10, 10))
        for i in range(num_samples):
            plt.subplot(4, 8, i+1)
            plt.imshow(generated[i].cpu().view(28, 28), cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()


# 推理：查看重建效果
def visualize_reconstructions(num_samples=16):
    model.load_state_dict(torch.load('vae_mnist.pth', map_location=device))
    model.eval()    # 设置为评估模式

    with torch.no_grad():
        # 获取测试集样本
        data, _ = next(iter(test_loader))
        data = data[:num_samples].to(device)

        # 重建图像
        recon, _, _ = model(data)

        # 显示原始图像和重建图像
        plt.figure(figsize=(10, 4))
        for i in range(num_samples):
            # 原始图像
            plt.subplot(2, num_samples, i+1)
            plt.imshow(data[i].cpu().view(28, 28), cmap='gray')
            plt.axis('off')

            # 重建图像
            plt.subplot(2, num_samples, i+1+num_samples)
            plt.imshow(recon[i].cpu().view(28, 28), cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.show()


# 主函数
if __name__ == '__main__':
    # 训练模型 (10个epoch)
    # train(epochs=10)

    # 可视化重建效果
    visualize_reconstructions()

    # 生成新的手写数字
    generate()

```



# 三、参考

[变分自编码器（一）：原来是这么一回事 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/5253)

[变分自编码器（二）：从贝叶斯观点出发 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/5343)