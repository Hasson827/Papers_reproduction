import math
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import tqdm

N = 10000 # # 3. 生成的样本分布
x_min, x_max = -4, 4
y_min, y_max = -4, 4
resolution = 100 # Resolution of the grid

x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)

length = 4
checkerboard = np.indices((length, length)).sum(axis=0) % 2

sampled_points = []
while len(sampled_points) < N:
    x_sample = np.random.uniform(x_min, x_max)
    y_sample = np.random.uniform(y_min, y_max)
    
    i = int((x_sample - x_min) / (x_max - x_min) * length)
    j = int((y_sample - y_min) / (y_max - y_min) * length)
    
    if checkerboard[j, i] == 1:
        sampled_points.append((x_sample, y_sample))

sampled_points = np.array(sampled_points)

class Block(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.ff = nn.Linear(channels, channels)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.ff(x))

class MLP(nn.Module):
    def __init__(self, channels_data: int = 2, layers: int = 5, channels: int = 512, channels_t: int = 512):
        super().__init__()
        self.channels_t = channels_t
        self.in_projection = nn.Linear(channels_data, channels)
        self.t_projection = nn.Linear(channels_t, channels)
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(layers)])
        self.out_projection = nn.Linear(channels, channels_data)

    def gen_t_embedding(self, t: Tensor, max_positions: int = 10000) -> Tensor:
        t = t * max_positions
        half_dim = self.channels_t // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.channels_t % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.in_projection(x)
        t = self.gen_t_embedding(t)
        t = self.t_projection(t)
        x = x + t
        x = self.blocks(x)
        x = self.out_projection(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.mps.is_available()
                      else "cpu")

# 内存优化设置
if device.type == 'mps':
    # 设置MPS内存分配策略
    torch.mps.set_per_process_memory_fraction(0.8)
model = MLP().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

data = torch.Tensor(sampled_points).to(device)
training_steps = int(1e5)
batch_size = 64
pbar = tqdm.tqdm(range(training_steps))
losses = []
for i in pbar:
    x1 = data[torch.randint(data.size(0), (batch_size,))]
    x0 = torch.randn_like(x1).to(device)
    target = x1 - x0
    t = torch.rand(x1.size(0)).to(device)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    pred = model(xt, t)
    loss = ((target - pred) ** 2).mean()
    loss.backward()
    optim.step()
    optim.zero_grad()
    pbar.set_postfix(loss=loss.item())
    losses.append(loss.item())    
    if i % 1000 == 0 and device.type == 'mps':
        torch.mps.empty_cache()

def sample_in_batches(model, num_samples=10000, batch_size=100, steps=1000, device=device):
    """分批采样以节省内存"""
    all_samples = []
    num_batches = num_samples // batch_size

    model.eval()
    with torch.no_grad():
        for batch_idx in tqdm.tqdm(range(num_batches), desc="采样中"):
            xt = torch.randn(batch_size, 2).to(device)
            for i, t in enumerate(torch.linspace(0, 1, steps)):
                pred = model(xt, t.expand(xt.size(0)).to(device))
                xt = xt + (1 / steps) * pred
            all_samples.append(xt.cpu())
            del xt, pred
            if device.type == 'mps':
                torch.mps.empty_cache()
    
    return torch.cat(all_samples, dim=0)

generated_samples_tensor = sample_in_batches(model, num_samples=10000, batch_size=100)
generated_samples = generated_samples_tensor.numpy()

# 可视化结果
plt.figure(figsize=(15, 5))

# 1. 原始数据分布
plt.subplot(1, 3, 1)
plt.scatter(sampled_points[:, 0], sampled_points[:, 1], alpha=0.6, s=1, c='blue')
plt.title('Original Distribution')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True, alpha=0.3)

# 2. 训练损失曲线
plt.subplot(1, 3, 2)
plt.plot(losses)
plt.title('Training Loss Curve')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# 3. 生成的样本分布
plt.subplot(1, 3, 3)
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6, s=1, c='red')
plt.title('Generated Distributions')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('flow_matching_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"可视化结果已保存为 flow_matching_results.png")
print(f"原始数据点数: {len(sampled_points)}")
print(f"生成样本点数: {len(generated_samples)}")
print(f"最终损失: {losses[-1]:.6f}")