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
    
    def forward(self, x: Tensor, t: Tensor, r: Tensor) -> Tensor:
        x = self.in_projection(x)
        t = self.gen_t_embedding(t)
        r = self.gen_t_embedding(r)
        t = self.t_projection(t)
        r = self.t_projection(r)
        x = x + t + r
        x = self.blocks(x)
        x = self.out_projection(x)
        return x