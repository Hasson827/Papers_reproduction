import torch
import torch.nn.functional as F
import numpy as np
from VFM import VFMContinuousModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import generate_original_distribution, visualization


def train_vfm(model, device, optimizer, data, n_epochs=100000, batch_size=10000):
    """训练VFM模型"""
    losses = []
    
    for epoch in range(n_epochs):
        # 随机选择时间点t ~ Uniform(0,1)
        t = torch.rand(batch_size, 1).to(device)
        
        # 从初始分布p0采样x0 (标准正态分布)
        x0 = torch.randn(batch_size, 2).to(device)

        x1 = data[torch.randint(data.size(0), (batch_size,))].to(device)  # 从数据集中采样x1
        
        # 计算中间状态 x = (1-t)x0 + tx1
        x = (1 - t) * x0 + t * x1
        
        # 预测最终状态的分布参数
        mu = model(x, t)
        
        # VFM目标函数: -E[log q_θ(x1|x)]
        diff = (x1 - mu) ** 2
        loss = torch.mean(diff)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch [{epoch}/{n_epochs}], Loss: {loss.item():.4f}")
    
    return losses

def sample_vfm(model, n_samples=20000, n_steps=10, device='cuda'):
    """使用训练好的VFM模型生成样本"""
    # 从初始分布p0采样 (标准正态分布)
    x = torch.randn(n_samples, 2).to(device)

    # 从t=0到t=1进行数值积分
    dt = 1.0 / n_steps
    
    for i in range(n_steps):
        t = torch.ones(n_samples, 1).to(device) * (i * dt)
        
        # 预测向量场
        with torch.no_grad():
            vt = model.predict_vector_field(x, t)
        
        # 使用欧拉方法更新状态
        x = x + vt * dt
    
    return x.cpu().numpy()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() 
                        else "mps" if torch.mps.is_available()
                        else "cpu")
    print(f"使用设备: {device}")
    
    # 生成原始数据分布
    original_data = generate_original_distribution(N=20000, distribution_type="checkerboard")
    data = torch.Tensor(original_data).to(device)

    model = VFMContinuousModel(hidden_dim=1024).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # 训练模型
    losses = train_vfm(model, device, optimizer, data)

    # 生成样本
    samples_1 = sample_vfm(model, n_steps=1)
    samples_2 = sample_vfm(model, n_steps=2)
    samples_4 = sample_vfm(model, n_steps=4)
    samples_8 = sample_vfm(model, n_steps=8)
    visualization(original_data=original_data, generated_data=samples_1, losses=losses, model_name="Variational Flow Matching 1")
    visualization(original_data=original_data, generated_data=samples_2, losses=losses, model_name="Variational Flow Matching 2")
    visualization(original_data=original_data, generated_data=samples_4, losses=losses, model_name="Variational Flow Matching 4")
    visualization(original_data=original_data, generated_data=samples_8, losses=losses, model_name="Variational Flow Matching 8")