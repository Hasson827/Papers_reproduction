import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from tqdm import tqdm
from utils import generate_original_distribution
from utils import visualization
from utils import MLP

"""
Score Matching 基本原理说明：

核心思想: 学习数据分布的score function ∇log p(x)
使用Denoising Score Matching: L = E[||s_θ(x + σε, σ ) - (-ε/σ)||²]
其中 x ~ p_data, ε ~ N(0,I), s_θ是score网络
"""

def train_score_matching(model, device, data, optim, training_steps, batch_size, 
                       noise_levels=None):
    """
    训练Score Matching模型
    损失函数: L = E[||s_θ(x + σε, σ) - (-ε/σ)||²]
    """
    if noise_levels is None:
        noise_levels = torch.tensor([1.0, 0.5, 0.2, 0.1, 0.05, 0.01]).to(device)
    else:
        noise_levels = torch.tensor(noise_levels).to(device)
    
    losses = []
    model.train()
    pbar = tqdm(range(training_steps))
    
    for i in pbar:
        x_clean = data[torch.randint(data.size(0), (batch_size,))]
        
        # 随机选择噪声水平
        noise_level_idx = torch.randint(len(noise_levels), (batch_size,))
        sigma_batch = noise_levels[noise_level_idx].unsqueeze(1)
        
        # 添加噪声: x_noisy = x_clean + σ * ε
        epsilon = torch.randn_like(x_clean)
        x_noisy = x_clean + sigma_batch * epsilon
        
        # 真实score: -ε/σ
        true_score = -epsilon / sigma_batch
        
        # 网络预测
        t = (noise_level_idx.float() / (len(noise_levels) - 1)).to(device)
        pred_score = model(x_noisy, t)
        
        # Denoising Score Matching Loss
        loss = ((pred_score - true_score) ** 2).mean()
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item())
    
    return losses, noise_levels

def sample_annealed_langevin(model, device, noise_levels, num_samples=10000, 
                           batch_size=100, steps_per_level=200, step_size=0.01):
    """
    退火Langevin采样: 从高噪声逐步降至低噪声
    使用标准的Langevin动力学: x_{t+1} = x_t + α * s_θ(x_t,σ) + √(2α) * ε
    """
    all_samples = []
    num_batches = num_samples // batch_size
    
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="退火采样中"):
            # 从最大噪声水平的先验分布开始
            x = torch.randn(batch_size, 2).to(device) * noise_levels[0]
            
            # 逐级降噪 
            for i, sigma in enumerate(noise_levels):
                t = torch.full((batch_size,), i / (len(noise_levels) - 1)).to(device)
                alpha = step_size * (sigma ** 2)
                
                for _ in range(steps_per_level):
                    score = model(x, t)
                    noise = torch.randn_like(x)
                    x = x + alpha * score + torch.sqrt(2 * alpha) * noise
            
            all_samples.append(x.cpu())
    
    return torch.cat(all_samples, dim=0).numpy()

if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() 
                        else "mps" if torch.mps.is_available()
                        else "cpu")
    print(f"使用设备: {device}")

    # 生成原始数据分布
    original_data = generate_original_distribution(N=10000, distribution_type="checkerboard")
    data = torch.Tensor(original_data).to(device)

    # 创建模型
    model = MLP(channels_data=2, layers=5, channels=512).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 训练参数
    training_steps = int(1e5)
    batch_size = 256
    
    losses, noise_levels = train_score_matching(model, device, data, optim, 
                                              training_steps, batch_size)

    samples = sample_annealed_langevin(model, device, noise_levels, num_samples=10000, 
                                     batch_size=200, steps_per_level=200)
    
    visualization(original_data=original_data, generated_data=samples, 
                 losses=losses, model_name="Denoising Score Matching")