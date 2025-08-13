import torch
import torch.nn as nn

class VFMContinuousModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        
        # 神经网络预测最终分布的均值和方差
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for time t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, input_dim)  # 输出均值
        )
    
    def forward(self, x, t):
        """预测给定x和t时, 最终状态x1的分布参数"""
        batch_size = x.shape[0]
        
        # 将时间t与输入x拼接
        t_expanded = t.view(-1, 1).expand(batch_size, 1)
        inputs = torch.cat([x, t_expanded], dim=1)

        mu = self.net(inputs)

        return mu
    
    def predict_vector_field(self, x, t):
        """预测向量场v_t(x) = E[q_θ(x1|x)][(x1 - x)/(1-t)]"""
        mu = self.forward(x, t)

        # 计算向量场 (注意处理t接近1的情况)
        epsilon = 1e-6
        vt = (mu - x) / (1 - t + epsilon)
        
        return vt