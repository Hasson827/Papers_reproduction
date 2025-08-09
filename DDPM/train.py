import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm

from DDPM import GaussianDiffusionTrainer
from UNet import UNet


def get_dataloader(batch_size=32, image_size=32):
    """获取CIFAR-10数据加载器"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
    ])
    
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    return dataloader


def train_ddpm():
    """主训练函数"""
    # 超参数设置
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 50
    T = 1000  # 扩散步数
    beta_1 = 1e-4
    beta_T = 0.02
    save_interval = 10
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() 
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f'使用设备: {device}')
    
    dataloader = get_dataloader(batch_size=batch_size)
    print(f'数据集大小: {len(dataloader.dataset)}')
    
    model = UNet(T).to(device)
    ddpm_trainer = GaussianDiffusionTrainer(model, beta_1, beta_T, T).to(device)    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建保存目录
    os.makedirs('checkpoints', exist_ok=True)
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M')
    print('开始训练...')
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (x_0, _) in enumerate(pbar):
            x_0 = x_0.to(device)           
             
            loss = ddpm_trainer(x_0).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.6f}')
        
        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'hyperparams': {
                    'T': T,
                    'beta_1': beta_1,
                    'beta_T': beta_T,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size
                }
            }
            torch.save(checkpoint, f'checkpoints/ddpm_epoch_{epoch+1}.pth')
            print(f'模型已保存: epoch {epoch+1}')
    
    print('训练完成!')

if __name__ == '__main__':
    train_ddpm()
