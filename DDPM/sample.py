import torch
from torchvision.utils import save_image
import os

from DDPM import GaussianDiffusionSampler
from UNet import UNet


def sample_images():
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() 
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f'使用设备: {device}')
    
    # 超参数（需与训练时保持一致）
    T = 1000
    beta_1 = 1e-4
    beta_T = 0.02
    image_size = 32
    image_channels = 3
    num_samples = 16  # 生成图片数量
    
    # 加载模型
    model = UNet(T).to(device)
    ddpm_sampler = GaussianDiffusionSampler(model, beta_1, beta_T, T).to(device)
    
    # 加载最新的检查点
    checkpoint_dir = 'checkpoints'
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'已加载模型: {checkpoint_path}')
    
    model.eval()
    with torch.no_grad():
        x_T = torch.randn(num_samples, image_channels, image_size, image_size).to(device)
        print('开始采样...')
        sampled_images = ddpm_sampler(x_T)
    
    os.makedirs('samples', exist_ok=True)
    save_path = 'samples/generated_images.png'
    save_image(sampled_images, save_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f'生成的图片已保存到: {save_path}')


if __name__ == '__main__':
    sample_images()
