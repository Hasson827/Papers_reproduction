import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from DDPM import GaussianDiffusionTrainer, GaussianDiffusionSampler
from utils import MLP, generate_original_distribution, visualization

def train(model, device, data, optimizer, scheduler, training_steps, batch_size):
    """
    Train the DDPM model with learning rate scheduling and improved stability.
    """
    losses = []
    model.train()
    trainer = GaussianDiffusionTrainer(model, beta_1=5e-5, beta_T=1e-2, T=1000).to(device)
    
    for step in tqdm(range(training_steps), desc="Training DDPM"):
        idx = torch.randint(0, data.size(0), (batch_size,))
        x0 = data[idx].to(device)
        
        # 更精细的数据归一化
        x0 = torch.clamp(x0 / 4.0, -0.95, 0.95)  # 避免过度归一化
        
        optimizer.zero_grad()
        loss = trainer(x0).mean()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    
    return losses


def sample(model, device, data_dim=2, num_samples=10000, batch_size=100):
    """
    Sample from the trained DDPM model using GaussianDiffusionSampler.
    """
    sampler = GaussianDiffusionSampler(model, beta_1=5e-5, beta_T=1e-2, T=1000).to(device)
    model.eval()
    all_samples = []
    num_batches = num_samples // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Sampling DDPM"):
            x_T = torch.randn(batch_size, data_dim).to(device)
            samples = sampler(x_T)
            all_samples.append(samples.cpu())
    
    # Convert back to original data range with proper scaling
    samples = torch.cat(all_samples, dim=0).numpy()
    samples = samples * 4.0  # Scale back to [-4, 4] range
    return samples


if __name__ == "__main__":
    # Setup device and data
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    
    original_data = generate_original_distribution(N=10000, distribution_type="checkerboard")  # 更多训练数据
    data = torch.tensor(original_data, dtype=torch.float32)

    # Initialize model and optimizer with better hyperparameters
    model = MLP(channels_data=2, layers=8, channels=1024, channels_t=256).to(device)  # 更大更深的模型
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)  # AdamW with weight decay

    # Training with more steps and cosine annealing
    training_steps = int(1e5)
    batch_size = 256
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_steps, eta_min=1e-5)
    
    print("Starting DDPM training...")
    losses = train(model, device, data, optimizer, scheduler, training_steps, batch_size)

    # Sampling
    print("Sampling from trained model...")
    samples = sample(model, device, data_dim=2, num_samples=10000, batch_size=200)

    # Visualization
    visualization(original_data=original_data, generated_data=samples, losses=losses, model_name="DDPM")
    visualization(original_data=original_data, generated_data=samples, losses=losses, model_name="DDPM")