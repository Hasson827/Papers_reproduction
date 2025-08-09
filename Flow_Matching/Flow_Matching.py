import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from utils import generate_original_distribution
from utils import visualization
from utils import MLP

def train(model, device, data, optim, training_steps, batch_size):
    losses = []
    model.train()
    pbar = tqdm(range(training_steps))
    for i in pbar:
        x1 = data[torch.randint(data.size(0), (batch_size,))]
        x0 = torch.randn_like(x1).to(device)
        target = x1 - x0 # The Conditional Flow
        t = torch.rand(x1.size(0)).to(device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        pred = model(xt, t)
        loss = ((target - pred) ** 2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item())
    return losses, model

def sample(model, device, num_samples=10000, batch_size=100, steps=1000):
    all_samples = []
    num_batches = num_samples // batch_size
    
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="采样中"):
            xt = torch.randn(batch_size, 2).to(device)
            for _, t in enumerate(torch.linspace(0, 1, steps)):
                pred = model(xt, t.expand(xt.size(0)).to(device))
                xt = xt + (1 / steps) * pred
            all_samples.append(xt.cpu())
            del xt, pred
    samples = torch.cat(all_samples, dim=0)
    return samples.numpy()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() 
                        else "mps" if torch.mps.is_available()
                        else "cpu")

    original_data = generate_original_distribution(N=10000, distribution_type="checkerboard")
    data = torch.Tensor(original_data).to(device)

    model = MLP().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    training_steps = int(1e4)
    batch_size = 128

    losses, model = train(model, device, data, optim, training_steps, batch_size)
    samples = sample(model, device, num_samples=10000, batch_size=200, steps=1000)

    visualization(original_data=original_data, generated_data=samples, losses=losses, model_name="Flow Matching")