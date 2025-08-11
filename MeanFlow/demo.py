import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from accelerate import Accelerator

from utils import generate_original_distribution
from utils import visualization
from utils import MLP
from meanflow import MeanFlow, triangular_MeanFlow

def train(model, meanflow_obj, data, optim, accelerator, training_steps, batch_size):
    model.train()
    # initialize loss history
    losses = []
    pbar = tqdm(range(training_steps))
    
    for i in pbar:
        # 随机采样数据点
        indices = torch.randint(data.size(0), (batch_size,))
        x = data[indices]

        loss = meanflow_obj.loss(model, x)

        accelerator.backward(loss)
        optim.step()
        optim.zero_grad()
        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item())
    
    return losses   

if __name__ == "__main__":
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using Accelerator device: {device}")

    # 生成原始数据
    original_data = generate_original_distribution(N=20000, distribution_type="checkerboard")
    # move data to the correct device for training
    data = torch.Tensor(original_data).to(device)

    channels = 2
    flow_ratio = 0.25
    time_dist = ['lognorm', -0.4, 1.0]
    training_steps = int(3e4)
    batch_size = 2048
    
    meanflow_obj = MeanFlow(
        channels=channels,
        flow_ratio=flow_ratio,
        time_dist=time_dist,
    )
    
    model = MLP(channels_data=2, layers=10, channels=1024, channels_t=512).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-5)
    model, optim = accelerator.prepare(model, optim)
    losses = train(model, meanflow_obj, data, optim, accelerator, training_steps, batch_size)
    with torch.no_grad():
        samples_tensor = meanflow_obj.sample(model, device, n=20000, batch_size=5000)
        samples = samples_tensor.cpu().numpy()
    visualization(original_data=original_data, generated_data=samples, losses=losses, model_name="MeanFlow")