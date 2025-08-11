import torch
from einops import rearrange
from functools import partial
import numpy as np
import math
from torch.nn.parallel import DistributedDataParallel

def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=0.25, c=1e-2):
    delta_sq = torch.mean(error ** 2, dim=1, keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    result = (stopgrad(w) * delta_sq).mean()
    return result

class MeanFlow:
    def __init__(self, channels, flow_ratio, time_dist):
        self.channels = channels
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
    
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r
    
    @torch.no_grad()
    def sample(self, model, device, n, batch_size=100):
        points = []
        while len(points) < n:
            cur_bs = min(batch_size, n - len(points))
            z = torch.randn(cur_bs, self.channels, device=device)
            t = torch.ones(cur_bs, device=device)
            r = torch.zeros(cur_bs, device=device)
            t_ = rearrange(t, 'B -> B 1').detach().clone()
            r_ = rearrange(r, 'B -> B 1').detach().clone()
            u = model(z, t, r)
            z = z - (t_ - r_) * u
            points.append(z.cpu())
        return torch.cat(points, dim=0)[:n]

    def loss(self, model, x):
        batch_size = x.shape[0]
        device = x.device
        # unwrap DDP if needed for functorch jvp
        model = model.module if isinstance(model, DistributedDataParallel) else model

        # sample time variables
        t, r = self.sample_t_r(batch_size, device)
        t_ = rearrange(t, 'B -> B 1').detach().clone()
        r_ = rearrange(r, 'B -> B 1').detach().clone()

        # compute noisy input
        e = torch.randn_like(x)
        z = (1 - t_) * x + t_ * e
        v = e - x

        # JVP using base_model
        jvp_args = (
            lambda z, t, r: model(z, t, r),
            (z, t, r),
            (v, torch.ones_like(t), torch.zeros_like(r))
        )
        u, dudt = self._jvp(*jvp_args)

        # compute target and return loss
        u_tgt = v - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        return adaptive_l2_loss(error)

    @staticmethod
    def _jvp(fn, inputs, tangents):
        try:
            return torch.func.jvp(fn, inputs, tangents)
        except (ImportError, AttributeError):
            return torch.autograd.functional.jvp(fn, inputs, tangents, create_graph=True)