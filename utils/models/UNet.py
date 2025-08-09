import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        super().__init__()
        # 创建位置编码
        emb = torch.arange(0, d_model, step=2).float() / d_model * math.log(10000.0)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=True),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t):
        return self.time_embedding(t)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class UNet(nn.Module):
    def __init__(self, T=1000, image_channels=3, base_channels=128, time_emb_dim=512):
        super().__init__()
        
        # Time embedding
        self.time_embedding = TimeEmbedding(T, base_channels, time_emb_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        
        # Encoder (Downsampling)
        self.enc1 = ResBlock(base_channels, base_channels, time_emb_dim)
        self.enc2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.enc3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        
        self.down1 = nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        
        # Middle
        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        
        self.dec1 = ResBlock(base_channels * 8, base_channels * 4, time_emb_dim)  # skip connection
        self.dec2 = ResBlock(base_channels * 4, base_channels * 2, time_emb_dim)  # skip connection  
        self.dec3 = ResBlock(base_channels * 2, base_channels, time_emb_dim)      # skip connection
        
        # Final output
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, 3, padding=1)
        )
        
    def forward(self, x, t):
        # Time embedding
        time_emb = self.time_embedding(t)
        
        # Initial
        x = self.init_conv(x)
        
        # Encoder with skip connections
        enc1 = self.enc1(x, time_emb)
        x = self.down1(enc1)
        
        enc2 = self.enc2(x, time_emb)  
        x = self.down2(enc2)
        
        enc3 = self.enc3(x, time_emb)
        
        # Middle
        x = self.mid1(enc3, time_emb)
        x = self.mid2(x, time_emb)
        
        # Decoder with skip connections
        x = self.up1(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec1(x, time_emb)
        
        x = self.up2(x)  
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x, time_emb)
        
        x = torch.cat([x, enc1], dim=1)
        x = self.dec3(x, time_emb)
        
        return self.final(x)