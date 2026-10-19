"""DINOv2-style ViT-tiny/14 trained from scratch, with a single CLS latent per frame."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .transformer import Block


class FrameEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.image_size
        self.checkpoint_blocks = cfg.activation_checkpointing
        width = cfg.encoder_width
        self.patch = nn.Conv2d(3, width, cfg.patch_size, stride=cfg.patch_size)
        tokens = (cfg.image_size // cfg.patch_size) ** 2 + 1
        self.cls = nn.Parameter(torch.empty(1, 1, width))
        self.position = nn.Parameter(torch.empty(1, tokens, width))
        self.blocks = nn.ModuleList(
            [
                Block(
                    width,
                    cfg.encoder_heads,
                    width // cfg.encoder_heads,
                    width * cfg.encoder_mlp_ratio,
                    layer_scale=cfg.layer_scale,
                )
                for _ in range(cfg.encoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.projector = nn.Sequential(
            nn.Linear(width, cfg.projector_hidden),
            nn.BatchNorm1d(cfg.projector_hidden),
            nn.GELU(),
            nn.Linear(cfg.projector_hidden, cfg.latent_dim),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        nn.init.normal_(self.cls, std=1e-6)
        nn.init.trunc_normal_(self.position, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pixels):
        """pixels: [B,T,3,H,W], float [0,1]; no temporal mixing before the projector."""
        if pixels.ndim != 5 or pixels.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected [B,T,3,{self.image_size},{self.image_size}], got {pixels.shape}"
            )
        b, t = pixels.shape[:2]
        x = pixels.flatten(0, 1)
        x = (x - self.mean) / self.std
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls.expand(b * t, -1, -1), x), 1) + self.position
        for block in self.blocks:
            if self.checkpoint_blocks and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.projector(self.norm(x)[:, 0]).reshape(b, t, -1)
