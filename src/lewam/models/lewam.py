"""The four masking modes and their joint loss live together for easy auditing."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from lewam.losses import SIGReg

from .encoder import FrameEncoder
from .transformer import Block, ConditionalBlock, SigmaEmbedding


@dataclass
class ModelConfig:
    kind: str = "lewam"
    image_size: int = 224
    patch_size: int = 14
    encoder_width: int = 192
    encoder_depth: int = 12
    encoder_heads: int = 3
    encoder_mlp_ratio: int = 4
    layer_scale: float = 1.0
    projector_hidden: int = 2048
    latent_dim: int = 384
    width: int = 512
    depth: int = 8
    heads: int = 16
    head_dim: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.0
    context_frames: int = 3
    chunk_size: int = 5
    action_dim: int = 2
    flow_steps: int = 8
    activation_checkpointing: bool = False

    def __post_init__(self):
        if self.kind not in {"lewam", "lewm", "policy"}:
            raise ValueError(f"Unknown model kind: {self.kind}")
        positive = (
            self.image_size,
            self.patch_size,
            self.encoder_width,
            self.encoder_depth,
            self.encoder_heads,
            self.projector_hidden,
            self.latent_dim,
            self.width,
            self.depth,
            self.heads,
            self.head_dim,
            self.mlp_dim,
            self.context_frames,
            self.chunk_size,
            self.action_dim,
            self.flow_steps,
        )
        if min(positive) < 1 or self.width % 2 or self.image_size % self.patch_size:
            raise ValueError(
                "Positive dimensions, even width, and patch-divisible image size required"
            )
        if self.encoder_width % self.encoder_heads:
            raise ValueError("Encoder width must be divisible by encoder heads")


class LeWAM(nn.Module):
    def __init__(self, cfg: ModelConfig, weights=None, sigreg=None, random_context=True):
        super().__init__()
        self.cfg = cfg
        self.weights = weights or {"forward": 2.0, "backward": 1.0, "inverse": 1.0, "policy": 1.0}
        self.sigreg_weight = (sigreg or {}).get("weight", 0.36)
        self.sigreg = SIGReg(**{k: v for k, v in (sigreg or {}).items() if k != "weight"})
        self.random_context = random_context
        self.encoder_frozen = False
        self.encoder = FrameEncoder(cfg)
        self.chunk_dim = cfg.action_dim * cfg.chunk_size
        self.frame_in = nn.Linear(cfg.latent_dim, cfg.width)
        self.action_in = nn.Linear(self.chunk_dim, cfg.width)
        self.position = nn.Parameter(torch.randn(1, cfg.context_frames + 1, cfg.width) * 0.02)
        self.modality = nn.Parameter(torch.randn(2, cfg.width) * 0.02)
        self.mask = nn.Parameter(torch.randn(cfg.width) * 0.02)
        self.unknown_action = nn.Parameter(torch.randn(cfg.width) * 0.02)
        self.sigma = SigmaEmbedding(cfg.width)
        self.blocks = (
            nn.ModuleList(
                [
                    Block(cfg.width, cfg.heads, cfg.head_dim, cfg.mlp_dim, cfg.dropout)
                    for _ in range(cfg.depth)
                ]
            )
            if cfg.kind != "lewm"
            else nn.ModuleList()
        )
        self.norm = nn.LayerNorm(cfg.width)
        self.latent_head = nn.Linear(cfg.width, cfg.latent_dim)
        self.action_head = nn.Linear(cfg.width, self.chunk_dim)
        self.causal_blocks = (
            nn.ModuleList(
                [
                    ConditionalBlock(cfg.width, cfg.heads, cfg.head_dim, cfg.mlp_dim, cfg.dropout)
                    for _ in range(cfg.depth)
                ]
            )
            if cfg.kind == "lewm"
            else nn.ModuleList()
        )
        if cfg.kind == "lewm":
            self.pred_projector = nn.Sequential(
                nn.Linear(cfg.width, cfg.projector_hidden),
                nn.BatchNorm1d(cfg.projector_hidden),
                nn.GELU(),
                nn.Linear(cfg.projector_hidden, cfg.latent_dim),
            )

    @property
    def has_policy(self):
        return self.cfg.kind in {"lewam", "policy"}

    @property
    def has_dynamics(self):
        return self.cfg.kind in {"lewam", "lewm"}

    def freeze_encoder(self):
        self.encoder_frozen = True
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()  # Also freeze BatchNorm running statistics.
        return self

    def encode(self, pixels):
        return self.encoder(pixels)

    def tokens(self, z, actions, mode, sigma=None):
        """Build interleaved [z0,a0,z1,a1,...,zK] tokens without mutating the targets."""
        b, frames, _ = z.shape
        if actions.shape != (b, frames - 1, self.chunk_dim):
            raise ValueError("One action chunk is required between each pair of frames")
        if mode not in {"forward", "backward", "inverse", "policy"}:
            raise ValueError(mode)
        f = self.frame_in(z)
        if mode in {"forward", "policy"}:
            f = torch.cat((f[:, :-1], self.mask.expand(b, 1, -1)), 1)
        elif mode == "backward":
            f = torch.cat((self.mask.expand(b, 1, -1), f[:, 1:]), 1)
        a = self.action_in(actions)
        if mode == "policy" and frames > 2:
            a = torch.cat((self.unknown_action.expand(b, frames - 2, -1), a[:, -1:]), 1)
        # Only the target chunk carries a nonzero flow time; clean actions have sigma=0.
        times = actions.new_zeros(b, frames - 1)
        if sigma is not None:
            times[:, -1] = sigma
        a = a + self.sigma(times.flatten()).reshape(b, frames - 1, -1)
        f = f + self.position[:, :frames] + self.modality[0]
        a = a + self.position[:, : frames - 1] + self.modality[1]
        paired = torch.stack((f[:, :-1], a), 2).flatten(1, 2)
        return torch.cat((paired, f[:, -1:]), 1)

    def predict_mode(self, z, actions, mode, sigma=None):
        x = self.tokens(z, actions, mode, sigma)
        for block in self.blocks:
            x = (
                checkpoint(block, x, use_reentrant=False)
                if (self.cfg.activation_checkpointing and self.training)
                else block(x)
            )
        x = self.norm(x)
        frame_index = 0 if mode == "backward" else -1
        return self.latent_head(x[:, frame_index]), self.action_head(x[:, -2])

    def lewm_predict(self, context, actions):
        x = self.frame_in(context) + self.position[:, : context.shape[1]]
        a = self.action_in(actions)
        for block in self.causal_blocks:
            x = (
                checkpoint(block, x, a, use_reentrant=False)
                if (self.cfg.activation_checkpointing and self.training)
                else block(x, a)
            )
        x = self.norm(x)
        return self.pred_projector(x.flatten(0, 1)).reshape(*x.shape[:2], -1)

    def forward(self, batch):
        z_all = self.encode(batch["pixels"])
        k = self.cfg.context_frames
        if self.training and self.random_context:
            k = int(torch.randint(1, k + 1, ()).item())
        z, actions = z_all[:, -(k + 1) :], batch["actions"][:, -k:]
        losses = {}
        if self.cfg.kind == "lewm":
            losses["forward"] = F.mse_loss(
                self.lewm_predict(z[:, :-1], actions).float(), z[:, 1:].float()
            )
        else:
            for mode, weight in self.weights.items():
                if weight == 0 or (self.cfg.kind == "policy" and mode != "policy"):
                    continue
                noisy, sigma, target = actions, None, None
                if mode in {"inverse", "policy"}:
                    sigma = torch.rand(z.shape[0], device=z.device)
                    epsilon = torch.randn_like(actions[:, -1])
                    target = epsilon - actions[:, -1]
                    mixed = (1 - sigma[:, None]) * actions[:, -1] + sigma[:, None] * epsilon
                    noisy = torch.cat((actions[:, :-1], mixed[:, None]), 1)
                latent, velocity = self.predict_mode(z, noisy, mode, sigma)
                if mode in {"forward", "backward"}:
                    losses[mode] = F.mse_loss(
                        latent.float(), z[:, 0 if mode == "backward" else -1].float()
                    )
                else:
                    losses[mode + "_flow"] = F.mse_loss(velocity.float(), target.float())
                    losses[mode] = losses[mode + "_flow"]
                    if mode == "policy" and self.cfg.kind == "lewam":
                        losses["policy_latent"] = F.mse_loss(latent.float(), z[:, -1].float())
                        losses[mode] = losses[mode] + losses["policy_latent"]
        total = sum(
            value * self.weights[key] for key, value in losses.items() if key in self.weights
        )
        if self.sigreg_weight:
            losses["sigreg"] = self.sigreg(z_all)
            total = total + self.sigreg_weight * losses["sigreg"]
        losses["loss"] = total
        losses["latent_std"] = z_all.float().std(dim=0, unbiased=False).mean().detach()
        return losses

    def predict_next(self, context, action_history, action):
        if not self.has_dynamics:
            raise ValueError("The policy-only model has no trained dynamics")
        actions = torch.cat((action_history, action[:, None]), 1)
        if self.cfg.kind == "lewm":
            return self.lewm_predict(context, actions)[:, -1]
        z = torch.cat((context, torch.zeros_like(context[:, -1:])), 1)
        return self.predict_mode(z, actions, "forward")[0]

    def sample_actions(self, context, noise=None, steps=None):
        if not self.has_policy:
            raise ValueError("Forward-only LeWM has no policy; use action-space planning")
        steps = self.cfg.flow_steps if steps is None else steps
        if steps < 1:
            raise ValueError("Euler steps must be positive")
        b, t, _ = context.shape
        action = torch.randn(b, self.chunk_dim, device=context.device) if noise is None else noise
        if action.shape != (b, self.chunk_dim):
            raise ValueError("Noise must have shape [batch, chunk_dim]")
        z = torch.cat((context, torch.zeros_like(context[:, -1:])), 1)
        past = action.new_zeros(b, t - 1, self.chunk_dim)
        # da/dsigma = epsilon-a, integrate from sigma=1 down to 0.
        for i in range(steps):
            sigma = action.new_full((b,), 1 - i / steps)
            _, velocity = self.predict_mode(
                z, torch.cat((past, action[:, None]), 1), "policy", sigma
            )
            action = action - velocity / steps
        return action
