"""SIGReg: random-projection characteristic-function test against N(0,I).

Uses the Epps-Pulley quadrature from LeJEPA/LeWM. Statistics are computed across
the global batch separately at each frame position, with differentiable gathering.
"""

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_gather


class SIGReg(nn.Module):
    def __init__(self, knots=17, projections=1024):
        super().__init__()
        if knots < 2 or projections < 1:
            raise ValueError("SIGReg requires knots >= 2 and projections >= 1")
        self.projections = projections
        t = torch.linspace(0, 3, knots)
        weights = torch.full((knots,), 3 / (knots - 1))
        weights[1:-1] *= 2
        phi = (-t.square() / 2).exp()
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, z):
        """z: [B,T,D]. No stop-gradient, including through the distributed statistics."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            if dist.is_initialized():
                # Flatten before gathering so backward collective outputs are contiguous.
                # PyTorch 2.8's Gloo all-gather backward preserves noncontiguous gradient
                # layouts with empty_like, which can scramble a transposed [B,T,D] gradient.
                shape = z.shape
                z = torch.cat(all_gather(z.reshape(-1)), dim=0).reshape(-1, *shape[1:])
            projections = torch.randn(z.shape[-1], self.projections, device=z.device)
            if dist.is_initialized():
                dist.broadcast(projections, src=0)
            projections = projections / projections.norm(dim=0).clamp_min(1e-8)
            phase = (z.transpose(0, 1) @ projections).unsqueeze(-1) * self.t
            # [T,B,P,K] -> expectations over B, not over frames or feature dimensions.
            error = (phase.cos().mean(1) - self.phi).square() + phase.sin().mean(1).square()
            return ((error @ self.weights) * z.shape[0]).mean()
