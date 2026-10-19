"""Sampling MPC in action space or the deterministic policy's Gaussian noise space."""

import math
import time

import torch


def project_shell(noise):
    """Project each chunk independently; give the zero vector a defined direction."""
    norm = noise.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(noise)
    fallback[..., 0] = math.sqrt(noise.shape[-1])
    scaled = noise * (math.sqrt(noise.shape[-1]) / norm.clamp_min(1e-8))
    return torch.where(norm > 1e-8, scaled, fallback)


class CMAState:
    """Positive-weight rank-mu CMA-ES with cumulative step-size adaptation.

    Distribution updates run on CPU; only batched model rollouts occupy the GPU.
    This keeps eigendecompositions portable to Apple MPS as well as CUDA.
    """

    def __init__(self, dimensions, elites):
        n = self.n = dimensions
        w = torch.log(torch.tensor(elites + 0.5)) - torch.log(torch.arange(1, elites + 1).float())
        self.weights = w / w.sum()
        self.mueff = float(1 / self.weights.square().sum())
        self.mean = torch.zeros(n)
        self.covariance = torch.eye(n)
        self.ps, self.pc = torch.zeros(n), torch.zeros(n)
        self.sigma, self.generation = 1.0, 0
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff)
        )
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    def factors(self):
        eigenvalues, vectors = torch.linalg.eigh(self.covariance)
        eigenvalues = eigenvalues.clamp_min(1e-8)
        return vectors * eigenvalues.sqrt(), (vectors * eigenvalues.rsqrt()) @ vectors.T

    def propose(self, samples, generator):
        factor, _ = self.factors()
        return self.mean + self.sigma * (
            torch.randn(samples, self.n, generator=generator) @ factor.T
        )

    def update(self, elite):
        old = self.mean
        self.mean = (self.weights[:, None] * elite).sum(0)
        delta = (self.mean - old) / self.sigma
        _, inverse = self.factors()
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (
            inverse @ delta
        )
        self.generation += 1
        hsig = float(
            self.ps.norm() / math.sqrt(1 - (1 - self.cs) ** (2 * self.generation))
            < (1.4 + 2 / (self.n + 1)) * self.chi
        )
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * delta
        y = (elite - old) / self.sigma
        rank_mu = (y.T * self.weights) @ y
        decay = 1 - self.c1 - self.cmu + self.c1 * (1 - hsig) * self.cc * (2 - self.cc)
        self.covariance = (
            decay * self.covariance + self.c1 * torch.outer(self.pc, self.pc) + self.cmu * rank_mu
        )
        self.covariance = (self.covariance + self.covariance.T) / 2
        self.sigma *= math.exp(self.cs / self.damps * (float(self.ps.norm()) / self.chi - 1))
        self.sigma = min(10.0, max(1e-4, self.sigma))


class MPCPlanner:
    def __init__(self, model, cfg, normalizer, action_low, action_high, seed=0):
        self.model, self.cfg, self.normalizer = model, dict(cfg), normalizer
        if cfg["space"] not in {"noise", "action"} or cfg["solver"] not in {"cem", "mppi", "cmaes"}:
            raise ValueError("Unsupported planning space or optimizer")
        if not model.has_dynamics:
            raise ValueError("Planning requires a model trained for dynamics")
        if cfg["space"] == "noise" and not model.has_policy:
            raise ValueError("Noise-space planning requires a trained policy")
        if not 1 <= cfg["elites"] <= cfg["samples"] or min(cfg["iterations"], cfg["horizon"]) < 1:
            raise ValueError("Invalid planner budget")
        if cfg["temperature"] <= 0 or cfg["candidate_batch_size"] < 1:
            raise ValueError("Positive temperature and candidate batch size required")
        self.generator = torch.Generator().manual_seed(seed)
        self.action_low = torch.tensor(action_low).repeat(model.cfg.chunk_size)
        self.action_high = torch.tensor(action_high).repeat(model.cfg.chunk_size)

    def bound_actions(self, action):
        mean = torch.as_tensor(self.normalizer.mean, device=action.device)
        std = torch.as_tensor(self.normalizer.std, device=action.device)
        low = (self.action_low.to(action.device) - mean) / std
        high = (self.action_high.to(action.device) - mean) / std
        return action.clamp(low, high)

    @torch.no_grad()
    def score(self, candidates, context, history, goal):
        """Decode every noise chunk at its own imagined state, then advance history."""
        costs = []
        for batch in candidates.split(self.cfg["candidate_batch_size"]):
            batch = batch.to(context.device)
            n = len(batch)
            z = context.expand(n, -1, -1)
            past = history.expand(n, -1, -1)
            for chunk in batch.unbind(1):
                action = (
                    self.model.sample_actions(z, chunk) if self.cfg["space"] == "noise" else chunk
                )
                action = self.bound_actions(action)
                predicted = self.model.predict_next(z, past, action)
                z = torch.cat((z[:, 1:], predicted[:, None]), 1)
                past = (
                    torch.cat((past, action[:, None]), 1)[:, -(z.shape[1] - 1) :]
                    if z.shape[1] > 1
                    else past[:, :0]
                )
            costs.append((z[:, -1].float() - goal.float()).norm(dim=-1).cpu())
        return torch.cat(costs)

    @torch.no_grad()
    def plan(self, context, history, goal):
        if context.shape[0] != 1:
            raise ValueError("plan accepts one environment; candidates are evaluated in parallel")
        start = time.perf_counter()
        h, a = self.cfg["horizon"], self.model.chunk_dim
        mean, std = torch.zeros(h, a), torch.ones(h, a)
        cma = CMAState(h * a, self.cfg["elites"]) if self.cfg["solver"] == "cmaes" else None
        best, best_cost = None, float("inf")
        for _ in range(self.cfg["iterations"]):
            proposals = (
                cma.propose(self.cfg["samples"], self.generator).reshape(-1, h, a)
                if cma
                else mean + std * torch.randn(self.cfg["samples"], h, a, generator=self.generator)
            )
            if self.cfg["space"] == "noise" and self.cfg["project_noise"]:
                proposals = project_shell(proposals)
            elif self.cfg["space"] == "action":
                proposals = self.bound_actions(proposals)
            costs = self.score(proposals, context, history, goal)
            if not torch.isfinite(costs).all():
                raise FloatingPointError("Non-finite planner costs")
            order = costs.argsort()
            if float(costs[order[0]]) < best_cost:
                best, best_cost = proposals[order[0]].clone(), float(costs[order[0]])
            elite = proposals[order[: self.cfg["elites"]]]
            if cma:
                cma.update(elite.flatten(1))
            elif self.cfg["solver"] == "cem":
                mean = elite.mean(0)
                std = elite.std(0, unbiased=False).clamp_min(self.cfg["min_std"])
            else:
                weights = torch.softmax(-(costs - costs.min()) / self.cfg["temperature"], 0)
                mean = (weights[:, None, None] * proposals).sum(0)
                std = (weights[:, None, None] * (proposals - mean).square()).sum(0).sqrt()
                std = std.clamp_min(self.cfg["min_std"])
        selected = mean if self.cfg["solver"] == "mppi" else best
        if self.cfg["space"] == "noise" and self.cfg["project_noise"]:
            selected = project_shell(selected)
        first = selected[:1].to(context.device)
        action = (
            self.model.sample_actions(context, first) if self.cfg["space"] == "noise" else first
        )
        return self.bound_actions(action), {
            "cost": best_cost,
            "seconds": time.perf_counter() - start,
            "evaluations": self.cfg["samples"] * self.cfg["iterations"],
        }
