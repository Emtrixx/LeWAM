from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lewam.data.windows import ActionNormalizer
from lewam.planning import MPCPlanner, project_shell


class LinearModel:
    chunk_dim = 2
    cfg = SimpleNamespace(chunk_size=1, action_dim=2)
    has_policy = has_dynamics = True

    def sample_actions(self, context, noise):
        return noise * 0.3 + context[:, -1] * 0.01

    def predict_next(self, context, history, action):
        assert history.shape[1] == context.shape[1] - 1
        return context[:, -1] + action


def test_projection_is_per_chunk_and_handles_zero():
    x = torch.randn(8, 5, 10)
    x[0, 0] = 0
    projected = project_shell(x)
    torch.testing.assert_close(projected.norm(dim=-1), torch.full((8, 5), 10**0.5))
    assert torch.isfinite(projected).all()


@pytest.mark.parametrize("solver", ["cem", "mppi", "cmaes"])
@pytest.mark.parametrize("space", ["noise", "action"])
def test_planners_run_same_budget_and_respect_bounds(config, solver, space):
    cfg = dict(
        config["planner"],
        solver=solver,
        space=space,
        samples=32,
        iterations=3,
        elites=8,
        horizon=2,
        candidate_batch_size=7,
    )
    normalizer = ActionNormalizer(np.zeros(2, dtype="float32"), np.ones(2, dtype="float32"))
    planner = MPCPlanner(LinearModel(), cfg, normalizer, [-1, -1], [1, 1], seed=5)
    z, history, goal = torch.zeros(1, 3, 2), torch.zeros(1, 2, 2), torch.full((1, 2), 0.3)
    action, metrics = planner.plan(z, history, goal)
    assert action.shape == (1, 2) and torch.isfinite(action).all()
    assert action.abs().max() <= 1
    assert metrics["evaluations"] == 96 and metrics["cost"] < 0.5


def test_rollout_decodes_at_imagined_state_and_shifts_actions(config):
    seen = []

    class Recording(LinearModel):
        def sample_actions(self, context, noise):
            seen.append(context.clone())
            return torch.ones_like(noise) * 0.1

        def predict_next(self, context, history, action):
            if len(seen) == 2:
                torch.testing.assert_close(history[:, -1], torch.ones_like(action) * 0.1)
            return super().predict_next(context, history, action)

    normalizer = ActionNormalizer(np.zeros(2, dtype="float32"), np.ones(2, dtype="float32"))
    planner = MPCPlanner(Recording(), config["planner"], normalizer, [-1, -1], [1, 1])
    planner.score(
        torch.zeros(2, 2, 2), torch.zeros(1, 3, 2), torch.zeros(1, 2, 2), torch.zeros(1, 2)
    )
    torch.testing.assert_close(seen[1][:, -1], torch.ones(2, 2) * 0.1)
