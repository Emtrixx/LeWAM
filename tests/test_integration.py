import torch

from lewam.integrations.stable_worldmodel import LeWAMCost
from lewam.training.trainer import build_model


def test_stable_worldmodel_cost_contract(config):
    model = build_model(config).eval()
    cost = LeWAMCost(model)
    info = {
        "pixels": torch.rand(2, 3, 3, 3, 56, 56),
        "action_history": torch.rand(2, 3, 2, 10),
        "goal_emb": torch.rand(2, 1, 64),
    }
    values = cost.get_cost(info, torch.rand(2, 3, 2, 10))
    assert values.shape == (2, 3) and torch.isfinite(values).all()
