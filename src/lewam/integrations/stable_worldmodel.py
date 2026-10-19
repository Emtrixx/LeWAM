"""An action-space Costable adapter for stable-worldmodel's documented planning API.

Candidates and action_history must be z-scored *chunks*, not simulator primitive
actions. Pixel tensors must be encoder-sized float RGB in [0,1]. Configure SWM's
solver bounds in normalized chunk units; its PushT defaults are not interchangeable
with the absolute-coordinate LeRobot dataset. Noise-space DS uses our MPCPlanner.
"""

import torch
from torch import nn


class LeWAMCost(nn.Module):
    def __init__(self, model):
        super().__init__()
        if not model.has_dynamics:
            raise ValueError("The adapter needs trained dynamics")
        self.model = model

    @torch.no_grad()
    def get_cost(self, info_dict, action_candidates):
        """Return [B,S] costs for strictly future [B,S,H,chunk_dim] candidates.

        info_dict: pixels [B,S,K,3,h,w], action_history [B,S,K-1,A],
        and goal_emb [B,1,D]. Context and goals may be pre-encoded as emb/goal_emb.
        """
        b, samples, horizon, chunk_dim = action_candidates.shape
        if chunk_dim != self.model.chunk_dim or horizon < 1:
            raise ValueError("Expected future action chunks with the model's chunk_dim")
        pixels, emb = info_dict.get("pixels"), info_dict.get("emb")
        if emb is None:
            if pixels is None:
                raise ValueError("Provide pixels or pre-encoded emb")
            emb = self.model.encode(pixels[:, 0])[:, None].expand(-1, samples, -1, -1)
        context = emb.flatten(0, 1)
        history = info_dict["action_history"].flatten(0, 1)
        if history.shape[1] != context.shape[1] - 1:
            raise ValueError("Executed history must connect the context frames")
        actions = action_candidates.flatten(0, 1)
        for action in actions.unbind(1):
            predicted = self.model.predict_next(context, history, action)
            context = torch.cat((context[:, 1:], predicted[:, None]), 1)
            history = torch.cat((history, action[:, None]), 1)
            history = (
                history[:, -(context.shape[1] - 1) :] if context.shape[1] > 1 else history[:, :0]
            )
        goal = info_dict["goal_emb"][:, -1]
        return (context[:, -1].reshape(b, samples, -1) - goal[:, None]).float().norm(dim=-1)

    def criterion(self, info_dict, action_candidates):
        return self.get_cost(info_dict, action_candidates)
