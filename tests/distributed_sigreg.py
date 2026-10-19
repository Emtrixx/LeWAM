"""Run with torchrun to check global SIGReg values and gradients, independent of BN."""

import os

import torch
import torch.distributed as dist

from lewam.losses import SIGReg


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_num_threads(1)
    torch.manual_seed(11)
    global_z = torch.randn(4 * world, 2, 8, requires_grad=True)
    local = global_z.detach()[4 * rank : 4 * (rank + 1)].clone().requires_grad_()
    regularizer = SIGReg(knots=17, projections=32)
    torch.manual_seed(1729)
    loss = regularizer(local)
    loss.backward()
    torch.manual_seed(1729)
    directions = torch.randn(8, 32)
    directions = directions / directions.norm(dim=0)
    phase = (global_z.transpose(0, 1) @ directions).unsqueeze(-1) * regularizer.t
    error = (phase.cos().mean(1) - regularizer.phi).square() + phase.sin().mean(1).square()
    reference = ((error @ regularizer.weights) * global_z.shape[0]).mean()
    reference.backward()
    torch.testing.assert_close(loss, reference)
    # all_gather backward sums identical objectives across ranks; DDP averages
    # parameter gradients, cancelling this world-size factor in real training.
    torch.testing.assert_close(local.grad / world, global_z.grad[4 * rank : 4 * (rank + 1)])
    if rank == 0:
        print(f"SIGReg global value and gradient parity passed ({world} ranks, pid {os.getpid()})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
