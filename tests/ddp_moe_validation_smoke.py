"""2-process gloo proof: rank0-only eval has no hidden MoE collective; train reduce remains aligned."""

import os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch import nn
from ultralytics.nn.modules.moe.loss import differentiable_balance_loss, should_reduce_ddp


class TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(4, 2)

    def forward(self, x):
        probs = self.router(x).softmax(-1)
        usage = probs.detach().mean(0)
        return differentiable_balance_loss(probs, usage, 2, reduce_ddp=should_reduce_ddp(self))


def main():
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=15))
    try:
        model = TinyMoE()
        # Aligned training forward: both ranks enter the same two all-reduces.
        model.train()
        loss = model(torch.ones(2, 4))
        loss.backward()
        dist.barrier()
        # Rank0-only validation forward: eval+no_grad must remain local-only.
        model.eval()
        if rank == 0:
            with torch.no_grad():
                value = model(torch.ones(2, 4))
            assert torch.isfinite(value)
        dist.barrier()
        if rank == 0:
            print("Rank0-only MoE validation smoke passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
