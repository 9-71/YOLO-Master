import os
from datetime import timedelta
import torch
import torch.distributed as dist
from ultralytics.nn.modules.moa.moa import _moa_router_aux_loss
from ultralytics.nn.modules.mot.mot import differentiable_balance_loss
from ultralytics.nn.modules.moe.loss import should_reduce_ddp
def main():
 r=int(os.environ['RANK']);dist.init_process_group('gloo',timeout=timedelta(seconds=20))
 try:
  l=torch.tensor([[[[3.]], [[-1.]]]] if r==0 else [[[[-1.]], [[3.]]]],requires_grad=True);_moa_router_aux_loss(l.softmax(1),l,1.).backward();assert torch.isfinite(l.grad).all()
  p=torch.tensor([[.9,.1]] if r==0 else [[.2,.8]],requires_grad=True);u=torch.tensor([1.,0.] if r==0 else [0.,3.]);differentiable_balance_loss(p,u,2,reduce_ddp=should_reduce_ddp()).backward();assert torch.isfinite(p.grad).all();dist.barrier()
  if r==0:
   with torch.no_grad(): _moa_router_aux_loss(l.softmax(1),l,1.);differentiable_balance_loss(p,u,2,reduce_ddp=should_reduce_ddp())
  dist.barrier()
  if r==0:print('MoA/MoT DDP smoke passed')
 finally:dist.destroy_process_group()
if __name__=='__main__':main()
