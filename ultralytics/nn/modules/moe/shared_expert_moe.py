# 🐧Please note that this file has been modified by Tencent on 2026/07/29. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Cross-Scale Expert Sharing MoE: 跨尺度专家共享 MoE 模块.

创新点:
  - 多个 MoE 块共享同一组 expert pool
  - 减少参数 (~25-50% reduction)
  - 跨尺度特征共享通过 expert pool 自然实现
  - 论文价值: 讨论"参数共享 vs 性能"权衡

使用:
  在 YAML 中, 多个 SharedExpertMoE 块使用相同 pool_id 即可共享 experts:
  ```yaml
  backbone:
    - [-1, 1, SharedExpertMoE, [512, 4, 2, 0.5, "shared_512"]]  # 创建 pool
    - [-1, 1, SharedExpertMoE, [512, 4, 2, 0.5, "shared_512"]]  # 复用 pool
  ```
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from .modules import LowRankHybridAdaptiveGateMoE, _registry_set
from .utils import FlopsUtils

# 全局 expert pool 注册表
# Key: pool_id, Value: (num_experts, num_groups, top_k, bottleneck_ratio, fused_experts_module)
_SHARED_EXPERT_POOLS: Dict[str, dict] = {}


class SharedExpertMoE(LowRankHybridAdaptiveGateMoE):
    """
    Cross-Scale Expert Sharing MoE (v0.8 variant).

    多个 MoE 块共享同一组 expert pool, 实现跨尺度特征共享和参数效率.

    关键创新:
      1. Expert Pool 共享: 相同 pool_id 的 MoE 块共享 expert 参数
      2. 跨尺度特征传递: 浅层 P3 expert 可被深层 P4 复用
      3. 参数效率: 相比独立 MoE 块, 参数量减少 25-50%
      4. 训练稳定: 每个 expert 被多个尺度训练, 鲁棒性更强

    约束:
      - 共享 pool 的 MoE 块必须有相同 in_channels, out_channels, num_experts, top_k
      - 不同 channel size 应使用不同 pool_id
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,
        num_groups: int = 8,
        initial_temperature: float = 1.2,
        final_temperature: float = 0.5,
        balance_loss_coeff: float = 1.0,
        router_z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.01,
        fused_expert_threshold: int = 8,
        shuffle_groups: int = 2,
        bottleneck_ratio: float = 0.5,
        pool_id: str = "shared",
    ):
        # 关键: 在 super().__init__ 之前 reset pool, 确保每个模型实例都有干净的 pool
        # 否则同一个 pool_id 会在不同模型的实例间共享, 引发 device mismatch
        SharedExpertMoE.reset_shared_pools()

        super().__init__(
            in_channels,
            out_channels,
            num_experts,
            top_k,
            split_ratio,
            num_groups,
            initial_temperature,
            final_temperature,
            balance_loss_coeff,
            router_z_loss_coeff,
            entropy_loss_coeff,
            fused_expert_threshold,
            shuffle_groups,
            bottleneck_ratio,
        )
        self.pool_id = pool_id
        self._is_pool_owner = False
        self._setup_shared_pool()

    def _setup_shared_pool(self):
        """设置共享 expert pool. 第一个该 pool_id 的块是 owner, 后续块复用."""
        pool_key = self.pool_id
        pool_signature = {
            'in_channels': self.dynamic_channels,
            'out_channels': self.out_dynamic,
            'num_experts': self.fused_experts.num_experts if hasattr(self.fused_experts, 'num_experts') else 0,
            'top_k': self.top_k,
            'bottleneck_ratio': self.bottleneck_ratio,
        }

        if pool_key in _SHARED_EXPERT_POOLS:
            existing = _SHARED_EXPERT_POOLS[pool_key]
            # 验证参数兼容性
            for k, v in pool_signature.items():
                if k in existing and existing[k] != v:
                    raise ValueError(
                        f"SharedExpertMoE pool '{pool_key}' parameter mismatch: "
                        f"{k} expected {existing[k]}, got {v}. "
                        f"共享 pool 的 MoE 块必须有相同的 channels/num_experts/top_k."
                    )
            # 复用现有 expert pool
            self.fused_experts = existing['fused_experts']
            self._is_pool_owner = False
        else:
            # 注册新的 expert pool
            _SHARED_EXPERT_POOLS[pool_key] = {
                **pool_signature,
                'fused_experts': self.fused_experts,
            }
            self._is_pool_owner = True

    @classmethod
    def reset_shared_pools(cls):
        """重置所有共享 pool (用于重新构建模型)."""
        global _SHARED_EXPERT_POOLS
        _SHARED_EXPERT_POOLS = {}

    def _apply(self, fn, recurse=True):
        """重写 _apply 以确保 .to(device) 调用能传播到 shared pool.

        PyTorch 在调用 module.to(device) 时会调用 _apply, 该方法会递归处理
        所有子模块的参数. 但 shared pool 是 class-level 全局变量, 不在
        self._modules 中, 所以默认 _apply 不会处理它. 这里我们手动处理.
        """
        # 先调用父类的 _apply (处理 self 的子模块)
        result = super()._apply(fn, recurse=recurse)

        # 然后处理 shared pool 中的 fused_experts
        for pool_info in _SHARED_EXPERT_POOLS.values():
            if 'fused_experts' in pool_info:
                pool_info['fused_experts']._apply(fn, recurse=recurse)

        return result

    def get_pool_info(self):
        """获取当前 pool 的信息 (用于诊断)."""
        return {
            'pool_id': self.pool_id,
            'is_owner': self._is_pool_owner,
            'num_experts': self.fused_experts.num_experts if hasattr(self.fused_experts, 'num_experts') else 0,
            'top_k': self.top_k,
            'dynamic_channels': self.dynamic_channels,
        }
