"""Contracts for additive registration of preserved mixture and PEFT settings."""

from pathlib import Path

import yaml

from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/governance/mixture-preservation-manifest.yaml"


def test_preserved_config_keys_are_registered_or_upstream_precision_aliases():
    keys = {item["name"] for item in yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["config_keys"]}
    assert keys - set(DEFAULT_CFG_DICT) == {"half", "int8"}


def test_adapter_runtime_metadata_can_flow_into_validator_args():
    args = get_cfg(overrides={"lora_r": 2, "lora_backend": "fallback"})
    runtime_fields = {
        "requested_lora_backend": "fallback",
        "requested_lora_variant": "lora",
        "requested_lora_init_lora_weights": True,
        "effective_lora_backend": "fallback",
        "effective_lora_variant": "lora",
        "effective_lora_type": "lora",
        "effective_lora_init_lora_weights": True,
        "lora_target_audit": {"selected_count": 1},
    }
    for key, value in runtime_fields.items():
        setattr(args, key, value)

    cloned = get_cfg(overrides=args)

    assert cloned.lora_r == 2
    for key, value in runtime_fields.items():
        assert getattr(cloned, key) == value


def test_lora_request_audit_fields_can_flow_into_validator_args():
    """Safety-audit request fields must not break validator config parsing."""
    overrides = {
        "lora_r": 16,
        "lora_backend": "peft",
        "lora_use_dora": False,
        "lora_use_rslora": True,
        "lora_lr_mult": 2.0,
        "lora_layer_decay": 0.85,
        "lora_alpha_warmup": 3,
        "lora_ortho_weight": 0.5,
        "requested_lora_use_dora": False,
        "requested_lora_use_rslora": True,
        "requested_lora_lr_mult": 2.0,
        "requested_lora_layer_decay": 0.85,
        "requested_lora_alpha_warmup": 3,
        "requested_lora_ortho_weight": 0.5,
    }

    cloned = get_cfg(overrides=get_cfg(overrides=overrides))

    for key, value in overrides.items():
        assert getattr(cloned, key) == value


def test_custom_config_types_are_validated_additively():
    args = get_cfg(
        overrides={
            "lora_r": 4,
            "lora_dropout": 0.1,
            "lora_include_head": True,
            "lora_backend": "fallback",
            "moe_map_saturation_window_size": 3,
        }
    )

    assert args.lora_r == 4
    assert args.lora_dropout == 0.1
    assert args.lora_include_head is True
    assert args.lora_backend == "fallback"
    assert args.moe_map_saturation_window_size == 3


def test_lora_init_accepts_bool_or_named_initialization():
    assert get_cfg(overrides={"lora_init_lora_weights": True}).lora_init_lora_weights is True
    assert get_cfg(overrides={"lora_init_lora_weights": "gaussian"}).lora_init_lora_weights == "gaussian"
