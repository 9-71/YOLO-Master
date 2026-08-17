"""Foundation recipe integrity checks."""

from pathlib import Path

from ultralytics.utils import YAML


RECIPE = (
    Path(__file__).resolve().parents[1]
    / "ultralytics/cfg/experiments/foundation/f08-foundation-distill-coco8-dinov3.yaml"
)
F09_RECIPE = (
    Path(__file__).resolve().parents[1]
    / "ultralytics/cfg/experiments/foundation/f09-foundation-foreground-coco8-dinov3.yaml"
)
F10_RECIPE = (
    Path(__file__).resolve().parents[1]
    / "ultralytics/cfg/experiments/foundation/f10-foundation-multiscale-coco8-dinov3.yaml"
)
F11_RECIPE = (
    Path(__file__).resolve().parents[1]
    / "ultralytics/cfg/experiments/foundation/f11-foundation-router-coco8-dinov3.yaml"
)
F12_RECIPE = (
    Path(__file__).resolve().parents[1] / "ultralytics/cfg/experiments/foundation/f12-foundation-siglip2-coco8.yaml"
)
F13_RECIPE = (
    Path(__file__).resolve().parents[1] / "ultralytics/cfg/experiments/foundation/f13-foundation-semantic-coco8.yaml"
)
F14_RECIPE = (
    Path(__file__).resolve().parents[1] / "ultralytics/cfg/experiments/foundation/f14-foundation-multirouter-coco8.yaml"
)
F15_RECIPE = (
    Path(__file__).resolve().parents[1] / "ultralytics/cfg/experiments/foundation/f15-foundation-multitask.yaml"
)


def test_f08_recipe_contains_executable_foundation_defaults():
    assert RECIPE.exists()
    recipe = YAML.load(RECIPE)

    assert recipe["model"] == "yolo26n.yaml"
    assert recipe["data"] == "coco8.yaml"
    assert recipe["foundation_enabled"] is True
    assert recipe["foundation_teacher"] == "dinov3"
    assert recipe["foundation_backend"] == "transformers"
    assert recipe["foundation_target_levels"] == ["p4"]
    assert recipe["foundation_model"] == "Tooony133/dinov3-vits16-pretrain-lvd1689m"
    assert recipe["foundation_teacher_dtype"] == "fp32"
    assert recipe["foundation_teacher_device"] == "cpu"
    assert recipe["foundation_loss"] == "hybrid"
    assert recipe["foundation_loss_weight"] > 0
    assert recipe["foundation_foreground_weighting"] is False


def test_f09_recipe_enables_gt_foreground_weighting():
    assert F09_RECIPE.exists()
    recipe = YAML.load(F09_RECIPE)
    assert recipe["foundation_foreground_weighting"] is True
    assert recipe["foundation_foreground_weight"] == 1.5
    assert recipe["foundation_boundary_weight"] == 1.0
    assert recipe["foundation_background_weight"] == 0.25


def test_f10_recipe_enables_independent_multiscale_targets():
    assert F10_RECIPE.exists()
    recipe = YAML.load(F10_RECIPE)
    assert recipe["foundation_multiscale"] is True
    assert recipe["foundation_target_levels"] == ["p3", "p4", "p5"]


def test_f11_recipe_enables_latent_foundation_router():
    assert F11_RECIPE.exists()
    recipe = YAML.load(F11_RECIPE)
    assert recipe["foundation_router_distill"] is True
    assert recipe["foundation_router_loss_weight"] > 0
    assert recipe["foundation_router_temperature"] == 2.0
    assert recipe["model"] == "yolo26-master-latent-n.yaml"


def test_f12_recipe_enables_siglip2_teacher():
    assert F12_RECIPE.exists()
    recipe = YAML.load(F12_RECIPE)
    assert recipe["foundation_teacher"] == "siglip2"
    assert recipe["foundation_model"] == "google/siglip2-base-patch16-512"
    assert recipe["foundation_loss_weight"] > 0


def test_f13_recipe_enables_positive_region_semantic_distillation():
    assert F13_RECIPE.exists()
    recipe = YAML.load(F13_RECIPE)
    assert recipe["foundation_teacher"] == "siglip2"
    assert recipe["foundation_semantic_distill"] is True
    assert recipe["foundation_semantic_loss_weight"] > 0
    assert recipe["foundation_semantic_text_weight"] == 1.0
    assert recipe["foundation_semantic_image_weight"] == 1.0


def test_f14_recipe_enables_ordered_multi_foundation_router():
    assert F14_RECIPE.exists()
    recipe = YAML.load(F14_RECIPE)
    assert recipe["foundation_teacher"] == "multi"
    assert recipe["foundation_router_distill"] is True
    assert recipe["foundation_router_teachers"] == ["dinov3", "siglip2"]
    assert recipe["foundation_router_native_state"] is True
    assert recipe["foundation_router_loss_weight"] > 0


def test_f15_recipe_requires_multitask_and_three_supervised_branches():
    assert F15_RECIPE.exists()
    recipe = YAML.load(F15_RECIPE)
    assert recipe["task"] == "multitask"
    assert recipe["foundation_multitask"] is True
    assert set(recipe["foundation_multitask_tasks"]) == {"detect", "segment", "pose"}
