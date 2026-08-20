"""F07 checkpoint reconstruction and deployment stripping contracts."""

import copy
import io
from types import SimpleNamespace

import torch

from tests.test_foundation_distill_model import DummyTeacher, TinyStudent, config
from ultralytics.nn.foundation_distill_model import (
    FoundationDistillationModel,
    rebuild_foundation_distillation_wrapper,
)
from ultralytics.utils.checkpoint_compat import checkpoint_runtime_metadata
from ultralytics.utils.torch_utils import strip_optimizer
from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer
from ultralytics.engine.extensions import TrainingRecoveryController
from ultralytics.utils.torch_utils import ModelEMA


def test_foundation_checkpoint_metadata_is_additive_and_json_safe():
    wrapper = FoundationDistillationModel(TinyStudent(), DummyTeacher(), config())

    metadata = checkpoint_runtime_metadata(wrapper)

    assert metadata["schema_version"] == 1
    assert metadata["graph"]["model_class"].endswith("TinyStudent")
    assert metadata["foundation"]["training_only"] is True
    assert metadata["foundation"]["target_levels"] == ["p4"]
    assert metadata["foundation"]["student_channels"] == 16
    assert metadata["foundation"]["teacher_channels"] == 10
    assert metadata["foundation"]["align_dim"] == 4


def test_foundation_resume_rebuild_restores_projector_and_teacher_boundary():
    source = FoundationDistillationModel(TinyStudent(), DummyTeacher(), config())
    assert all(not parameter.requires_grad for parameter in source.projector.teacher_proj.parameters())
    with torch.no_grad():
        source.projector.student_proj[0].weight.fill_(0.125)
    checkpoint_model = copy.deepcopy(source)

    restored = rebuild_foundation_distillation_wrapper(
        TinyStudent(),
        config(),
        checkpoint_model=checkpoint_model,
        device="cpu",
        teacher_manager=DummyTeacher(),
    )

    assert isinstance(restored, FoundationDistillationModel)
    assert restored.teacher_manager is not checkpoint_model.teacher_manager
    assert restored.tap is not None
    assert torch.equal(restored.projector.student_proj[0].weight, source.projector.student_proj[0].weight)


def test_foundation_resume_rebuild_restores_cosine_gate_ema():
    source = FoundationDistillationModel(TinyStudent(), DummyTeacher(), config(foundation_weight_schedule="gate_decay"))
    source.__dict__["_cosine_ema"] = 0.93
    checkpoint_model = copy.deepcopy(source)

    restored = rebuild_foundation_distillation_wrapper(
        TinyStudent(),
        config(foundation_weight_schedule="gate_decay"),
        checkpoint_model=checkpoint_model,
        device="cpu",
        teacher_manager=DummyTeacher(),
    )

    assert restored.__dict__["_cosine_ema"] == 0.93
    # A pre-schedule checkpoint carries no EMA; the rebuilt wrapper must start fresh.
    checkpoint_model.__dict__["_cosine_ema"] = None
    restored_fresh = rebuild_foundation_distillation_wrapper(
        TinyStudent(),
        config(foundation_weight_schedule="gate_decay"),
        checkpoint_model=checkpoint_model,
        device="cpu",
        teacher_manager=DummyTeacher(),
    )
    assert restored_fresh.__dict__["_cosine_ema"] is None


def test_multiscale_checkpoint_metadata_records_per_level_channels():
    wrapper = FoundationDistillationModel(
        TinyStudent(), DummyTeacher(), config(foundation_multiscale=True, foundation_target_levels=["p3", "p4", "p5"])
    )
    metadata = checkpoint_runtime_metadata(wrapper)["foundation"]
    assert metadata["multiscale"] is True
    assert metadata["target_levels"] == ["p3", "p4", "p5"]
    assert metadata["student_channels"] == {"p3": 8, "p4": 16, "p5": 32}


def test_strip_optimizer_removes_foundation_components(tmp_path):
    wrapper = FoundationDistillationModel(TinyStudent(), DummyTeacher(), config())
    wrapper.student_model.task = "detect"
    path = tmp_path / "foundation.pt"
    torch.save({"model": wrapper, "ema": None, "train_args": {}}, path)

    payload = strip_optimizer(path)

    assert type(payload["model"]).__name__ == "TinyStudent"
    assert payload["model"].task == "detect"
    assert not any("projector" in key for key in payload["model"].state_dict())
    assert not any(layer._forward_hooks for layer in payload["model"].modules())


def test_recovery_checkpoint_contains_foundation_metadata():
    wrapper = FoundationDistillationModel(TinyStudent(), DummyTeacher(), config())
    trainer = SimpleNamespace(
        model=wrapper,
        optimizer=torch.optim.AdamW(wrapper.parameters(), lr=1e-3),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        ema=ModelEMA(wrapper),
        optimizer_steps=0,
        start_epoch=0,
        epoch=0,
        best_fitness=0.0,
        fitness=0.0,
        metrics={},
        args=SimpleNamespace(**vars(config())),
        read_results_csv=lambda: {},
        adapter_controller=None,
    )
    trainer.recovery_controller = TrainingRecoveryController(trainer)

    payload = torch.load(
        io.BytesIO(trainer.recovery_controller.serialize_checkpoint(include_online_model=True)),
        map_location="cpu",
        weights_only=False,
    )

    assert payload["foundation"]["training_only"] is True
    assert payload["mixture_checkpoint"]["foundation"]["align_dim"] == 4


def test_foundation_wrapper_resolves_student_mixture_ema_buffer():
    """Foundation wrappers keep the routed EMA buffer under their student module."""
    wrapper = FoundationDistillationModel(TinyStudent(), DummyTeacher(), config())
    expected = initialize_mixture_loss_ema_buffer(wrapper.student_model)

    assert torch.equal(initialize_mixture_loss_ema_buffer(wrapper), expected)
    assert "_mixture_loss_ema_buf" not in wrapper._buffers
    assert "_mixture_loss_ema_buf" in wrapper.student_model._buffers
