"""Tests for PEFT Planner enhancements (audit, cache, unwrap, metadata).

Covers features added in the optimization pass:
  - DecisionAudit serialization and persistence
  - ArchitectureFingerprint WeakKeyDictionary caching
  - _unwrap_model for DDP / DataParallel / compiled models
  - PlacementDecision.to_dict() for metadata propagation
  - Empty-model robust handling
"""

import json

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

from ultralytics.utils.lora.planner import (
    ArchitectureFingerprint,
    DecisionAudit,
    LOVODataCollector,
    LOVODataPoint,
    PEFTPlanner,
    PEFTVariantProfile,
    PlacementDecision,
    _fingerprint_cache,
)


# ============================================================================
# DecisionAudit
# ============================================================================


class TestDecisionAudit:
    """Test the structured audit record."""

    def test_to_dict(self):
        audit = DecisionAudit(
            timestamp="2026-07-05T12:00:00",
            model_name="yolo11s",
            fingerprint={"phi_attn": 0.0, "phi_text": 0.0},
            variant="lora",
            requested_rank=16,
            decision_status="ACCEPT",
            predicted_delta=0.0656,
            target_modules_count=42,
        )
        d = audit.to_dict()
        assert d["decision_status"] == "ACCEPT"
        assert d["predicted_delta"] == pytest.approx(0.0656)
        assert d["target_modules_count"] == 42

    def test_save_and_load(self, tmp_path):
        audit = DecisionAudit(
            timestamp="2026-07-05T12:00:00",
            model_name="test_model",
            fingerprint={"phi_attn": 0.0},
            variant="lora",
            requested_rank=8,
            decision_status="ADAPT",
            recommended_variant="loha",
            recommended_rank=4,
            predicted_delta=0.03,
            safety_overrides={"r": 4},
            target_modules_count=10,
        )
        path = audit.save(audit_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

        loaded = DecisionAudit.load(path)
        assert loaded.decision_status == "ADAPT"
        assert loaded.recommended_variant == "loha"
        assert loaded.safety_overrides == {"r": 4}

    def test_save_creates_directory(self, tmp_path):
        audit_dir = tmp_path / "nested" / "audit"
        audit = DecisionAudit(
            timestamp="2026-07-05T12:00:00",
            model_name="m",
            fingerprint={},
            variant="lora",
            requested_rank=0,
            decision_status="REFUSE",
            refusal_reason="test",
            target_modules_count=0,
        )
        path = audit.save(audit_dir=audit_dir)
        assert path.exists()


# ============================================================================
# ArchitectureFingerprint caching
# ============================================================================


class TestFingerprintCache:
    """Test the WeakKeyDictionary cache."""

    def test_cache_hit(self):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)

        model = _Model()
        fp1 = ArchitectureFingerprint.compute(model)
        fp2 = ArchitectureFingerprint.compute(model)
        assert fp1 is fp2  # same object (cached)

    def test_cache_auto_invalidation_on_gc(self):
        """WeakKeyDictionary drops entries when the model is GC'd."""

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)

        # Count before
        before = len(_fingerprint_cache)

        # Create model, compute fingerprint, then drop reference
        model = _Model()
        ArchitectureFingerprint.compute(model)
        assert len(_fingerprint_cache) == before + 1

        del model
        # Force gc so WeakKeyDictionary can clean up
        import gc

        gc.collect()

        assert len(_fingerprint_cache) == before

    def test_invalidate_cache_explicit(self):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)

        model = _Model()
        ArchitectureFingerprint.compute(model)
        assert model in _fingerprint_cache

        ArchitectureFingerprint.invalidate_cache(model)
        assert model not in _fingerprint_cache

    def test_cache_no_leak_across_different_instances(self):
        """Different model instances should not share cached fingerprints."""

        class _Model(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.conv = nn.Conv2d(3, n, 3)

        m1 = _Model(16)
        m2 = _Model(32)
        fp1 = ArchitectureFingerprint.compute(m1)
        fp2 = ArchitectureFingerprint.compute(m2)
        assert fp1 is not fp2


# ============================================================================
# _unwrap_model
# ============================================================================


class TestUnwrapModel:
    """Test DDP / DataParallel / torch.compile unwrapping."""

    def test_plain_model(self):
        m = nn.Conv2d(3, 16, 3)
        assert ArchitectureFingerprint._unwrap_model(m) is m

    def test_dataparallel(self):
        m = nn.Sequential(nn.Conv2d(3, 16, 3))
        dp = DataParallel(m)
        unwrapped = ArchitectureFingerprint._unwrap_model(dp)
        assert unwrapped is m

    def test_nested_dataparallel(self):
        m = nn.Sequential(nn.Conv2d(3, 16, 3))
        dp = DataParallel(DataParallel(m))
        unwrapped = ArchitectureFingerprint._unwrap_model(dp)
        assert unwrapped is m

    def test_compiled_model(self):
        """torch.compile wraps model in _orig_mod."""
        m = nn.Sequential(nn.Conv2d(3, 16, 3))
        # Mock the compile wrapper by injecting _orig_mod
        compiled = nn.Sequential(nn.Conv2d(3, 16, 3))
        compiled._orig_mod = m
        unwrapped = ArchitectureFingerprint._unwrap_model(compiled)
        assert unwrapped is m

    def test_dataparallel_with_orig_mod(self):
        """DDP + compile: both .module and ._orig_mod."""
        inner = nn.Sequential(nn.Conv2d(3, 16, 3))
        compiled = nn.Sequential(nn.Conv2d(3, 16, 3))
        compiled._orig_mod = inner
        dp = DataParallel(compiled)
        unwrapped = ArchitectureFingerprint._unwrap_model(dp)
        assert unwrapped is inner


# ============================================================================
# PlacementDecision.to_dict
# ============================================================================


class TestPlacementDecisionDict:
    def test_accept_to_dict(self):
        d = PlacementDecision(status="ACCEPT", predicted_delta=0.05)
        dd = d.to_dict()
        assert dd["status"] == "ACCEPT"
        assert dd["predicted_delta"] == pytest.approx(0.05)
        assert dd["target_modules_hint_count"] == 0

    def test_refuse_to_dict(self):
        d = PlacementDecision(
            status="REFUSE",
            refusal_reason="unsafe",
            predicted_delta=-0.1,
            target_modules_hint=["a", "b"],
        )
        dd = d.to_dict()
        assert dd["status"] == "REFUSE"
        assert dd["refusal_reason"] == "unsafe"
        assert dd["target_modules_hint_count"] == 2

    def test_adapt_to_dict(self):
        d = PlacementDecision(
            status="ADAPT",
            recommended_variant="loha",
            recommended_rank=8,
            predicted_delta=0.03,
            safety_overrides={"r": 8},
        )
        dd = d.to_dict()
        assert dd["recommended_variant"] == "loha"
        assert dd["recommended_rank"] == 8
        assert dd["safety_overrides"] == {"r": 8}


# ============================================================================
# Empty model handling
# ============================================================================


class TestEmptyModel:
    """Test that models with no Conv2d / Linear return zero fingerprint."""

    def test_batchnorm_only_model(self):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(16)

        m = _Model()
        fp = ArchitectureFingerprint.compute(m)
        assert fp.phi_attn == 0.0
        assert fp.phi_dw == 0.0
        assert fp.phi_group == 0.0
        assert fp.phi_linear == 0.0

    def test_relu_only_model(self):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = nn.ReLU()

        m = _Model()
        fp = ArchitectureFingerprint.compute(m)
        assert fp.phi_attn == 0.0


# ============================================================================
# PEFTPlanner audit integration
# ============================================================================


class TestPlannerAuditIntegration:
    """Test that plan() emits audit records."""

    def test_audit_file_created_on_accept(self, tmp_path):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)

        from ultralytics.utils.lora.config import LoRAConfig

        planner = PEFTPlanner(audit_dir=tmp_path)
        config = LoRAConfig(peft_type="lora", r=8)
        decision = planner.plan(_Model(), config)
        assert decision.status == "ACCEPT"

        # Audit should be saved
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["decision_status"] == "ACCEPT"
        assert data["variant"] == "lora"

    def test_audit_file_created_on_refuse(self, tmp_path):
        class _RTDETR(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.decoder = nn.Module()
                self.decoder.__class__.__name__ = "RTDETRDecoder"

        from ultralytics.utils.lora.config import LoRAConfig

        planner = PEFTPlanner(audit_dir=tmp_path)
        config = LoRAConfig(peft_type="lora", r=16)
        decision = planner.plan(_RTDETR(), config)
        assert decision.status == "REFUSE"

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["decision_status"] == "REFUSE"
        assert "RT-DETR" in data["refusal_reason"]

    def test_audit_file_created_on_adapt(self, tmp_path):
        # Use a model with phi_attn between 0.3 and 0.7 to trigger
        # DoRA→LoRA downgrade (Guardrail A) but NOT Guardrail B.
        # 1 AAttn + 2 conv = total_modules=2 (conv+linear), attn=1 → phi_attn=0.5
        class _YOLO12(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3)
                self.conv2 = nn.Conv2d(16, 32, 3)
                self.attn = nn.Module()
                self.attn.__class__.__name__ = "AAttn"

        from ultralytics.utils.lora.config import LoRAConfig

        planner = PEFTPlanner(audit_dir=tmp_path)
        config = LoRAConfig(peft_type="dora", r=16)
        decision = planner.plan(_YOLO12(), config)
        assert decision.status == "ADAPT", f"Expected ADAPT (DoRA→LoRA downgrade), got {decision.status}"

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["decision_status"] == "ADAPT"
        assert data["recommended_variant"] == "lora"  # DoRA degraded to LoRA


class TestPlannerCalibrationGates:
    """Regression decisions expose calibration evidence and fail conservatively."""

    @staticmethod
    def _history(n=10):
        variants = ("lora", "dora", "loha", "lokr", "ia3", "hra")
        return [
            (
                ArchitectureFingerprint(phi_attn=(i % 3) * 0.15, phi_text=0.0, phi_dw=(i % 2) * 0.1),
                variants[i % len(variants)],
                0.05 + (i % 4) * 0.002,
            )
            for i in range(n)
        ]

    def test_fit_records_sample_size_and_effective_rank(self):
        planner = PEFTPlanner()
        planner.fit(self._history())

        metadata = planner._calibration_metadata()
        assert metadata["calibration_fitted"] is True
        assert metadata["calibration_n_samples"] == 10
        assert 0 < metadata["calibration_effective_rank"] <= metadata["calibration_n_features"]
        assert metadata["calibration_regularization"] > 0
        assert metadata["calibration_condition_number"] >= 1
        assert metadata["calibration_noise_variance"] >= 0
        assert metadata["calibration_effective_dof"] > 0
        assert metadata["calibration_rank_deficient"] is True
        assert metadata["calibration_posterior_available"] is True
        assert metadata["low_confidence"] is True

    def test_rank_deficient_large_dataset_remains_low_confidence(self):
        planner = PEFTPlanner()
        planner.fit(self._history(30))

        metadata = planner._calibration_metadata()
        assert metadata["calibration_n_samples"] == 30
        assert metadata["calibration_rank_deficient"] is True
        assert metadata["low_confidence"] is True

    def test_rank_deficient_fit_shrinks_unobserved_features_to_prior(self):
        planner = PEFTPlanner()
        planner.fit(self._history())

        assert all(abs(value) < 1e-12 for value in planner._coeffs[5:10])
        assert all(torch.isfinite(torch.tensor(planner._coeffs)))

    def test_identifiable_signal_is_preserved_by_prior_centered_ridge(self):
        history = []
        for i in range(40):
            phi_attn = i / 50.0
            fingerprint = ArchitectureFingerprint(phi_attn=phi_attn, phi_width=0.4 + 0.1 * (i % 3))
            delta = 0.04 - 0.2 * phi_attn + PEFTVariantProfile.from_variant("lora").xi
            history.append((fingerprint, "lora", delta))

        planner = PEFTPlanner()
        planner.fit(history)

        assert planner._coeffs[1] == pytest.approx(-0.2, abs=0.03)

    def test_posterior_uncertainty_grows_off_distribution(self):
        history = [
            (
                ArchitectureFingerprint(
                    phi_attn=0.18 + (i % 5) * 0.01,
                    phi_width=0.45 + (i % 3) * 0.01,
                    phi_depth=0.35 + (i % 2) * 0.01,
                ),
                "lora",
                0.06 - 0.03 * (i % 5) * 0.01,
            )
            for i in range(30)
        ]
        planner = PEFTPlanner()
        planner.fit(history)

        _, near_uncertainty = planner.predict_with_uncertainty(
            ArchitectureFingerprint(phi_attn=0.2, phi_width=0.46, phi_depth=0.35), "lora"
        )
        _, far_uncertainty = planner.predict_with_uncertainty(
            ArchitectureFingerprint(phi_attn=0.9, phi_width=1.0, phi_depth=1.0), "lora"
        )

        assert far_uncertainty > near_uncertainty

    def test_small_fitted_dataset_downgrades_accept_to_adapt(self):
        from ultralytics.utils.lora.config import LoRAConfig

        model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU())
        planner = PEFTPlanner()
        planner.fit(self._history())
        decision = planner.plan(model, LoRAConfig(peft_type="lora", r=8))

        assert decision.status == "ADAPT"
        assert decision.safety_overrides["planner_low_confidence"] is True
        assert decision.metadata["calibration_n_samples"] == 10
        assert decision.metadata["low_confidence"] is True

    def test_uncertainty_lower_bound_can_refuse(self, monkeypatch):
        from ultralytics.utils.lora.config import LoRAConfig

        model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU())
        planner = PEFTPlanner()
        planner.fit(self._history(30))
        monkeypatch.setattr(planner, "predict_with_uncertainty", lambda *args, **kwargs: (0.01, 0.04))
        decision = planner.plan(model, LoRAConfig(peft_type="lora", r=8))

        assert decision.status == "REFUSE"
        assert decision.safety_overrides["uncertainty_guard"] is True
        assert decision.metadata["prediction_lower_95"] < planner.REFUSE_THRESHOLD


class TestPlannerDecisionEvidence:
    """Every planner outcome reports whether it used observations or a fallback rule."""

    @staticmethod
    def _model():
        return nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU())

    @staticmethod
    def _config(variant="lora"):
        from ultralytics.utils.lora.config import LoRAConfig

        return LoRAConfig(peft_type=variant, r=8)

    def test_zero_observation_accept_is_explicit_low_confidence_cold_start(self):
        decision = PEFTPlanner().plan(self._model(), self._config())

        assert decision.status == "ACCEPT"
        assert decision.evidence == {
            "state": "cold_start",
            "source": "default_prior",
            "observation_count": 0,
            "minimum_fit_observations": 5,
            "confidence": "low",
            "confidence_reasons": ["no_observations", "default_coefficients"],
            "uses_learned_evidence": False,
            "decision_basis": "prior_prediction",
            "guardrails": [],
        }
        assert decision.metadata["low_confidence"] is True

    def test_pre_fit_observations_are_counted_without_claiming_learned_evidence(self):
        fingerprint = ArchitectureFingerprint(phi_attn=0.0, phi_dw=0.0)
        collector = LOVODataCollector(
            [LOVODataPoint(fingerprint=fingerprint, variant="lora", delta_mAP=0.05) for _ in range(4)]
        )

        decision = PEFTPlanner(lovo_collector=collector).plan(self._model(), self._config())

        assert decision.evidence["state"] == "cold_start"
        assert decision.evidence["observation_count"] == 4
        assert decision.evidence["confidence_reasons"] == ["insufficient_observations", "default_coefficients"]
        assert decision.evidence["uses_learned_evidence"] is False

    def test_fitted_limited_evidence_is_distinct_from_cold_start(self):
        planner = PEFTPlanner()
        planner.fit(TestPlannerCalibrationGates._history())

        decision = planner.plan(self._model(), self._config())

        assert decision.status == "ADAPT"
        assert decision.evidence["state"] == "limited_evidence"
        assert decision.evidence["source"] == "learned_regression"
        assert decision.evidence["observation_count"] == 10
        assert decision.evidence["confidence"] == "low"
        assert decision.evidence["uses_learned_evidence"] is True
        assert decision.evidence["decision_basis"] == "learned_prediction"

    def test_guardrail_refusal_identifies_rule_and_persists_in_audit(self, tmp_path):
        class _RTDETR(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.decoder = nn.Module()
                self.decoder.__class__.__name__ = "RTDETRDecoder"

        decision = PEFTPlanner(audit_dir=tmp_path).plan(_RTDETR(), self._config())

        assert decision.status == "REFUSE"
        assert decision.evidence["decision_basis"] == "guardrail_fallback"
        assert decision.evidence["guardrails"] == ["rtdetr_lora_family"]
        assert decision.evidence["uses_learned_evidence"] is False
        audit = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert audit["evidence"] == decision.evidence
