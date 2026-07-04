import json
from pathlib import Path

import pytest

from capabilities.profile_loader import load_profile, load_profiles


PROFILE_ROOT = Path("capabilities/profiles")


def test_all_bundled_profiles_load():
    paths = sorted(PROFILE_ROOT.glob("*/*.json"))

    profiles = load_profiles(paths)

    assert len(profiles) == 8
    assert {profile["capability_type"] for profile in profiles} == {
        "hardware",
        "backend",
        "kernels",
    }


def test_malformed_profile_rejected(tmp_path: Path):
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps({"capability_type": "hardware"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        load_profile(path)


def test_unknown_profile_type_rejected(tmp_path: Path):
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps({"capability_type": "accelerator"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown capability type"):
        load_profile(path)


def test_profile_types_are_recognized():
    assert load_profile(PROFILE_ROOT / "hardware/apple_m5.json")["capability_type"] == "hardware"
    assert load_profile(PROFILE_ROOT / "backend/coreml.json")["capability_type"] == "backend"
    assert load_profile(PROFILE_ROOT / "kernels/coreml_builtin.json")["capability_type"] == "kernels"


def test_apple_profile_does_not_claim_measured_ane_utilization():
    profile = load_profile(PROFILE_ROOT / "hardware/apple_m5.json")
    capability = profile["capability"]

    assert capability["vendor"] == "Apple"
    assert capability["attributes"]["ane"] is True
    assert capability["attributes"]["ane_utilization_measured"] is False
    assert "Exact ANE utilization is not directly measured." in capability["notes"]


def test_gtx1650_profile_does_not_claim_fp8_nvfp4_mxfp4():
    profile = load_profile(PROFILE_ROOT / "hardware/nvidia_gtx1650_maxq.json")
    attributes = profile["capability"]["attributes"]

    assert attributes["fp16"] is True
    assert attributes["fp8"] is False
    assert attributes["nvfp4"] is False
    assert attributes["mxfp4"] is False


def test_vllm_profile_does_not_claim_this_repo_implements_vllm():
    profile = load_profile(PROFILE_ROOT / "backend/vllm.json")

    assert "this repo implements vLLM" in profile["does_not_claim"]
    assert "this repo modifies vLLM kernels" in profile["does_not_claim"]


def test_coreml_profile_does_not_claim_custom_kernels():
    backend = load_profile(PROFILE_ROOT / "backend/coreml.json")
    kernels = load_profile(PROFILE_ROOT / "kernels/coreml_builtin.json")
    custom_kernel = _kernel_by_operation(kernels, "CustomKernel")

    assert "custom kernel replacement" in backend["does_not_claim"]
    assert custom_kernel["availability"] == "unsupported"
    assert custom_kernel["support_status"] == "unsupported_in_this_project"


def test_vllm_runtime_marks_future_features_unmeasured():
    profile = load_profile(PROFILE_ROOT / "kernels/vllm_runtime.json")
    prefix_cache = _kernel_by_operation(profile, "PrefixCache")
    speculative = _kernel_by_operation(profile, "SpeculativeDecoding")
    flash_attn = _kernel_by_operation(profile, "FlashAttention2")

    assert prefix_cache["support_status"] == "backend_supported_or_future"
    assert prefix_cache["measured"] is False
    assert speculative["support_status"] == "backend_supported_or_future"
    assert speculative["measured"] is False
    assert flash_attn["availability"] == "unsupported"


def _kernel_by_operation(profile: dict, operation: str) -> dict:
    for kernel in profile["kernels"]:
        if kernel["operation"] == operation:
            return kernel
    raise AssertionError(f"missing kernel operation: {operation}")
