import json
from pathlib import Path

import pytest

from capabilities.profile_loader import load_profile, load_profiles


PROFILE_ROOT = Path("../ml-platform-capabilities/profiles")


def test_all_bundled_profiles_load():
    paths = sorted(PROFILE_ROOT.glob("*/*.json"))

    profiles = load_profiles(paths)

    assert len(profiles) == 17
    assert {profile["capability_type"] for profile in profiles} == {
        "hardware",
        "backend",
        "kernels",
        "platform",
        "model",
        "workload",
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


def test_vllm_backend_references_kernel_provider_profiles():
    profile = load_profile(PROFILE_ROOT / "backend/vllm.json")

    assert profile["supported_kernel_libraries"] == [
        "flashattention2",
        "cutlass",
        "cublas",
        "triton",
        "xformers",
    ]


def test_flashattention2_provider_marks_gtx1650_attention_unsupported():
    profile = load_profile(PROFILE_ROOT / "kernels/flashattention2.json")
    attention = _kernel_by_operation(profile, "Attention")

    assert attention["availability"] == "unsupported"
    assert attention["measured"] is False


def _kernel_by_operation(profile: dict, operation: str) -> dict:
    for kernel in profile["kernels"]:
        if kernel["operation"] == operation:
            return kernel
    raise AssertionError(f"missing kernel operation: {operation}")


def test_gtx1650_profile_declares_phase1_single_gpu_capabilities():
    profile = load_profile(PROFILE_ROOT / "hardware/nvidia_gtx1650_maxq.json")
    capability = profile["capability"]
    attributes = capability["attributes"]

    assert capability["memory"]["vram_gb"] == 4
    assert attributes["cuda_runtime"] == "13.2"
    assert attributes["compute_capability"] == "7.5"
    assert attributes["tensor_core_support"] == "unknown"
    assert attributes["bf16"] is False
    assert attributes["int8"] is False
    assert attributes["int4"] is False
    assert attributes["multi_gpu"] is False


def test_vllm_backend_profile_declares_phase1_policy_capabilities():
    profile = load_profile(PROFILE_ROOT / "backend/vllm.json")
    supports = profile["supports"]

    assert supports["quantization"] == ["none"]
    assert supports["speculative_decoding"] == {"supported": False, "measured": False}
    assert supports["prefix_cache"] == {"supported": True, "measured": False}
    assert supports["chunked_prefill"] == {"supported": True, "measured": False}
    assert supports["paged_attention"] == {"supported": True, "measured": False}
    assert supports["tensor_parallel"] == {"supported": False, "max_size": 1, "measured": False}
    assert supports["pipeline_parallel"] == {"supported": False, "max_size": 1, "measured": False}


def test_kernel_provider_profiles_declare_phase1_kernel_categories():
    triton = load_profile(PROFILE_ROOT / "kernels/triton.json")
    cutlass = load_profile(PROFILE_ROOT / "kernels/cutlass.json")
    cublas = load_profile(PROFILE_ROOT / "kernels/cublas.json")

    assert _kernel_by_operation(triton, "Dequant")["availability"] == "unsupported"
    assert _kernel_by_operation(triton, "Requant")["availability"] == "unsupported"
    assert _kernel_by_operation(triton, "RMSNorm")["availability"] == "opaque"
    assert _kernel_by_operation(triton, "Attention")["availability"] == "opaque"
    assert _kernel_by_operation(cutlass, "FusedKernels")["availability"] == "opaque"
    assert _kernel_by_operation(cublas, "MatMul")["availability"] == "builtin"
