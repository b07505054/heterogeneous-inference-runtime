from pathlib import Path

import pytest

from deployment.execution_plan_v2.capability_view import (
    CapabilityValidationError,
    CapabilityValidationView,
)
from deployment.execution_plan_v2.schema import CapabilityBundleRef


def test_capability_view_validates_refs_against_ml_platform_capabilities():
    view = CapabilityValidationView()
    bundle = CapabilityBundleRef(
        hardware_profile_ref="hardware/nvidia_gtx1650_maxq.json",
        backend_profile_refs=("backend/vllm.json",),
        kernel_profile_refs=("kernels/triton.json",),
        workload_ref="workloads/qwen_short_to_medium_32.json",
    )

    assert view.validate_bundle(bundle) == []
    assert view.load_ref("backend/vllm.json")["backend_id"] == "vllm"


def test_capability_view_rejects_missing_ref(tmp_path: Path):
    view = CapabilityValidationView(profile_root=tmp_path)

    with pytest.raises(CapabilityValidationError, match="missing_capability_ref"):
        view.validate_refs(("backend/vllm.json",))
