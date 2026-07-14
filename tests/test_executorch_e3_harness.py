import json
from pathlib import Path

import pytest

from evaluation.executorch_e3 import e3_harness


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPILER_TOOL = REPO_ROOT.parent / "ml-graph-compiler-runtime" / "tools" / "e3_xnnpack_contract.py"
OLD_KERNEL_ID = "portable_fused_matmul_bias_relu_bm32_bn128_bk32"


def _write_artifacts(tmp_path):
    pte = tmp_path / "model.pte"
    runner = tmp_path / "runner"
    pte.write_bytes(b"pte")
    runner.write_bytes(b"runner")
    runner.chmod(0o755)
    return pte, runner


def _generate_contract(tmp_path, threads=4):
    pte, runner = _write_artifacts(tmp_path)
    out = tmp_path / "contract.json"
    args = type("Args", (), {
        "compiler_tool": str(COMPILER_TOOL),
        "shape": "64x64x64",
        "pte": str(pte),
        "runner": str(runner),
        "selected_threads": threads,
        "contract_out": str(out),
    })()
    e3_harness.invoke_live_compiler(args)
    return out, pte, runner


def _rewrite_contract(path, mutate):
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _rewrite_contract_with_hash(path, mutate):
    data = json.loads(path.read_text())
    mutate(data)
    data["contract_sha256"] = e3_harness.canonical_contract_hash(data)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


@pytest.mark.skipif(not COMPILER_TOOL.exists(), reason="requires sibling compiler checkout")
def test_e3_harness_invokes_live_compiler_and_validates_contract(tmp_path):
    contract_path, _, _ = _generate_contract(tmp_path, threads=4)
    contract = e3_harness.load_and_validate_contract(contract_path)
    assert contract["runner_contract"] == "executorch_xnnpack_runner_contract"
    assert contract["selected_candidate"]["requested_thread_count"] == 4
    assert contract["selected_candidate"]["library"] == "xnnpack"


def test_e3_harness_source_contains_no_portable_decision_constants():
    text = (REPO_ROOT / "evaluation" / "executorch_e3" / "e3_harness.py").read_text()
    assert "THRESHOLD" not in text
    assert "262144" not in text
    assert OLD_KERNEL_ID not in text
    assert "P-SERIAL" not in text
    assert "P-4T" not in text


@pytest.mark.skipif(not COMPILER_TOOL.exists(), reason="requires sibling compiler checkout")
def test_contract_hash_mutation_rejects(tmp_path):
    contract_path, _, _ = _generate_contract(tmp_path)
    _rewrite_contract(contract_path, lambda d: d.__setitem__("library", "not_xnnpack"))
    with pytest.raises(e3_harness.ContractError, match="contract_hash_mismatch"):
        e3_harness.load_and_validate_contract(contract_path)


@pytest.mark.skipif(not COMPILER_TOOL.exists(), reason="requires sibling compiler checkout")
def test_runner_hash_mutation_rejects(tmp_path):
    contract_path, _, _ = _generate_contract(tmp_path)
    _rewrite_contract_with_hash(contract_path, lambda d: d["runner"].__setitem__("sha256", "0" * 64))
    with pytest.raises(e3_harness.ContractError, match="runner_hash_mismatch"):
        e3_harness.load_and_validate_contract(contract_path)


@pytest.mark.skipif(not COMPILER_TOOL.exists(), reason="requires sibling compiler checkout")
def test_pte_hash_mutation_rejects(tmp_path):
    contract_path, _, _ = _generate_contract(tmp_path)
    _rewrite_contract_with_hash(contract_path, lambda d: d["pte"].__setitem__("sha256", "0" * 64))
    with pytest.raises(e3_harness.ContractError, match="pte_hash_mismatch"):
        e3_harness.load_and_validate_contract(contract_path)


@pytest.mark.skipif(not COMPILER_TOOL.exists(), reason="requires sibling compiler checkout")
def test_thread_mode_mutation_rejects(tmp_path):
    contract_path, _, _ = _generate_contract(tmp_path)
    def mutate(d):
        d["requested_thread_mode"]["threads"] = 1
    _rewrite_contract_with_hash(contract_path, mutate)
    with pytest.raises(e3_harness.ContractError, match="thread_mode_mismatch"):
        e3_harness.load_and_validate_contract(contract_path)


@pytest.mark.skipif(not COMPILER_TOOL.exists(), reason="requires sibling compiler checkout")
def test_source_provenance_mutation_rejects(tmp_path):
    contract_path, _, _ = _generate_contract(tmp_path)
    def mutate(d):
        d["executorch"]["commit"] = "bad"
    _rewrite_contract_with_hash(contract_path, mutate)
    with pytest.raises(e3_harness.ContractError, match="executorch_commit_mismatch"):
        e3_harness.load_and_validate_contract(contract_path)


@pytest.mark.skipif(not COMPILER_TOOL.exists(), reason="requires sibling compiler checkout")
def test_runner_self_report_mismatch_rejects(tmp_path):
    contract_path, _, _ = _generate_contract(tmp_path)
    contract = e3_harness.load_and_validate_contract(contract_path)
    report = {
        "runner_sha256": contract["runner"]["sha256"],
        "pte_sha256": contract["pte"]["sha256"],
        "executorch_commit": e3_harness.EXPECTED_EXECUTORCH_COMMIT,
        "xnnpack_commit": e3_harness.EXPECTED_XNNPACK_COMMIT,
        "requested_threads": contract["requested_thread_mode"]["threads"],
        "backend": "portable",
    }
    with pytest.raises(e3_harness.ContractError, match="runner_backend_mismatch"):
        e3_harness.validate_runner_report(contract, report)
