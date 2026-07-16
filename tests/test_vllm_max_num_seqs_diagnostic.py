import importlib.util
from pathlib import Path


path = Path(__file__).parents[1] / "scripts/run_vllm_max_num_seqs_diagnostic.py"
spec = importlib.util.spec_from_file_location("diagnostic", path)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_prometheus_parser_requires_exact_metric_name():
    text = """
vllm:num_requests_waiting{model_name="m"} 2
vllm:num_requests_waiting_for_remote_kv{model_name="m"} 7
vllm:num_requests_running{model_name="m"} 2
"""
    parsed = diagnostic._parse_prometheus(
        text, ("num_requests_waiting", "num_requests_running")
    )
    assert parsed == {"num_requests_waiting": 2.0, "num_requests_running": 2.0}
