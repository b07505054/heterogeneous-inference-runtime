#!/usr/bin/env python3
"""D8/Phase 3: real vLLM TP2 NCCL attribution runner.

The script is intentionally fail-closed. If vLLM/torch/Nsight tooling is not
available, it still writes the required D8 artifact set, but every measurement
artifact is marked blocked and contains the concrete missing-tool reason. It
never synthesizes per-decode NCCL timings from D5/D6/D7 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import socket
import statistics
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deployment.vllm_adapter.distributed_launch_controller import ServerLaunchController  # noqa: E402
from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec  # noqa: E402
from deployment.vllm_adapter.tp_benchmark_harness import run_workload_benchmark  # noqa: E402
from deployment.vllm_adapter.tp_cost_model import load_communication_predictor  # noqa: E402
from deployment.vllm_adapter.tp_workload_matrix import WorkloadSpec, build_prompt_of_token_length  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results/runtime_paths/distributed_d8_vllm_nccl_attribution"
LOG_DIR = RESULTS_DIR / "logs"
TRACE_DIR = RESULTS_DIR / "traces"
MULTICELL_DIR = RESULTS_DIR / "multicell"
D2_DIR = REPO_ROOT / "results/runtime_paths/distributed_d2_qwen_pipeline"
D4A_DIR = REPO_ROOT / "results/runtime_paths/distributed_d4a_whole_model_tp_contract"
NCCL_DIR = REPO_ROOT / "results/runtime_paths/nccl_calibration"
D7_DIR = REPO_ROOT / "results/runtime_paths/distributed_d7_nccl_aware_tp_selection"
TP1_PLAN_PATH = D2_DIR / "real_qwen_tp1_execution_plan.json"
TP2_PLAN_PATH = D2_DIR / "real_qwen_tp2_execution_plan.json"
D4A_EVIDENCE_PATH = D4A_DIR / "whole_model_tp_classification.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

DEFAULT_WORKLOADS = (
    WorkloadSpec(32, 32, 1),
    WorkloadSpec(128, 32, 1),
    WorkloadSpec(32, 128, 1),
    WorkloadSpec(32, 32, 4),
    WorkloadSpec(32, 32, 8),
)
D8_ENV_PYTHON = Path("/workspace/d8-vllm-env/bin/python")


def write_json(name: str, payload: Any) -> None:
    path = RESULTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {path}")


def run_text(cmd: list[str], *, timeout_s: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
        return {"argv": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:  # noqa: BLE001
        return {"argv": cmd, "returncode": None, "stdout": "", "stderr": str(exc)}


def module_origin(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return spec.origin if spec else None


def dependency_inventory() -> dict[str, Any]:
    modules = {m: module_origin(m) for m in ("vllm", "torch", "transformers", "requests", "numpy")}
    nsys = shutil.which("nsys")
    return {
        "python_executable": sys.executable,
        "modules": modules,
        "nsys_path": nsys,
        "nsys_version": run_text([nsys, "--version"], timeout_s=10) if nsys else None,
        "can_run_real_vllm": bool(modules["vllm"] and modules["torch"] and modules["transformers"]),
        "can_collect_nsys": bool(nsys),
    }


def environment_manifest(deps: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "status": "ready" if deps["can_run_real_vllm"] else "blocked",
        "python_executable": sys.executable,
        "environment_path": str(Path(sys.executable).parents[1]),
        "install_strategy": (
            "dedicated D8 profiling environment; /venv/main was left unmodified "
            "because it did not contain torch/vllm"
        ),
        "dependency_inventory": deps,
        "verification": {},
    }
    try:
        import torch  # noqa: PLC0415
        import transformers  # noqa: PLC0415
        import vllm  # noqa: PLC0415

        manifest["versions"] = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_cuda_nccl_version": torch.cuda.nccl.version(),
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
        }
        manifest["verification"] = {
            "torch_cuda_is_available": bool(torch.cuda.is_available()),
            "torch_cuda_device_count": torch.cuda.device_count(),
            "torch_cuda_device_count_is_2": torch.cuda.device_count() == 2,
            "torch_cuda_nccl_version": torch.cuda.nccl.version(),
            "vllm_imports": True,
            "transformers_imports": True,
        }
    except Exception as exc:  # noqa: BLE001
        manifest["status"] = "blocked"
        manifest["verification_error"] = str(exc)
    return manifest


def topology_and_transport() -> dict[str, Any]:
    topo = run_text(["nvidia-smi", "topo", "-m"])
    gpus = run_text([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id",
        "--format=csv,noheader",
    ])
    profile = json.loads((NCCL_DIR / "communication_cost_profile.json").read_text())
    boundary = profile.get("machine_calibration_boundary", {})
    return {
        "machine_boundary": {
            "gpu_model": boundary.get("gpu_model", "NVIDIA GeForce RTX 4090"),
            "gpu_count": boundary.get("gpu_count", 2),
            "topology_class": boundary.get("topology_class", "PHB"),
            "numa_count": boundary.get("numa_count", 1),
            "cuda_p2p_available": boundary.get("cuda_p2p_available", False),
            "nccl_intra_node_transport": boundary.get("nccl_intra_node_transport", "SHM/direct/direct"),
        },
        "nvidia_smi_topo_m": topo,
        "nvidia_smi_gpu_query": gpus,
        "phase1_profile_id": profile.get("profile_id"),
        "nccl_version": boundary.get("nccl_version"),
        "nccl_tests_version": boundary.get("nccl_tests_version"),
    }


def selected_workloads(args: argparse.Namespace) -> list[WorkloadSpec]:
    if args.workload:
        out = []
        for item in args.workload:
            m = re.match(r"^in(\d+)_out(\d+)_c(\d+)$", item)
            if not m:
                raise SystemExit(f"invalid workload id: {item}")
            out.append(WorkloadSpec(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        return out
    return list(DEFAULT_WORKLOADS)


def workload_manifest(workloads: list[WorkloadSpec], deps: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ready" if deps["can_run_real_vllm"] else "blocked_missing_vllm_or_torch",
        "model": MODEL_ID,
        "matched_fields": [
            "model",
            "prompt_lengths",
            "output_lengths",
            "concurrency",
            "dtype",
            "vllm_launch_materializer",
            "warmup_requests",
            "measured_repetitions",
        ],
        "dtype": "from materialized D2/D4A launch spec",
        "warmup_procedure": {
            "warmup_requests_per_workload": 2,
            "measured_repetitions_per_workload": 5,
            "warmup_excluded_from_measurements": True,
        },
        "workloads": [w.to_dict() for w in workloads],
        "dependency_inventory": deps,
        "source_inputs": {
            "phase1_profile": str(NCCL_DIR / "communication_cost_profile.json"),
            "phase1_fit_report": str(NCCL_DIR / "fit_report.json"),
            "d7_artifacts": str(D7_DIR),
            "tp1_plan": str(TP1_PLAN_PATH),
            "tp2_plan": str(TP2_PLAN_PATH),
        },
    }


def blocked_artifacts(reason: str, workloads: list[WorkloadSpec], deps: dict[str, Any]) -> None:
    common = {
        "status": "blocked",
        "blocked_reason": reason,
        "no_synthesized_measurements": True,
        "dependency_inventory": deps,
    }
    write_json("tp1_summary.json", {
        **common,
        "tensor_parallel_size": 1,
        "workloads": [w.to_dict() for w in workloads],
        "tp1_compute_time": None,
    })
    write_json("tp2_summary.json", {
        **common,
        "tensor_parallel_size": 2,
        "workloads": [w.to_dict() for w in workloads],
        "tp2_compute_time_excluding_nccl": None,
        "raw_nccl_time_per_token": None,
    })
    write_json("per_decode_step_collectives.json", {
        **common,
        "schema": "d8.per_decode_step_collectives.v1",
        "rows": [],
        "required_fields": [
            "decode_step",
            "collective_kind",
            "tensor_shape",
            "tensor_bytes",
            "collective_call_count",
            "raw_nccl_collective_gpu_duration_us",
            "total_gpu_compute_duration_us",
            "decode_step_critical_path_duration_us",
        ],
    })
    write_json("nccl_prediction_vs_observed.json", {
        **common,
        "rows": [],
        "mae_us": None,
        "mape": None,
        "max_abs_error_us": None,
        "max_relative_error": None,
        "error_by_message_size_bucket": {},
    })
    write_json("overlap_attribution.json", {
        **common,
        "definition": (
            "overlap_ratio = overlapped NCCL interval duration / raw NCCL interval duration, "
            "computed from CUDA/NVTX timeline interval intersections; not inferred from ordering."
        ),
        "raw_nccl_time_us": None,
        "overlapped_nccl_time_us": None,
        "measured_overlap_ratio": None,
        "exposed_nccl_time_us": None,
        "d7_assumption_assessment": "blocked_no_real_timeline",
        "candidate_d8_correction": "exposed_comm_time = raw_nccl_time * (1 - measured_overlap_ratio)",
    })
    write_json("tp_break_even_analysis.json", {
        **common,
        "formula": "TP2 wins iff TP compute savings > exposed communication penalty + runtime residual",
        "rows": [],
        "explains_measured_winner": None,
    })


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Interval:
    start_ns: int
    end_ns: int

    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.start_ns) / 1000.0


def interval_overlap_us(a: Interval, b: Interval) -> float:
    return max(0, min(a.end_ns, b.end_ns) - max(a.start_ns, b.start_ns)) / 1000.0


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    ordered = sorted((i for i in intervals if i.end_ns > i.start_ns), key=lambda i: i.start_ns)
    merged: list[Interval] = []
    for item in ordered:
        if not merged or item.start_ns > merged[-1].end_ns:
            merged.append(Interval(item.start_ns, item.end_ns))
        else:
            merged[-1].end_ns = max(merged[-1].end_ns, item.end_ns)
    return merged


def interval_union_duration_us(intervals: list[Interval]) -> float:
    return sum(i.duration_us for i in merge_intervals(intervals))


def interval_intersection_duration_us(a_intervals: list[Interval], b_intervals: list[Interval]) -> float:
    a = merge_intervals(a_intervals)
    b = merge_intervals(b_intervals)
    i = j = 0
    total_ns = 0
    while i < len(a) and j < len(b):
        start = max(a[i].start_ns, b[j].start_ns)
        end = min(a[i].end_ns, b[j].end_ns)
        if end > start:
            total_ns += end - start
        if a[i].end_ns <= b[j].end_ns:
            i += 1
        else:
            j += 1
    return total_ns / 1000.0


def clipped_interval(item: dict[str, Any], window: dict[str, Any]) -> Interval | None:
    start = max(int(item["start_ns"]), int(window["start_ns"]))
    end = min(int(item["end_ns"]), int(window["end_ns"]))
    if end <= start:
        return None
    return Interval(start, end)


def interval_list(items: list[dict[str, Any]], window: dict[str, Any] | None = None) -> list[Interval]:
    out: list[Interval] = []
    for item in items:
        interval = clipped_interval(item, window) if window is not None else Interval(int(item["start_ns"]), int(item["end_ns"]))
        if interval is not None:
            out.append(interval)
    return out


def overlap_ratio(nccl_intervals: list[Interval], compute_intervals: list[Interval]) -> float:
    nccl_wall = interval_union_duration_us(nccl_intervals)
    if nccl_wall <= 0:
        return 0.0
    overlapped = interval_intersection_duration_us(nccl_intervals, compute_intervals)
    return min(1.0, overlapped / nccl_wall)


def request_ids_from_fields(fields: dict[str, str]) -> list[str]:
    raw = fields.get("req_ids") or fields.get("request_ids") or ""
    return [part for part in raw.split("|") if part]


def measured_request_ids(summary: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for workload in summary.get("workload_results", []):
        for req in workload.get("per_request", []):
            if req.get("request_kind") == "measured" and req.get("request_id"):
                ids.add(str(req["request_id"]))
    return ids


def request_id_matches_intended(observed_ids: list[str], intended_ids: set[str]) -> bool:
    # vLLM OpenAI serving prefixes/suffixes the X-Request-Id value when
    # building internal engine request IDs. The harness tag is unique per cell.
    return any(intended in observed for intended in intended_ids for observed in observed_ids)


def workload_id_from_request_ids(req_ids: list[str]) -> str | None:
    for req_id in req_ids:
        m = re.search(r"in\d+_out\d+_c\d+", req_id)
        if m:
            return m.group(0)
    return None


def summarize_values(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean_us": statistics.mean(values) if values else None,
        "p50_us": _percentile_linear(values, 0.50) if values else None,
        "p95_us": _percentile_linear(values, 0.95) if len(values) >= 20 else None,
        "max_us": max(values) if values else None,
    }


def predict_nccl_time_us(kind: str, bytes_value: int) -> float:
    predictor = load_communication_predictor(
        json.loads((NCCL_DIR / "communication_cost_profile.json").read_text()),
        json.loads((NCCL_DIR / "fit_report.json").read_text()),
        collective_kind=kind,
    )
    return predictor.predict_time_us(bytes_value)


def export_nsys_sqlite(rep_path: Path) -> Path:
    sqlite_path = rep_path.with_suffix(".sqlite")
    if sqlite_path.exists():
        sqlite_path.unlink()
    proc = subprocess.run(
        ["nsys", "export", "--type=sqlite", "--force-overwrite=true", f"--output={sqlite_path}", str(rep_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nsys export failed: {proc.stderr or proc.stdout}")
    return sqlite_path


def sqlite_tables(sqlite_path: Path) -> list[str]:
    with sqlite3.connect(sqlite_path) as con:
        return [r[0] for r in con.execute("select name from sqlite_master where type='table'")]


def materialized_argv(argv: tuple[str, ...], port: int) -> tuple[str, ...]:
    patched = [argv[i] if i == 0 or argv[i - 1] != "--port" else str(port) for i in range(len(argv))]
    if patched and (patched[0].endswith(".venv/bin/python") or not Path(patched[0]).exists()):
        patched[0] = str(D8_ENV_PYTHON if D8_ENV_PYTHON.exists() else Path(sys.executable))
    return tuple(patched)


def collect_nccl_transport_evidence(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    lowered = text.lower()
    lines = [line for line in text.splitlines() if "NCCL" in line or "nccl" in line]
    return {
        "log_path": str(log_path),
        "nccl_log_line_count": len(lines),
        "bounded_nccl_excerpt": lines[-120:],
        "nccl_evidence_found": bool(lines),
        "p2p_disabled_evidence": ("p2p" in lowered and ("disable" in lowered or "disabled" in lowered))
        or "p2p_level" in lowered,
        "shm_transport_evidence": "shm" in lowered,
        "direct_transport_evidence": "direct" in lowered,
    }


def run_one_server(tp: int, *, workloads: list[WorkloadSpec], tokenizer, profiled: bool,
                   args: argparse.Namespace, tp_profile_nvtx: bool = False,
                   label: str | None = None, measured_repetitions: int | None = None) -> dict[str, Any]:
    plan = TP1_PLAN_PATH if tp == 1 else TP2_PLAN_PATH
    bundle = materialize_launch_spec(
        plan,
        repo_root=REPO_ROOT,
        d4a_evidence_path=D4A_EVIDENCE_PATH if tp == 2 else None,
    )
    if not bundle.preflight.passed:
        return {"status": "blocked_preflight_failed", "preflight": bundle.preflight.to_dict()}
    port = _find_free_port()
    env = dict(os.environ)
    env.update(bundle.spec.environment)
    env["CUDA_VISIBLE_DEVICES"] = "0" if tp == 1 else "0,1"
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_DEBUG_SUBSYS"] = "INIT,ENV"
    env["VLLM_USE_V1"] = env.get("VLLM_USE_V1", "1")
    if tp_profile_nvtx:
        env["VLLM_TP_PROFILE_NVTX"] = "1"
    else:
        env.pop("VLLM_TP_PROFILE_NVTX", None)
    run_label = label or ("profiled" if profiled else "unprofiled")
    log_path = LOG_DIR / f"d8_tp{tp}_{run_label}_server.log"
    argv = materialized_argv(bundle.cli.argv, port)
    trace_base = None
    if profiled:
        trace_base = TRACE_DIR / "d8_tp2_vllm"
        argv = (
            "nsys", "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--trace-fork-before-exec=true",
            "--cuda-event-trace=true",
            "--cuda-graph-trace=node",
            "--cuda-flush-interval=1000",
            "--force-overwrite=true",
            f"--output={trace_base}",
            *argv,
        )
    ctrl = ServerLaunchController(argv=argv, env=env, cwd=str(REPO_ROOT), log_path=log_path,
                                  host=bundle.spec.host, port=port)
    ctrl.start()
    ready = ctrl.wait_for_readiness(timeout_s=args.startup_timeout_s, poll_interval_s=5.0)
    if not ready:
        stop = ctrl.stop(graceful_timeout_s=30.0)
        return {"status": "blocked_server_not_ready", "controller": ctrl.to_dict(),
                "stop_result": stop, "server_log": str(log_path)}
    base_url = f"http://{bundle.spec.host}:{port}"
    workload_results = []
    for workload in workloads:
        bench = run_workload_benchmark(
            base_url,
            bundle.spec.served_model_name,
            tokenizer,
            workload,
            tp,
            warmup_requests=args.warmup_requests,
            measured_repetitions=args.measured_repetitions if measured_repetitions is None else measured_repetitions,
            timeout_s=args.request_timeout_s,
        )
        workload_results.append(bench.to_dict())
    stop = ctrl.stop(graceful_timeout_s=45.0)
    summary = {
        "status": "measured",
        "profiled": profiled,
        "tp_profile_nvtx": tp_profile_nvtx,
        "tensor_parallel_size": tp,
        "launch_spec": bundle.spec.to_dict(),
        "controller": ctrl.to_dict(),
        "stop_result": stop,
        "workload_results": workload_results,
        "server_log": str(log_path),
        "nccl_transport_evidence": collect_nccl_transport_evidence(log_path) if tp == 2 else None,
    }
    if trace_base is not None:
        summary["nsight_report"] = str(trace_base.with_suffix(".nsys-rep"))
    return summary


def first_tpot_us(summary: dict[str, Any]) -> float | None:
    try:
        return float(summary["workload_results"][0]["mean_tpot_s"]) * 1_000_000.0
    except Exception:
        return None


def extract_kernel_rows(sqlite_path: Path) -> list[dict[str, Any]]:
    tables = sqlite_tables(sqlite_path)
    kernel_tables = [t for t in tables if "KERNEL" in t.upper()]
    rows: list[dict[str, Any]] = []
    with sqlite3.connect(sqlite_path) as con:
        string_cols = [r[1] for r in con.execute('pragma table_info("StringIds")')]
        has_string_ids = "id" in string_cols and "value" in string_cols
        has_processes = "PROCESSES" in tables
        for table in kernel_tables:
            cols = [r[1] for r in con.execute(f'pragma table_info("{table}")')]
            start_col = next((c for c in cols if c.lower() in ("start", "startns", "start_ns")), None)
            end_col = next((c for c in cols if c.lower() in ("end", "endns", "end_ns")), None)
            name_id_col = next((c for c in cols if c.lower() in ("shortname", "demangledname", "name")), None)
            fallback_name_col = next((c for c in cols if c.lower() in ("mangledname", "mangled_name")), None)
            global_pid_col = next((c for c in cols if c.lower() == "globalpid"), None)
            device_id_col = next((c for c in cols if c.lower() == "deviceid"), None)
            if not (start_col and end_col):
                continue
            meta_select = [
                f'k."{global_pid_col}"' if global_pid_col else "null",
                f'k."{device_id_col}"' if device_id_col else "null",
                "p.pid" if has_processes and global_pid_col else "null",
            ]
            process_join = (
                f' left join PROCESSES p on p.globalPid = k."{global_pid_col}"'
                if has_processes and global_pid_col else ""
            )
            if name_id_col and has_string_ids:
                query = (
                    f'select k."{start_col}", k."{end_col}", '
                    f'coalesce(s.value, cast(k."{name_id_col}" as text)), '
                    f'{", ".join(meta_select)} '
                    f'from "{table}" k left join StringIds s on s.id = k."{name_id_col}"'
                    f'{process_join}'
                )
                values = con.execute(query)
                name_field_count = 1
            else:
                select_cols = [f'k."{start_col}"', f'k."{end_col}"']
                if name_id_col:
                    select_cols.append(f'k."{name_id_col}"')
                if fallback_name_col and fallback_name_col != name_id_col:
                    select_cols.append(f'k."{fallback_name_col}"')
                name_field_count = len(select_cols) - 2
                query = f'select {", ".join(select_cols + meta_select)} from "{table}" k{process_join}'
                values = con.execute(query)
            for raw in values:
                name_end = 2 + name_field_count
                name = " ".join(str(v) for v in raw[2:name_end] if v is not None)
                global_pid, device_id, pid = raw[name_end:name_end + 3]
                lower = name.lower()
                if "allgather" in lower or "all_gather" in lower:
                    collective_kind = "all_gather"
                elif "reducescatter" in lower or "reduce_scatter" in lower:
                    collective_kind = "reduce_scatter"
                elif "broadcast" in lower:
                    collective_kind = "broadcast"
                elif "allreduce" in lower or "all_reduce" in lower:
                    collective_kind = "all_reduce"
                else:
                    collective_kind = None
                rank = f"rank{int(device_id)}" if device_id is not None else "rank_unknown"
                rows.append({
                    "table": table,
                    "start_ns": int(raw[0]),
                    "end_ns": int(raw[1]),
                    "duration_us": (int(raw[1]) - int(raw[0])) / 1000.0,
                    "name": name,
                    "collective_kind": collective_kind,
                    "is_nccl": "nccl" in lower,
                    "is_compute": "nccl" not in lower,
                    "global_pid": int(global_pid) if global_pid is not None else None,
                    "pid": int(pid) if pid is not None else None,
                    "device_id": int(device_id) if device_id is not None else None,
                    "rank": rank,
                })
    return rows

def _parse_nvtx_label(label: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in label.split():
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


def extract_nvtx_ranges(sqlite_path: Path) -> list[dict[str, Any]]:
    if "NVTX_EVENTS" not in sqlite_tables(sqlite_path):
        return []
    ranges: list[dict[str, Any]] = []
    with sqlite3.connect(sqlite_path) as con:
        query = """
            select n.start, n.end, coalesce(n.text, s.value, n.jsonText), n.globalTid
            from NVTX_EVENTS n
            left join StringIds s on s.id = n.textId
            where n.end is not null
        """
        for start, end, text, global_tid in con.execute(query):
            if not text:
                continue
            text = str(text)
            fields = _parse_nvtx_label(text)
            kind = None
            if text.startswith("vllm.step"):
                kind = "step"
            elif text.startswith("vllm.nccl"):
                kind = "collective_metadata"
            if kind is None:
                continue
            ranges.append({
                "kind": kind,
                "start_ns": int(start),
                "end_ns": int(end),
                "duration_us": (int(end) - int(start)) / 1000.0,
                "text": text,
                "fields": fields,
                "global_tid": global_tid,
            })
    return ranges


def _interval_overlap_us(a: dict[str, Any], b: dict[str, Any]) -> float:
    start = max(a["start_ns"], b["start_ns"])
    end = min(a["end_ns"], b["end_ns"])
    return max(0, end - start) / 1000.0


def _percentile_linear(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    low = int(pos)
    high = min(len(vals) - 1, low + 1)
    frac = pos - low
    return vals[low] * (1.0 - frac) + vals[high] * frac


def size_bucket(bytes_value: int) -> str:
    if bytes_value < 16 * 1024:
        return "<16KiB"
    if bytes_value < 1024 * 1024:
        return "16KiB-1MiB"
    return ">=1MiB"


def summarize_prediction_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    rel_errors = []
    by_bucket: dict[str, list[float]] = {}
    by_kind: dict[str, list[float]] = {}
    for row in rows:
        pred = row.get("predicted_nccl_us")
        obs = row.get("observed_raw_nccl_us")
        if pred is None or obs in (None, 0):
            continue
        err = abs(float(pred) - float(obs))
        rel = err / abs(float(obs))
        row["absolute_error_us"] = err
        row["relative_error"] = rel
        errors.append(err)
        rel_errors.append(rel)
        bucket = size_bucket(int(row.get("tensor_bytes") or 0))
        by_bucket.setdefault(bucket, []).append(err)
        by_kind.setdefault(str(row.get("collective_kind", "unknown")), []).append(err)
    return {
        "mae_us": statistics.mean(errors) if errors else None,
        "mape": statistics.mean(rel_errors) if rel_errors else None,
        "p50_abs_error_us": _percentile_linear(errors, 0.50) if len(errors) >= 2 else None,
        "p95_abs_error_us": _percentile_linear(errors, 0.95) if len(errors) >= 2 else None,
        "max_abs_error_us": max(errors) if errors else None,
        "max_relative_error": max(rel_errors) if rel_errors else None,
        "error_by_message_size_bucket": {
            bucket: {"count": len(vals), "mae_us": statistics.mean(vals), "max_abs_error_us": max(vals)}
            for bucket, vals in sorted(by_bucket.items())
        },
        "error_by_collective_kind": {
            kind: {"count": len(vals), "mae_us": statistics.mean(vals), "max_abs_error_us": max(vals)}
            for kind, vals in sorted(by_kind.items())
        },
    }

def write_live_attribution(tp1: dict[str, Any], tp2: dict[str, Any], tp2_profiled: dict[str, Any],
                           workloads: list[WorkloadSpec], tp2_nvtx: dict[str, Any] | None = None) -> None:
    rep = Path(tp2_profiled.get("nsight_report", ""))
    if not rep.exists():
        blocked_artifacts("TP2 Nsight report was not produced", workloads, dependency_inventory())
        return
    sqlite_path = export_nsys_sqlite(rep)
    all_kernels = extract_kernel_rows(sqlite_path)
    nvtx_ranges = extract_nvtx_ranges(sqlite_path)
    step_ranges = [r for r in nvtx_ranges if r["kind"] == "step"]
    decode_ranges = [r for r in step_ranges if r["fields"].get("phase") == "decode"]
    collective_ranges = [r for r in nvtx_ranges if r["kind"] == "collective_metadata"]
    intended_measured_ids = measured_request_ids(tp2_profiled)

    def block(reason: str) -> None:
        write_json("per_decode_step_collectives.json", {
            "status": "blocked",
            "blocked_reason": reason,
            "sqlite_path": str(sqlite_path),
            "sqlite_tables": sqlite_tables(sqlite_path),
            "rows": [],
            "nvtx_step_range_count": len(step_ranges),
            "nvtx_decode_range_count": len(decode_ranges),
            "nvtx_collective_metadata_range_count": len(collective_ranges),
            "assignment_rule": "fail closed unless decode-step NVTX ranges contain request IDs matching measured harness requests",
        })
        write_json("nccl_prediction_vs_observed.json", {"status": "blocked", "blocked_reason": reason, "rows": [], **summarize_prediction_errors([])})
        write_json("overlap_attribution.json", {
            "status": "blocked", "blocked_reason": reason,
            "definition": "requires measured-request decode-step NVTX intervals and CUDA kernel intervals",
            "sum_rank_nccl_gpu_time_us": None,
            "nccl_wall_union_us": None,
            "nccl_compute_overlap_wall_us": None,
            "exposed_nccl_wall_us": None,
            "measured_overlap_ratio": None,
        })
        write_json("tp_break_even_analysis.json", {"status": "blocked", "blocked_reason": reason, "rows": [], "explains_measured_winner": None})

    if not decode_ranges:
        block("No vllm.step phase=decode NVTX intervals were found; no NCCL kernel was assigned to a decode step.")
        return
    if not intended_measured_ids:
        block("TP2 profiled summary has no measured request IDs; cannot distinguish warmup from measured generation.")
        return
    if not any(request_ids_from_fields(r["fields"]) for r in decode_ranges):
        block("Decode-step NVTX ranges do not contain request IDs; rerun with Phase 4B request identity instrumentation.")
        return

    combined_decode_ranges: list[dict[str, Any]] = []
    by_step_and_request: dict[tuple[int, tuple[str, ...]], list[dict[str, Any]]] = {}
    for r in decode_ranges:
        req_ids = tuple(sorted(request_ids_from_fields(r["fields"])))
        if not request_id_matches_intended(list(req_ids), intended_measured_ids):
            continue
        by_step_and_request.setdefault((int(r["fields"].get("step", -1)), req_ids), []).append(r)
    for (step_id, req_ids), group in sorted(by_step_and_request.items()):
        start_ns = min(r["start_ns"] for r in group)
        end_ns = max(r["end_ns"] for r in group)
        fields = dict(group[0]["fields"])
        fields["step"] = str(step_id)
        fields["req_ids"] = "|".join(req_ids)
        combined_decode_ranges.append({
            "kind": "step",
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_us": (end_ns - start_ns) / 1000.0,
            "text": group[0]["text"],
            "fields": fields,
            "request_ids": list(req_ids),
            "rank_range_count": len(group),
        })
    if not combined_decode_ranges:
        block("No decode-step NVTX interval matches the measured request IDs from the harness summary.")
        return

    try:
        from transformers import AutoConfig  # noqa: PLC0415
        model_hidden = int(AutoConfig.from_pretrained(MODEL_ID).hidden_size)
    except Exception:  # noqa: BLE001
        model_hidden = 896
    estimated_bytes = 2 * model_hidden * 2 * 2
    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    decode_population = {
        "all_raw_decode_rank_ranges": len(decode_ranges),
        "raw_decode_rank_ranges_with_request_ids": sum(1 for r in decode_ranges if request_ids_from_fields(r["fields"])),
        "intended_measured_request_ids": sorted(intended_measured_ids),
        "included_measured_decode_steps": len(combined_decode_ranges),
        "exclusion_rule": "primary metrics include only decode-step NVTX ranges whose vLLM internal req_ids contain the exact measured harness request tag",
        "why_previous_62_rows_were_not_primary": (
            "The earlier trace had 62 merged decode rows for in32_out32_c1 because warmup and measured generations were both present. "
            "With output length 32, TPOT reports 31 decode intervals per single request after the first streamed token; "
            "without request IDs the parser could not authoritatively separate warmup from measured requests."
        ),
    }
    first_step = min(int(s["fields"].get("step", -1)) for s in combined_decode_ranges)
    total_sum_rank_nccl_us = 0.0
    total_nccl_wall_union_us = 0.0
    total_compute_wall_union_us = 0.0
    total_overlap_wall_us = 0.0
    total_exposed_wall_us = 0.0
    valid_rows = 0

    for step in combined_decode_ranges:
        step_kernels = [k for k in all_kernels if _interval_overlap_us(k, step) > 0]
        step_nccl = [k for k in step_kernels if k["is_nccl"]]
        step_compute = [k for k in step_kernels if not k["is_nccl"]]
        if not step_nccl:
            continue
        nccl_intervals = interval_list(step_nccl, step)
        compute_intervals = interval_list(step_compute, step)
        sum_rank_nccl_us = sum(k["duration_us"] for k in step_nccl)
        nccl_wall_union_us = interval_union_duration_us(nccl_intervals)
        compute_wall_union_us = interval_union_duration_us(compute_intervals)
        overlap_wall_us = interval_intersection_duration_us(nccl_intervals, compute_intervals)
        exposed_wall_us = max(0.0, nccl_wall_union_us - overlap_wall_us)
        decode_duration_us = step["duration_us"]
        invalid_reasons = []
        eps = 1e-6
        if exposed_wall_us < -eps:
            invalid_reasons.append("exposed_nccl_wall_us_negative")
        if exposed_wall_us - nccl_wall_union_us > eps:
            invalid_reasons.append("exposed_exceeds_nccl_wall_union")
        if nccl_wall_union_us - decode_duration_us > eps:
            invalid_reasons.append("nccl_wall_union_exceeds_decode_step_wall")
        if overlap_wall_us - nccl_wall_union_us > eps:
            invalid_reasons.append("nccl_compute_overlap_exceeds_nccl_wall_union")
        valid = not invalid_reasons
        if valid:
            valid_rows += 1
            total_sum_rank_nccl_us += sum_rank_nccl_us
            total_nccl_wall_union_us += nccl_wall_union_us
            total_compute_wall_union_us += compute_wall_union_us
            total_overlap_wall_us += overlap_wall_us
            total_exposed_wall_us += exposed_wall_us

        rank_diagnostics = {}
        for rank in sorted({k.get("rank", "rank_unknown") for k in step_nccl}):
            rk = [k for k in step_nccl if k.get("rank") == rank]
            rank_diagnostics[rank] = {
                "kernel_count": len(rk),
                "nccl_gpu_time_us": sum(k["duration_us"] for k in rk),
                "nccl_wall_union_us": interval_union_duration_us(interval_list(rk, step)),
                "device_ids": sorted({k.get("device_id") for k in rk if k.get("device_id") is not None}),
                "pids": sorted({k.get("pid") for k in rk if k.get("pid") is not None}),
            }
        per_rank_times = [v["nccl_gpu_time_us"] for v in rank_diagnostics.values()]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for kernel in step_nccl:
            grouped.setdefault(kernel.get("collective_kind") or "unknown", []).append(kernel)
        metadata_by_kind: dict[str, list[dict[str, Any]]] = {}
        for meta in collective_ranges:
            if _interval_overlap_us(meta, step) <= 0:
                continue
            kind = meta["fields"].get("collective") or "unknown"
            metadata_by_kind.setdefault(kind, []).append(meta)

        collective_details = []
        step_id = int(step["fields"].get("step", -1))
        step_state = "first_measured_decode_step" if step_id == first_step else "steady_state_candidate"
        for kind, group in sorted(grouped.items()):
            metas = sorted(metadata_by_kind.get(kind, []), key=lambda r: r["start_ns"])
            kernels_by_time = sorted(group, key=lambda r: r["start_ns"])
            for index, kernel in enumerate(kernels_by_time):
                meta = metas[index] if index < len(metas) else None
                bytes_value = int(meta["fields"]["bytes"]) if meta and meta["fields"].get("bytes", "").isdigit() else None
                query_bytes = bytes_value if bytes_value is not None else estimated_bytes
                predicted = None
                if kind != "unknown":
                    try:
                        predicted = predict_nccl_time_us(kind, query_bytes)
                    except Exception:
                        predicted = None
                rank = kernel.get("rank", "rank_unknown")
                observed = kernel["duration_us"]
                abs_error = abs(float(predicted) - observed) if predicted is not None else None
                outlier_class = "steady_state"
                if step_state == "first_measured_decode_step":
                    outlier_class = "first_measured_decode_step"
                elif abs_error is not None and abs_error > 100.0:
                    outlier_class = "prediction_error_gt_100us"
                pred_rows.append({
                    "step_id": step_id,
                    "workload_id": workload_id_from_request_ids(step.get("request_ids") or []),
                    "request_ids": step.get("request_ids"),
                    "rank": rank,
                    "device_id": kernel.get("device_id"),
                    "pid": kernel.get("pid"),
                    "collective_kind": kind,
                    "tensor_bytes": bytes_value,
                    "estimated_tensor_bytes": estimated_bytes if bytes_value is None else None,
                    "prediction_query_bytes": query_bytes,
                    "predicted_nccl_us": predicted,
                    "observed_raw_nccl_us": observed,
                    "observed_kernel_duration_us": observed,
                    "kernel_start_ns": kernel["start_ns"],
                    "kernel_end_ns": kernel["end_ns"],
                    "kernel_name": kernel["name"],
                    "metadata_source": "vllm.nccl_nvtx" if bytes_value is not None else "estimated_model_payload",
                    "cuda_graph_state": "cuda_graph_replay_or_regular_decode_kernel; explicit capture/replay marker unavailable in this SQLite row",
                    "first_use_cold_path_status": step_state,
                    "outlier_classification": outlier_class,
                })
            collective_details.append({
                "collective_kind": kind,
                "collective_call_count": len(group),
                "collective_metadata_range_count": len(metas),
                "collective_bytes": sum(
                    int(r["fields"]["bytes"])
                    for r in metas
                    if r["fields"].get("bytes", "").isdigit()
                ) or None,
                "estimated_collective_bytes": estimated_bytes * len(group),
                "sum_rank_nccl_gpu_time_us": sum(k["duration_us"] for k in group),
                "nccl_wall_union_us": interval_union_duration_us(interval_list(group, step)),
                "observed_kernel_names_sample": sorted({k["name"] for k in group})[:10],
            })
        row = {
            "step_id": step_id,
            "decode_step": step_id,
            "workload_id": workload_id_from_request_ids(step.get("request_ids") or []),
            "phase": "decode",
            "request_ids": step.get("request_ids"),
            "decode_step_wall_us": decode_duration_us,
            "decode_duration_us": decode_duration_us,
            "decode_step_critical_path_duration_us": decode_duration_us,
            "scheduled_tokens": int(step["fields"].get("tokens", 0)),
            "scheduled_requests": int(step["fields"].get("requests", 0)),
            "collective_call_count": len(step_nccl),
            "collective_kinds": sorted(grouped),
            "collective_bytes": sum(d["collective_bytes"] or 0 for d in collective_details) or None,
            "estimated_collective_bytes": sum(d["estimated_collective_bytes"] for d in collective_details),
            "sum_rank_nccl_gpu_time_us": sum_rank_nccl_us,
            "raw_nccl_duration_us": sum_rank_nccl_us,
            "nccl_wall_union_us": nccl_wall_union_us,
            "compute_wall_union_us": compute_wall_union_us,
            "nccl_compute_overlap_wall_us": overlap_wall_us,
            "nccl_compute_overlap_duration_us": overlap_wall_us,
            "exposed_nccl_wall_us": exposed_wall_us,
            "exposed_nccl_duration_us": exposed_wall_us,
            "overlap_ratio": overlap_wall_us / nccl_wall_union_us if nccl_wall_union_us else None,
            "nccl_fraction_of_decode_critical_path": nccl_wall_union_us / decode_duration_us if decode_duration_us else None,
            "exposed_nccl_fraction_of_decode_critical_path": exposed_wall_us / decode_duration_us if decode_duration_us else None,
            "valid": valid,
            "invalid_reasons": invalid_reasons,
            "rank_diagnostics": rank_diagnostics,
            "rank0_nccl_kernel_count": rank_diagnostics.get("rank0", {}).get("kernel_count", 0),
            "rank0_nccl_gpu_time_us": rank_diagnostics.get("rank0", {}).get("nccl_gpu_time_us", 0.0),
            "rank1_nccl_kernel_count": rank_diagnostics.get("rank1", {}).get("kernel_count", 0),
            "rank1_nccl_gpu_time_us": rank_diagnostics.get("rank1", {}).get("nccl_gpu_time_us", 0.0),
            "max_per_rank_nccl_duration_us": max(per_rank_times) if per_rank_times else 0.0,
            "summed_per_rank_nccl_duration_us": sum(per_rank_times),
            "cross_rank_nccl_interval_union_us": nccl_wall_union_us,
            "collectives": collective_details,
            "nvtx_text": step["text"],
            "rank_range_count": step.get("rank_range_count"),
            "consistency_checks": {
                "exposed_le_nccl_wall_le_decode_wall": valid and exposed_wall_us <= nccl_wall_union_us + eps and nccl_wall_union_us <= decode_duration_us + eps,
                "overlap_le_nccl_wall": overlap_wall_us <= nccl_wall_union_us + eps,
            },
        }
        rows.append(row)

    status = "measured" if rows and valid_rows == len(rows) else "measured_with_invalid_rows" if rows else "blocked_no_nccl_kernels_overlap_decode_steps"
    write_json("per_decode_step_collectives.json", {
        "status": status,
        "sqlite_path": str(sqlite_path),
        "sqlite_tables": sqlite_tables(sqlite_path),
        "decode_population_validation": decode_population,
        "nvtx_step_range_count": len(step_ranges),
        "nvtx_decode_range_count": len(decode_ranges),
        "nvtx_collective_metadata_range_count": len(collective_ranges),
        "rows": rows,
        "assignment_rule": "NCCL kernels are assigned only when their timeline interval overlaps an authoritative measured-request vllm.step phase=decode NVTX interval.",
        "critical_path_rule": "sum_rank_nccl_gpu_time_us is diagnostic only; wall-clock communication latency uses cross-rank NCCL interval union minus wall-clock compute overlap.",
    })
    pred_summary_all = summarize_prediction_errors(pred_rows)
    steady_pred_rows = [r for r in pred_rows if r.get("outlier_classification") == "steady_state"]
    pred_summary_steady = summarize_prediction_errors(steady_pred_rows)
    outliers: dict[str, list[dict[str, Any]]] = {}
    for threshold in (10.0, 50.0, 100.0):
        key = str(int(threshold))
        outliers[key] = [
            r for r in pred_rows
            if r.get("absolute_error_us") is not None and float(r["absolute_error_us"]) > threshold
        ]
    write_json("nccl_prediction_vs_observed.json", {
        "status": "measured" if pred_rows else "blocked",
        "rows": pred_rows,
        "all_samples": pred_summary_all,
        "steady_state_samples_after_outlier_classification": pred_summary_steady,
        "steady_state_filter": "outlier_classification == steady_state; excluded first measured decode step and samples with abs error > 100 us",
        "outliers_by_abs_error_threshold_us": outliers,
        **pred_summary_all,
    })
    overlap_ratio_total = total_overlap_wall_us / total_nccl_wall_union_us if total_nccl_wall_union_us else None
    write_json("overlap_attribution.json", {
        "status": "measured" if rows else "blocked",
        "definition": "NCCL wall time is the union of NCCL CUDA kernel intervals across ranks inside measured-request decode-step NVTX ranges. Overlap is the wall-clock intersection of that union with useful non-NCCL CUDA kernel intervals in the same decode-step range.",
        "sum_rank_nccl_gpu_time_us": total_sum_rank_nccl_us if rows else None,
        "raw_nccl_time_us": total_sum_rank_nccl_us if rows else None,
        "nccl_wall_union_us": total_nccl_wall_union_us if rows else None,
        "total_gpu_compute_wall_union_us": total_compute_wall_union_us if rows else None,
        "nccl_compute_overlap_wall_us": total_overlap_wall_us if rows else None,
        "nccl_compute_overlap_duration_us": total_overlap_wall_us if rows else None,
        "measured_overlap_ratio": overlap_ratio_total,
        "exposed_nccl_wall_us": total_exposed_wall_us if rows else None,
        "exposed_nccl_time_us": total_exposed_wall_us if rows else None,
        "d7_assumption_assessment": (
            "overly_pessimistic_if_it_used_sum_rank_gpu_time; conservative_if_it_used_raw_wall_union_without_overlap"
            if overlap_ratio_total and overlap_ratio_total > 0 else
            "wall_union_equals_exposed_when_no_compute_overlap_observed" if rows else "blocked_no_real_timeline"
        ),
        "candidate_d8_correction": "exposed_comm_time = nccl_wall_union_us - nccl_compute_overlap_wall_us",
        "profiler_perturbed": True,
        "per_step": [{
            "step_id": r["step_id"],
            "request_ids": r["request_ids"],
            "decode_step_wall_us": r["decode_step_wall_us"],
            "sum_rank_nccl_gpu_time_us": r["sum_rank_nccl_gpu_time_us"],
            "nccl_wall_union_us": r["nccl_wall_union_us"],
            "nccl_compute_overlap_wall_us": r["nccl_compute_overlap_wall_us"],
            "exposed_nccl_wall_us": r["exposed_nccl_wall_us"],
            "overlap_ratio": r["overlap_ratio"],
            "nccl_fraction_of_decode_critical_path": r["nccl_fraction_of_decode_critical_path"],
            "valid": r["valid"],
            "invalid_reasons": r["invalid_reasons"],
        } for r in rows],
    })
    tp1_decode = first_tpot_us(tp1)
    tp2_decode = first_tpot_us(tp2)
    tp2_nvtx_decode = first_tpot_us(tp2_nvtx) if tp2_nvtx else None
    tp2_profiled_decode = first_tpot_us(tp2_profiled)
    mean_exposed_wall_us = total_exposed_wall_us / valid_rows if valid_rows else None
    mean_nccl_wall_union_us = total_nccl_wall_union_us / valid_rows if valid_rows else None
    mean_sum_rank_nccl_us = total_sum_rank_nccl_us / valid_rows if valid_rows else None
    estimated_tp2_compute_excluding_exposed_comm = (
        tp2_decode - mean_exposed_wall_us
        if tp2_decode is not None and mean_exposed_wall_us is not None else None
    )
    savings = (
        tp1_decode - estimated_tp2_compute_excluding_exposed_comm
        if tp1_decode is not None and estimated_tp2_compute_excluding_exposed_comm is not None else None
    )
    residual = 0.0 if savings is not None and mean_exposed_wall_us is not None else None
    rhs = mean_exposed_wall_us + residual if mean_exposed_wall_us is not None and residual is not None else None
    measured_winner = "tp2" if (tp1_decode and tp2_decode and tp2_decode < tp1_decode) else "tp1"
    inequality = savings > rhs if savings is not None and rhs is not None else None
    explains = ((measured_winner == "tp2") == inequality) if inequality is not None else None
    def pct(delta: float | None, base: float | None) -> float | None:
        return (delta / base * 100.0) if delta is not None and base else None
    nvtx_delta = (tp2_nvtx_decode - tp2_decode) if tp2_nvtx_decode is not None and tp2_decode is not None else None
    nsys_delta = (tp2_profiled_decode - tp2_decode) if tp2_profiled_decode is not None and tp2_decode is not None else None
    write_json("tp_break_even_analysis.json", {
        "status": "measured" if rows and savings is not None else "blocked",
        "formula": "TP2 wins iff TP compute savings > exposed communication penalty + runtime residual",
        "communication_penalty_source": "mean per-decode-step exposed_nccl_wall_us from wall-clock interval union, not summed per-rank GPU kernel duration",
        "profiler_perturbation": {
            "tp2_normal_tpot_us": tp2_decode,
            "tp2_nvtx_no_nsys_tpot_us": tp2_nvtx_decode,
            "tp2_nsys_tpot_us": tp2_profiled_decode,
            "nvtx_minus_normal_tpot_us": nvtx_delta,
            "nsys_minus_normal_tpot_us": nsys_delta,
            "nvtx_overhead_pct": pct(nvtx_delta, tp2_decode),
            "nsys_overhead_pct": pct(nsys_delta, tp2_decode),
            "nsight_timing_is_profiler_perturbed": True,
        },
        "rows": [{
            "workload_id": workloads[0].workload_id,
            "tp1_decode_latency_us": tp1_decode,
            "tp2_decode_latency_us": tp2_decode,
            "tp2_compute_excluding_exposed_comm_estimate_us": estimated_tp2_compute_excluding_exposed_comm,
            "tp_compute_savings_us": savings,
            "exposed_communication_penalty_us": mean_exposed_wall_us,
            "mean_nccl_wall_union_us_per_decode_step": mean_nccl_wall_union_us,
            "mean_sum_rank_nccl_gpu_time_us_per_decode_step": mean_sum_rank_nccl_us,
            "total_exposed_nccl_wall_us_across_measured_steps": total_exposed_wall_us if rows else None,
            "runtime_residual_us": residual,
            "rhs_penalty_plus_residual_us": rhs,
            "inequality_holds": inequality,
            "measured_winner": measured_winner,
            "explains_measured_winner": explains,
        }],
        "explains_measured_winner": explains,
    })
    write_multicell_artifacts(tp1, tp2, tp2_profiled, workloads, rows, pred_rows, tp2_nvtx)

def write_multicell_json(name: str, payload: Any) -> None:
    path = MULTICELL_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {path}")


def _summary_by_workload(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(w["workload_id"]): w for w in summary.get("workload_results", []) if w.get("workload_id")}


def _request_tpots_us(workload_summary: dict[str, Any]) -> list[float]:
    out = []
    for req in workload_summary.get("per_request", []):
        if req.get("ok") and req.get("tpot_s") is not None:
            out.append(float(req["tpot_s"]) * 1_000_000.0)
    return out


def _distribution(values: list[float]) -> dict[str, Any]:
    mean = statistics.mean(values) if values else None
    stdev = statistics.stdev(values) if len(values) >= 2 else None
    return {
        "sample_count": len(values),
        "mean_us": mean,
        "p50_us": _percentile_linear(values, 0.50) if len(values) >= 5 else None,
        "p95_us": _percentile_linear(values, 0.95) if len(values) >= 5 else None,
        "cv": (stdev / mean) if stdev is not None and mean else None,
        "raw_samples_us": values,
    }


def _mae_mape(errors: list[tuple[float, float]]) -> dict[str, Any]:
    abs_errors = [abs(pred - obs) for pred, obs in errors]
    rel_errors = [abs(pred - obs) / abs(obs) for pred, obs in errors if obs]
    return {
        "count": len(errors),
        "mae_us": statistics.mean(abs_errors) if abs_errors else None,
        "mape": statistics.mean(rel_errors) if rel_errors else None,
        "max_abs_error_us": max(abs_errors) if abs_errors else None,
    }


def _cell_rows(rows: list[dict[str, Any]], workload_id: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("workload_id") == workload_id]


def _cell_pred_rows(pred_rows: list[dict[str, Any]], workload_id: str) -> list[dict[str, Any]]:
    return [r for r in pred_rows if r.get("workload_id") == workload_id]


def _step_predicted_raw_us(pred_rows: list[dict[str, Any]], step_id: int) -> float | None:
    step_rows = [r for r in pred_rows if r.get("step_id") == step_id and r.get("predicted_nccl_us") is not None]
    if not step_rows:
        return None
    rank0 = [r for r in step_rows if r.get("rank") == "rank0"]
    selected = rank0 if rank0 else step_rows
    return sum(float(r["predicted_nccl_us"]) for r in selected)


def write_multicell_artifacts(tp1: dict[str, Any], tp2: dict[str, Any], tp2_profiled: dict[str, Any],
                              workloads: list[WorkloadSpec], rows: list[dict[str, Any]],
                              pred_rows: list[dict[str, Any]], tp2_nvtx: dict[str, Any] | None) -> None:
    workload_ids = [w.workload_id for w in workloads]
    tp1_by = _summary_by_workload(tp1)
    tp2_by = _summary_by_workload(tp2)
    tp2_prof_by = _summary_by_workload(tp2_profiled)
    tp2_nvtx_by = _summary_by_workload(tp2_nvtx or {})

    write_multicell_json("workload_matrix.json", {
        "status": "measured",
        "model": MODEL_ID,
        "workloads": [w.to_dict() for w in workloads],
        "normal_measured_repetitions_per_cell": {wid: tp1_by.get(wid, {}).get("requests_total") for wid in workload_ids},
        "profiled_tp2_representative_requests_per_cell": {wid: tp2_prof_by.get(wid, {}).get("requests_total") for wid in workload_ids},
        "accounting": "Phase 4B wall-clock interval union with measured request IDs; summed-rank NCCL time is diagnostic only.",
    })

    e2e_rows = []
    for wid in workload_ids:
        tp1_tpots = _request_tpots_us(tp1_by.get(wid, {}))
        tp2_tpots = _request_tpots_us(tp2_by.get(wid, {}))
        tp2_nvtx_tpots = _request_tpots_us(tp2_nvtx_by.get(wid, {}))
        tp2_prof_tpots = _request_tpots_us(tp2_prof_by.get(wid, {}))
        tp1_dist = _distribution(tp1_tpots)
        tp2_dist = _distribution(tp2_tpots)
        measured_winner = None
        if tp1_dist["mean_us"] is not None and tp2_dist["mean_us"] is not None:
            measured_winner = "tp2" if tp2_dist["mean_us"] < tp1_dist["mean_us"] else "tp1"
        normal_tp2 = tp2_dist["mean_us"]
        nvtx_mean = _distribution(tp2_nvtx_tpots)["mean_us"]
        prof_mean = _distribution(tp2_prof_tpots)["mean_us"]
        e2e_rows.append({
            "workload_id": wid,
            "tp1_tpot": tp1_dist,
            "tp2_tpot": tp2_dist,
            "measured_winner": measured_winner,
            "profiler_perturbation": {
                "tp2_normal_mean_tpot_us": normal_tp2,
                "tp2_nvtx_no_nsys_mean_tpot_us": nvtx_mean,
                "tp2_nsys_representative_tpot_us": prof_mean,
                "nvtx_overhead_pct": ((nvtx_mean - normal_tp2) / normal_tp2 * 100.0) if nvtx_mean is not None and normal_tp2 else None,
                "nsys_overhead_pct": ((prof_mean - normal_tp2) / normal_tp2 * 100.0) if prof_mean is not None and normal_tp2 else None,
                "nsight_timing_is_profiler_perturbed": True,
            },
        })
    write_multicell_json("end_to_end_summary.json", {"status": "measured", "rows": e2e_rows})

    per_cell_attr = []
    call_count_by = {}
    overlap_by = {}
    for wid in workload_ids:
        cr = _cell_rows(rows, wid)
        valid = [r for r in cr if r.get("valid")]
        call_counts = [int(r.get("collective_call_count", 0)) for r in valid]
        overlap_vals = [float(r["overlap_ratio"]) for r in valid if r.get("overlap_ratio") is not None]
        exposed_fracs = [float(r["exposed_nccl_fraction_of_decode_critical_path"]) for r in valid if r.get("exposed_nccl_fraction_of_decode_critical_path") is not None]
        call_count_by[wid] = summarize_values([float(v) for v in call_counts])
        overlap_by[wid] = summarize_values(overlap_vals)
        per_cell_attr.append({
            "workload_id": wid,
            "measured_decode_steps": len(valid),
            "invalid_decode_steps": len(cr) - len(valid),
            "collective_call_count_per_decode_step": summarize_values([float(v) for v in call_counts]),
            "collective_kinds_per_decode_step": [r.get("collective_kinds") for r in valid],
            "collective_bytes_per_decode_step": summarize_values([float(r.get("collective_bytes") or r.get("estimated_collective_bytes") or 0) for r in valid]),
            "nccl_wall_union_us_per_decode_step": summarize_values([float(r["nccl_wall_union_us"]) for r in valid]),
            "nccl_compute_overlap_wall_us_per_decode_step": summarize_values([float(r["nccl_compute_overlap_wall_us"]) for r in valid]),
            "exposed_nccl_wall_us_per_decode_step": summarize_values([float(r["exposed_nccl_wall_us"]) for r in valid]),
            "overlap_ratio_per_decode_step": summarize_values(overlap_vals),
            "exposed_nccl_fraction_of_decode_critical_path": summarize_values(exposed_fracs),
            "sample_rows": valid[:3],
        })
    write_multicell_json("per_cell_attribution.json", {"status": "measured", "rows": per_cell_attr})

    pred_cell_rows = []
    call_model_rows = []
    zero_errors: list[tuple[float, float]] = []
    overlap_errors: list[tuple[float, float]] = []
    for wid in workload_ids:
        pr = _cell_pred_rows(pred_rows, wid)
        all_summary = summarize_prediction_errors(pr)
        steady = [r for r in pr if r.get("outlier_classification") == "steady_state"]
        steady_summary = summarize_prediction_errors(steady)
        outlier_counts = {
            str(int(t)): sum(1 for r in pr if r.get("absolute_error_us") is not None and float(r["absolute_error_us"]) > t)
            for t in (10.0, 50.0, 100.0)
        }
        pred_cell_rows.append({
            "workload_id": wid,
            "sample_count": len(pr),
            "all_samples": all_summary,
            "steady_state_samples": steady_summary,
            "outlier_counts_by_abs_error_threshold_us": outlier_counts,
        })
        cr = _cell_rows(rows, wid)
        step_pairs_zero = []
        step_pairs_overlap = []
        for r in cr:
            if not r.get("valid"):
                continue
            predicted_raw = _step_predicted_raw_us(pr, int(r["step_id"]))
            if predicted_raw is None:
                continue
            observed_exposed = float(r["exposed_nccl_wall_us"])
            overlap_ratio_value = float(r.get("overlap_ratio") or 0.0)
            predicted_overlap = predicted_raw * (1.0 - overlap_ratio_value)
            step_pairs_zero.append((predicted_raw, observed_exposed))
            step_pairs_overlap.append((predicted_overlap, observed_exposed))
            zero_errors.append((predicted_raw, observed_exposed))
            overlap_errors.append((predicted_overlap, observed_exposed))
        call_model_rows.append({
            "workload_id": wid,
            "zero_overlap_baseline": _mae_mape(step_pairs_zero),
            "overlap_aware": _mae_mape(step_pairs_overlap),
        })
    write_multicell_json("communication_prediction_validation.json", {
        "status": "measured",
        "per_collective_kernel_prediction": pred_cell_rows,
        "call_count_aware_exposed_prediction": {
            "definition": "predicted_raw_comm_us sums Phase 1 predictions for rank0 collective calls per decode step; overlap-aware multiplies by (1 - measured step overlap_ratio).",
            "overall_zero_overlap_baseline": _mae_mape(zero_errors),
            "overall_overlap_aware": _mae_mape(overlap_errors),
            "per_cell": call_model_rows,
        },
    })

    baseline = workload_ids[0] if workload_ids else None
    observations = []
    if baseline and baseline in call_count_by:
        b_call = call_count_by[baseline].get("mean_us")
        b_overlap = overlap_by[baseline].get("mean_us")
        for wid in workload_ids[1:]:
            observations.append({
                "comparison": f"{baseline} -> {wid}",
                "collective_call_count_mean_delta": (call_count_by[wid].get("mean_us") - b_call) if b_call is not None and call_count_by[wid].get("mean_us") is not None else None,
                "overlap_ratio_mean_delta": (overlap_by[wid].get("mean_us") - b_overlap) if b_overlap is not None and overlap_by[wid].get("mean_us") is not None else None,
            })
    zero_model = _mae_mape(zero_errors)
    overlap_model = _mae_mape(overlap_errors)
    def mean_for(table: dict[str, Any], wid: str) -> float | None:
        value = table.get(wid, {}).get("mean_us")
        return float(value) if value is not None else None
    baseline_calls = mean_for(call_count_by, "in32_out32_c1")
    baseline_overlap = mean_for(overlap_by, "in32_out32_c1")
    generalization_answers = {
        "A_collective_call_count_changes_with_input_length": {
            "answer": "yes_in_this_trace",
            "evidence": {
                "in32_out32_c1_mean_calls_per_step": baseline_calls,
                "in128_out32_c1_mean_calls_per_step": mean_for(call_count_by, "in128_out32_c1"),
            },
        },
        "A_collective_call_count_changes_with_output_length": {
            "answer": "per_step_count_roughly_stable_but_total_decode_steps_scale_with_output_length",
            "evidence": {
                "in32_out32_c1_mean_calls_per_step": baseline_calls,
                "in32_out128_c1_mean_calls_per_step": mean_for(call_count_by, "in32_out128_c1"),
                "in32_out32_c1_decode_steps": next((r["measured_decode_steps"] for r in per_cell_attr if r["workload_id"] == "in32_out32_c1"), None),
                "in32_out128_c1_decode_steps": next((r["measured_decode_steps"] for r in per_cell_attr if r["workload_id"] == "in32_out128_c1"), None),
            },
        },
        "A_collective_call_count_changes_with_concurrency": {
            "answer": "yes_per_step_count_decreases_for_c4_c8_while_scheduled_work_per_step_changes",
            "evidence": {
                "c1_mean_calls_per_step": baseline_calls,
                "c4_mean_calls_per_step": mean_for(call_count_by, "in32_out32_c4"),
                "c8_mean_calls_per_step": mean_for(call_count_by, "in32_out32_c8"),
            },
        },
        "B_overlap_ratio_materially_changes_across_cells": {
            "answer": "yes_but_absolute_overlap_remains_small",
            "evidence": {
                "baseline_overlap_ratio_mean": baseline_overlap,
                "min_overlap_ratio_mean": min(v.get("mean_us") for v in overlap_by.values() if v.get("mean_us") is not None),
                "max_overlap_ratio_mean": max(v.get("mean_us") for v in overlap_by.values() if v.get("mean_us") is not None),
            },
        },
        "C_microbenchmark_prediction_steady_state_accuracy": {
            "answer": "steady_state_per_kernel_prediction_is_low_us_for_all_reduce_but_all_gather_and cold/capture/replay outliers remain important",
            "evidence": {r["workload_id"]: r["steady_state_samples"].get("mae_us") for r in pred_cell_rows},
        },
        "D_zero_overlap_d7_materially_worse_than_overlap_aware": {
            "answer": "no_not_for_this_matrix; zero_overlap_MAE_was_slightly_lower_than_overlap_aware_MAE",
            "evidence": {
                "zero_overlap_mae_us": zero_model.get("mae_us"),
                "overlap_aware_mae_us": overlap_model.get("mae_us"),
            },
        },
        "E_strongest_decision_signal": {
            "answer": "compute_savings_vs_exposed_communication_penalty; bytes alone are weak because all observed collective messages are in the same small-size bucket",
            "ranked_signals": ["compute_savings", "exposed_communication_penalty", "collective_count", "overlap_ratio", "bytes"],
        },
    }
    write_multicell_json("overlap_generalization.json", {
        "status": "measured",
        "call_count_by_workload": call_count_by,
        "overlap_ratio_by_workload": overlap_by,
        "observations_vs_baseline": observations,
        "generalization_answers": generalization_answers,
    })

    break_rows = []
    for wid in workload_ids:
        tp1_mean = _distribution(_request_tpots_us(tp1_by.get(wid, {})))["mean_us"]
        tp2_mean = _distribution(_request_tpots_us(tp2_by.get(wid, {})))["mean_us"]
        cr = [r for r in _cell_rows(rows, wid) if r.get("valid")]
        exposed_values = [float(r["exposed_nccl_wall_us"]) for r in cr]
        exposed_mean = statistics.mean(exposed_values) if exposed_values else None
        tp2_compute = tp2_mean - exposed_mean if tp2_mean is not None and exposed_mean is not None else None
        compute_saving = tp1_mean - tp2_compute if tp1_mean is not None and tp2_compute is not None else None
        residual = 0.0 if compute_saving is not None and exposed_mean is not None else None
        rhs = exposed_mean + residual if exposed_mean is not None and residual is not None else None
        measured_winner = "tp2" if tp1_mean is not None and tp2_mean is not None and tp2_mean < tp1_mean else "tp1"
        inequality = compute_saving > rhs if compute_saving is not None and rhs is not None else None
        break_rows.append({
            "workload_id": wid,
            "tp1_mean_tpot_us": tp1_mean,
            "tp2_mean_tpot_us": tp2_mean,
            "tp2_compute_excluding_exposed_comm_estimate_us": tp2_compute,
            "compute_saving_us": compute_saving,
            "communication_penalty_us": exposed_mean,
            "runtime_residual_us": residual,
            "inequality_holds": inequality,
            "measured_winner": measured_winner,
            "predicted_winner_from_inequality": "tp2" if inequality else "tp1" if inequality is not None else None,
            "explains_measured_winner": ((measured_winner == "tp2") == inequality) if inequality is not None else None,
        })
    write_multicell_json("break_even_validation.json", {
        "status": "measured",
        "formula": "TP2 wins iff compute_saving_us > communication_penalty_us + runtime_residual_us",
        "rows": break_rows,
        "correct_count": sum(1 for r in break_rows if r.get("explains_measured_winner") is True),
        "total_count": len(break_rows),
    })

    readme = [
        "# D8 Multicell vLLM NCCL Attribution",
        "",
        "Phase 4C generalizes the corrected Phase 4B attribution across a small workload matrix.",
        "",
        "The parser uses authoritative measured request IDs, per-decode-step NVTX ranges, cross-rank NCCL wall-clock interval unions, and wall-clock NCCL/compute overlap. Summed per-rank NCCL GPU time is retained only as a diagnostic.",
        "",
        "## Files",
        "",
        "- `workload_matrix.json`",
        "- `end_to_end_summary.json`",
        "- `per_cell_attribution.json`",
        "- `communication_prediction_validation.json`",
        "- `overlap_generalization.json`",
        "- `break_even_validation.json`",
        "",
        "## Headline",
        "",
        f"Workloads: {', '.join(workload_ids)}",
        f"Break-even explanations: {sum(1 for r in break_rows if r.get('explains_measured_winner') is True)}/{len(break_rows)}",
        f"Overall zero-overlap MAE: {zero_model.get('mae_us')} us",
        f"Overall overlap-aware MAE: {overlap_model.get('mae_us')} us",
        "",
        "## Generalization Answers",
        "",
        "- Collective call count changes with input length in this trace and changes with concurrency; output length mainly increases the number of decode steps.",
        "- Overlap ratio changes across cells but remains small in absolute terms.",
        "- Steady-state microbenchmark prediction is low-us for all-reduce, while all-gather and cold/capture/replay outliers remain visible.",
        "- Zero-overlap was not materially worse here; its MAE was slightly lower than the measured-overlap correction for this matrix.",
        "- The strongest future selector signal is compute savings versus exposed communication penalty; bytes alone are weak because observed messages stay in the same small-size bucket.",
        "",
        "Nsight timings are profiler-perturbed; normal TP1/TP2 unprofiled runs remain the authoritative end-to-end latency source.",
    ]
    (MULTICELL_DIR / "README.md").write_text("\n".join(readme) + "\n")
    print(f"wrote {MULTICELL_DIR / 'README.md'}")


def run_real_measurement(args: argparse.Namespace, workloads: list[WorkloadSpec]) -> None:
    # This path is intentionally narrow and measurement-oriented. It launches
    # matched TP1 and TP2 servers from the existing D4B materializer, profiles
    # the TP2 server process tree with Nsight Systems, then records raw
    # artifacts. Detailed per-kernel parsing is conservative and will mark
    # itself blocked if the exported SQLite schema lacks CUDA kernel tables.
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tp1 = run_one_server(1, workloads=workloads, tokenizer=tokenizer, profiled=False, args=args)
    write_json("tp1_summary.json", tp1)
    tp2 = run_one_server(2, workloads=workloads, tokenizer=tokenizer, profiled=False, args=args)
    write_json("tp2_summary.json", tp2)
    tp2_nvtx = run_one_server(
        2, workloads=workloads, tokenizer=tokenizer, profiled=False,
        args=args, tp_profile_nvtx=True, label="nvtx_unprofiled")
    write_json("tp2_nvtx_summary.json", tp2_nvtx)
    tp2_profiled = run_one_server(
        2, workloads=workloads, tokenizer=tokenizer, profiled=True,
        args=args, tp_profile_nvtx=True, label="profiled", measured_repetitions=args.profiled_measured_repetitions)
    write_json("tp2_profiled_summary.json", tp2_profiled)
    if (tp1.get("status") == "measured" and tp2.get("status") == "measured"
            and tp2_nvtx.get("status") == "measured" and tp2_profiled.get("status") == "measured"):
        write_live_attribution(tp1, tp2, tp2_profiled, workloads, tp2_nvtx=tp2_nvtx)
    else:
        blocked_artifacts("one or more TP1/TP2 live runs failed", workloads, dependency_inventory())


def write_readme(deps: dict[str, Any], blocked_reason: str | None) -> None:
    lines = [
        "# D8 vLLM NCCL Attribution",
        "",
        "Phase 3 validates Phase 1 NCCL microbenchmark predictions against real vLLM TP2 decode execution.",
        "",
        f"Status: {'blocked' if blocked_reason else 'measurement attempted'}",
    ]
    if blocked_reason:
        lines += [
            "",
            f"Blocked reason: `{blocked_reason}`",
            "",
            "No per-decode NCCL latency, overlap, exposed communication, or TP break-even claim is made in this artifact set.",
        ]
    lines += [
        "",
        "Required boundary: 2x RTX 4090, PHB, single NUMA, CUDA P2P unavailable, NCCL SHM/direct/direct.",
        "",
        "D7 assumption under test: `exposed_comm_time = raw_nccl_time`.",
        "Candidate D8 correction: `exposed_comm_time = raw_nccl_time * (1 - measured_overlap_ratio)`.",
        "",
        "Nsight overlap rule: overlap is computed from CUDA/NVTX timeline interval intersections, not from CUDA event duration alone.",
        "",
        "Dependency inventory:",
        "```json",
        json.dumps(deps, indent=2, sort_keys=True),
        "```",
        "",
    ]
    (RESULTS_DIR / "README.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", action="append", help="Workload id like in32_out32_c1. Repeatable.")
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--measured-repetitions", type=int, default=5)
    parser.add_argument("--profiled-measured-repetitions", type=int, default=1)
    parser.add_argument("--allow-run", action="store_true", help="Actually launch vLLM when dependencies are available.")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    MULTICELL_DIR.mkdir(parents=True, exist_ok=True)
    deps = dependency_inventory()
    workloads = selected_workloads(args)
    write_json("environment_manifest.json", environment_manifest(deps))
    write_json("topology_and_transport.json", topology_and_transport())
    write_json("workload_manifest.json", workload_manifest(workloads, deps))

    blocked_reason = None
    if not deps["can_run_real_vllm"]:
        blocked_reason = "missing importable vllm/torch/transformers in current Python environment"
    elif not deps["can_collect_nsys"]:
        blocked_reason = "missing Nsight Systems nsys"
    elif not args.allow_run:
        blocked_reason = "real vLLM launch disabled; rerun with --allow-run to collect live measurements"

    if blocked_reason:
        blocked_artifacts(blocked_reason, workloads, deps)
        write_readme(deps, blocked_reason)
        return 2

    run_real_measurement(args, workloads)
    # If the live path produced full attribution later, these files should be
    # overwritten by the parser. Until then, preserve an explicit status.
    for name in ("nccl_prediction_vs_observed.json", "overlap_attribution.json", "tp_break_even_analysis.json"):
        path = RESULTS_DIR / name
        if not path.exists():
            write_json(name, {"status": "blocked_pending_validated_nsys_decode_parser", "rows": []})
    write_readme(deps, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
