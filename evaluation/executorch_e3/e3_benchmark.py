#!/usr/bin/env python3
"""E3 ExecuTorch/XNNPACK same-stack discovery benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ATOL = 1e-3
RTOL = 1e-4
SEED_OFFSET = 310000
MODES = ("X1", "X4", "DEFAULT")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(vals: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(vals), *vals)


def unpack(path: Path) -> list[float]:
    data = path.read_bytes()
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


def gen(m: int, n: int, k: int, seed: int) -> tuple[list[float], list[float], list[float]]:
    rng = random.Random(seed)
    return ([rng.uniform(-3, 3) for _ in range(m * k)],
            [rng.uniform(-3, 3) for _ in range(k * n)],
            [rng.uniform(-3, 3) for _ in range(n)])


def ref(a: list[float], b: list[float], bias: list[float], m: int, n: int, k: int) -> list[float]:
    out = []
    for i in range(m):
        for j in range(n):
            v = sum(a[i * k + kk] * b[kk * n + j] for kk in range(k)) + bias[j]
            out.append(struct.unpack("<f", struct.pack("<f", v if v > 0 else 0.0))[0])
    return out


def err(actual: list[float], expected: list[float]) -> dict:
    diffs = [abs(x - y) for x, y in zip(actual, expected)]
    passed = len(actual) == len(expected) and all(
        (not math.isnan(x)) and (not math.isinf(x)) and abs(x - y) <= ATOL + RTOL * abs(y)
        for x, y in zip(actual, expected)
    )
    return {
        "passed": passed,
        "max_abs_error": max(diffs) if diffs else None,
        "mean_abs_error": statistics.fmean(diffs) if diffs else None,
    }


def make_manifest(args: argparse.Namespace) -> dict:
    p1d = json.loads(Path(args.p1d_manifest).read_text())
    workloads = []
    for w in p1d["workloads"]:
        if w.get("split") not in {"calibration", "held_out"}:
            continue
        m, n, k = w["m"], w["n"], w["k"]
        seed = w["seed"] + SEED_OFFSET
        a, b, bias = gen(m, n, k, seed)
        r = ref(a, b, bias, m, n, k)
        pte = Path(args.pte_dir) / f"fused_matmul_bias_relu_{m}x{n}x{k}_xnnpack.pte"
        report = Path(args.pte_dir) / f"fused_matmul_bias_relu_{m}x{n}x{k}_xnnpack_export_report.json"
        er = json.loads(report.read_text())
        workloads.append({
            "workload_id": w["workload_id"], "split": w["split"], "category": w.get("category"),
            "m": m, "n": n, "k": k, "seed": seed, "mnk": m * n * k,
            "input_hashes": {"a": sha_bytes(pack(a)), "b": sha_bytes(pack(b)), "bias": sha_bytes(pack(bias)), "reference_fp64": sha_bytes(pack(r))},
            "pte": {"path": str(pte), "sha256": sha_file(pte), "bytes": pte.stat().st_size,
                    "delegate_call_count": er.get("delegate_call_count"), "classification": er.get("classification")},
        })
    man = {
        "schema": "e3_xnnpack_discovery_manifest", "schema_version": 1,
        "comparison_id": "E3_RPI5_FP32_FUSED_MATMUL_BIAS_RELU_XNNPACK_2026_07_14",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "modes": list(MODES),
        "protocol": {"warmups": args.warmups, "repeats": args.repeats, "sessions": args.sessions, "affinity": args.affinity, "governor": "performance"},
        "correctness_predicate": {"atol": ATOL, "rtol": RTOL, "form": "abs(actual - expected) <= atol + rtol * abs(expected)"},
        "executorch": {"tag": "v1.3.1", "commit": "e2f18eb23c45bd22ca332b0b8b49a81de304b472"},
        "xnnpack": {"commit": "1adaa7c709d4839d29e1f219cb962b01c9e6a905"},
        "default_classification": "executor_runner cpu_threads=-1 heuristic derives performant cores",
        "workloads": workloads,
    }
    frozen = json.dumps(man, sort_keys=True, separators=(",", ":")).encode()
    man["manifest_sha256"] = hashlib.sha256(frozen).hexdigest()
    return man


def write_inputs(man: dict, input_dir: Path) -> None:
    for w in man["workloads"]:
        root = input_dir / w["workload_id"]
        root.mkdir(parents=True, exist_ok=True)
        a, b, bias = gen(w["m"], w["n"], w["k"], w["seed"])
        r = ref(a, b, bias, w["m"], w["n"], w["k"])
        for name, vals in (("a", a), ("b", b), ("bias", bias), ("reference_fp64", r)):
            data = pack(vals)
            (root / f"{name}.bin").write_bytes(data)
            if sha_bytes(data) != w["input_hashes"][name]:
                raise RuntimeError(f"input hash mismatch {w['workload_id']} {name}")


def requested(mode: str) -> str:
    return {"X1": "1", "X4": "4", "DEFAULT": "default"}[mode]


def order(items: list[tuple[dict, str]], session: int) -> list[tuple[dict, str]]:
    out = list(items)
    if session % 3 == 1:
        out.reverse()
    elif session % 3 == 2:
        random.Random(90210 + session).shuffle(out)
    return out


def run_discovery(args: argparse.Namespace) -> None:
    man = json.loads(Path(args.manifest).read_text())
    input_dir = Path(args.input_dir)
    tmp = Path(args.tmp_dir); tmp.mkdir(parents=True, exist_ok=True)
    records = []
    items = [(w, m) for w in man["workloads"] for m in MODES]
    for session in range(man["protocol"]["sessions"]):
        for w, mode in order(items, session):
            out = tmp / f"{w['workload_id']}_{mode}_s{session}.bin"
            report = tmp / f"{w['workload_id']}_{mode}_s{session}.json"
            inp = input_dir / w["workload_id"]
            cmd = [args.runner, "--executor_runner", args.executor_runner, "--model_path", w["pte"]["path"],
                   "--input_a", str(inp / "a.bin"), "--input_b", str(inp / "b.bin"), "--input_bias", str(inp / "bias.bin"),
                   "--requested_threads", requested(mode), "--warmups", str(man["protocol"]["warmups"]),
                   "--repeats", str(man["protocol"]["repeats"]), "--output", str(out), "--result_json", str(report),
                   "--affinity", man["protocol"]["affinity"]]
            subprocess.run(cmd, check=True)
            rr = json.loads(report.read_text())
            actual = unpack(out)
            expected = unpack(inp / "reference_fp64.bin")
            correctness = err(actual, expected)
            warm = rr.get("warm_samples_ms", [])
            rec = {"workload_id": w["workload_id"], "split": w["split"], "m": w["m"], "n": w["n"], "k": w["k"],
                   "mode": mode, "session": session, "runner_report": rr, "correctness": correctness,
                   "warm_median_ms": statistics.median(warm) if warm else None,
                   "load_time_ms": rr.get("load_time_ms"), "timing_samples_complete": rr.get("timing_samples_complete")}
            print(session, w["workload_id"], mode, rec["warm_median_ms"], correctness["passed"], flush=True)
            records.append(rec)
    Path(args.out).write_text(json.dumps({"schema": "e3_xnnpack_discovery_raw", "manifest_sha256": man["manifest_sha256"], "records": records}, indent=2, sort_keys=True) + "\n")


def analyze(args: argparse.Namespace) -> None:
    raw = json.loads(Path(args.raw).read_text())
    per: dict[tuple[str, str], list[float]] = {}
    failures = []
    incomplete = []
    for r in raw["records"]:
        if not r["correctness"]["passed"]:
            failures.append(r)
        if not r.get("timing_samples_complete"):
            incomplete.append(r)
        per.setdefault((r["workload_id"], r["mode"]), []).append(r["warm_median_ms"])
    tie_threshold_percent = 5.0
    rows = []
    for wid in sorted({r["workload_id"] for r in raw["records"]}):
        modes = {m: statistics.median(per[(wid, m)]) for m in MODES}
        best = min((modes[m], m) for m in ("X1", "X4"))
        best_ms = best[0]
        x1_regret = (modes["X1"] - best_ms) / best_ms * 100.0
        x4_regret = (modes["X4"] - best_ms) / best_ms * 100.0
        material = abs(modes["X1"] - modes["X4"]) / best_ms * 100.0 > tie_threshold_percent
        rows.append({"workload_id": wid, "median_ms": modes, "best_requested_mode": best[1],
                     "default_ms": modes["DEFAULT"], "x1_regret_percent": x1_regret,
                     "x4_regret_percent": x4_regret, "material_x1_x4_difference": material})
    winners = {m: sum(1 for row in rows if row["best_requested_mode"] == m) for m in ("X1", "X4")}
    x1_static_max_regret = max(row["x1_regret_percent"] for row in rows)
    material_rows = [row for row in rows if row["material_x1_x4_difference"]]
    if failures or incomplete:
        verdict = "INVALID_MEASUREMENT"
    elif not material_rows:
        verdict = "XNNPACK_NO_MATERIAL_DECISION_SPACE"
    elif x1_static_max_regret <= tie_threshold_percent:
        verdict = "XNNPACK_ONE_STATIC_WINNER"
    elif 0 in winners.values():
        verdict = "XNNPACK_ONE_STATIC_WINNER"
    else:
        verdict = "XNNPACK_STABLE_MULTI_REGION_DECISION"
    out = {"schema": "e3_xnnpack_discovery_analysis", "records": len(raw["records"]), "correctness_failures": len(failures),
           "incomplete_timing_records": len(incomplete), "winner_counts": winners,
           "tie_threshold_percent": tie_threshold_percent, "x1_static_max_regret_percent": x1_static_max_regret,
           "material_decision_workloads": len(material_rows), "policy_recommendation": "static_X1" if x1_static_max_regret <= tie_threshold_percent else "needs_boundary",
           "per_workload": rows, "candidate_space_verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest"); p.add_argument("--p1d-manifest", required=True); p.add_argument("--pte-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--input-dir", required=True); p.add_argument("--warmups", type=int, default=5); p.add_argument("--repeats", type=int, default=20); p.add_argument("--sessions", type=int, default=3); p.add_argument("--affinity", default="0-3")
    r = sub.add_parser("run"); r.add_argument("--manifest", required=True); r.add_argument("--input-dir", required=True); r.add_argument("--runner", required=True); r.add_argument("--executor-runner", required=True); r.add_argument("--tmp-dir", required=True); r.add_argument("--out", required=True)
    a = sub.add_parser("analyze"); a.add_argument("--raw", required=True); a.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "manifest":
        man = make_manifest(args)
        Path(args.out).write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
        write_inputs(man, Path(args.input_dir))
        print(man["manifest_sha256"])
    elif args.cmd == "run":
        run_discovery(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
