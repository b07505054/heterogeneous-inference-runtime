#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


class FusedMatmulBiasRelu(torch.nn.Module):
    def forward(self, a, b, bias):
        return torch.relu(torch.matmul(a, b) + bias)


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def graph_text(obj):
    try:
        return str(obj.graph_module)
    except Exception as e:
        return f"<graph unavailable: {type(e).__name__}: {e}>"


def count_delegate_calls(text):
    return text.count("executorch_call_delegate")


def export_one(m, n, k, out_dir: Path, mode: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = FusedMatmulBiasRelu().eval()
    a = torch.randn(m, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    bias = torch.randn(n, dtype=torch.float32)
    exported = torch.export.export(model, (a, b, bias))
    exported_text = graph_text(exported)
    if mode == "xnnpack":
        edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    elif mode == "portable":
        edge = to_edge(exported)
    else:
        raise ValueError(mode)
    edge_text = graph_text(edge.exported_program())
    exec_prog = edge.to_executorch()
    pte = out_dir / f"fused_matmul_bias_relu_{m}x{n}x{k}_{mode}.pte"
    with pte.open("wb") as f:
        exec_prog.write_to_file(f)
    report = {
        "schema": "e1_executorch_export_report",
        "mode": mode,
        "shape": {"m": m, "n": n, "k": k},
        "dtype": "float32",
        "semantic": "Y = ReLU(A @ B + bias)",
        "pte_path": str(pte),
        "pte_sha256": sha256_file(pte),
        "pte_bytes": pte.stat().st_size,
        "exported_graph": exported_text,
        "edge_graph": edge_text,
        "delegate_call_count": count_delegate_calls(edge_text),
        "classification": "FULL_REGION_DELEGATED_FUSION_UNKNOWN" if mode == "xnnpack" and count_delegate_calls(edge_text) >= 1 else ("PORTABLE_ONLY" if mode == "portable" else "UNKNOWN"),
        "truth_boundary": "Delegate partition presence is verified from edge graph; internal XNNPACK fusion is not proven by this export script."
    }
    rp = out_dir / f"fused_matmul_bias_relu_{m}x{n}x{k}_{mode}_export_report.json"
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"mode": mode, "shape": [m,n,k], "pte": str(pte), "sha256": report["pte_sha256"], "delegate_call_count": report["delegate_call_count"], "classification": report["classification"]}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shape", action="append", required=True, help="MxNxK")
    ap.add_argument("--mode", choices=["portable", "xnnpack"], action="append", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    for s in args.shape:
        m,n,k = map(int, s.lower().split("x"))
        for mode in args.mode:
            export_one(m,n,k,out_dir,mode)

if __name__ == "__main__":
    main()
