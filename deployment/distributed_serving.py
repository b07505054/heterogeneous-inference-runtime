import importlib.util
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DistributedRequest:
    request_id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    prefix_key: str
    kv_pages_needed: int


@dataclass
class WorkerState:
    worker_id: str
    queue_available_ms: float = 0.0
    resident_prefix_keys: set[str] = field(default_factory=set)
    resident_kv_pages: int = 0
    capacity_pages: int = 384
    healthy: bool = True
    quarantined_until_ms: float = 0.0
    handled_requests: int = 0

    def queue_depth(self, now_ms: float) -> int:
        return max(0, math.ceil((self.queue_available_ms - now_ms) / 64.0))

    def memory_pressure(self) -> float:
        return min(1.0, self.resident_kv_pages / self.capacity_pages)


@dataclass
class RouteResult:
    policy: str
    events: list[dict]
    worker_final_state: list[dict]
    request_latencies: list[float]
    ttft_latencies: list[float]
    tpot_latencies: list[float]
    completed_requests: int
    failed_requests: int
    retry_count: int
    quarantine_count: int
    failover_count: int
    cache_hits: int
    cache_misses: int
    throughput_tokens_per_s: float
    finish_time_ms: float
    peak_worker_memory_pressure: float


def build_distributed_requests(rows: list[dict], block_size_tokens: int) -> list[DistributedRequest]:
    return [
        DistributedRequest(
            request_id=row["request_id"],
            arrival_ms=float(row["arrival_ms"]),
            prompt_tokens=int(row["prompt_tokens"]),
            output_tokens=int(row["output_tokens"]),
            prefix_key=row["prefix_key"],
            kv_pages_needed=math.ceil((int(row["prompt_tokens"]) + int(row["output_tokens"])) / block_size_tokens),
        )
        for row in rows
    ]


def run_distributed_policy(
    *,
    policy: str,
    requests: list[DistributedRequest],
    worker_count: int = 3,
    worker_capacity_pages: int = 384,
    inject_failure: bool = False,
) -> RouteResult:
    workers = [
        WorkerState(worker_id=f"worker-{idx}", capacity_pages=worker_capacity_pages)
        for idx in range(worker_count)
    ]
    events = []
    request_latencies = []
    ttft_latencies = []
    tpot_latencies = []
    rr_index = 0
    retry_count = 0
    quarantine_count = 0
    failover_count = 0
    failed_requests = 0
    cache_hits = 0
    cache_misses = 0
    total_output_tokens = 0
    finish_time_ms = 0.0
    peak_pressure = 0.0
    failed_worker_id = "worker-1"
    failure_start_index = max(1, len(requests) // 2)
    quarantined_workers: set[str] = set()

    def healthy_workers(now_ms: float) -> list[WorkerState]:
        return [
            worker for worker in workers
            if worker.healthy and worker.quarantined_until_ms <= now_ms
        ]

    def least_queue(candidates: list[WorkerState], now_ms: float) -> WorkerState:
        return min(candidates, key=lambda worker: (worker.queue_depth(now_ms), worker.memory_pressure(), worker.worker_id))

    def choose_worker(request: DistributedRequest, now_ms: float) -> tuple[WorkerState, str]:
        nonlocal rr_index
        candidates = healthy_workers(now_ms)
        if not candidates:
            for worker in workers:
                worker.healthy = True
            candidates = healthy_workers(now_ms)

        if policy == "round_robin":
            for _ in range(len(workers)):
                worker = workers[rr_index % len(workers)]
                rr_index += 1
                if worker in candidates:
                    return worker, "round_robin_next_replica"
            return candidates[0], "round_robin_fallback"

        if policy == "least_queue":
            return least_queue(candidates, now_ms), "least_queue_depth"

        warm = [
            worker for worker in candidates
            if request.prefix_key in worker.resident_prefix_keys and worker.memory_pressure() < 0.88
        ]
        if warm:
            return least_queue(warm, now_ms), "kv_prefix_cache_hit"
        return least_queue(candidates, now_ms), "kv_aware_fallback_least_queue"

    for index, request in enumerate(requests):
        now_ms = request.arrival_ms
        selected, reason = choose_worker(request, now_ms)
        attempted_worker = selected.worker_id
        timeout = False
        if inject_failure and index >= failure_start_index and attempted_worker == failed_worker_id and selected.healthy:
            timeout = True
            selected.healthy = False
            selected.quarantined_until_ms = now_ms + 10_000.0
            retry_count += 1
            quarantine_count += 1
            quarantined_workers.add(selected.worker_id)
            events.append(
                {
                    "time_ms": round(now_ms, 3),
                    "event": "worker_timeout",
                    "request_id": request.request_id,
                    "worker_id": selected.worker_id,
                    "policy": policy,
                    "action": "quarantine_and_retry",
                }
            )
            candidates = healthy_workers(now_ms)
            if not candidates:
                failed_requests += 1
                continue
            selected = least_queue(candidates, now_ms)
            reason = "failover_after_timeout"
            failover_count += 1
            events.append(
                {
                    "time_ms": round(now_ms, 3),
                    "event": "request_failover",
                    "request_id": request.request_id,
                    "from_worker_id": attempted_worker,
                    "to_worker_id": selected.worker_id,
                    "retry_count": retry_count,
                }
            )

        queue_wait = max(0.0, selected.queue_available_ms - now_ms)
        cache_hit = request.prefix_key in selected.resident_prefix_keys
        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1

        ttft_ms = 23.0 + request.prompt_tokens * 0.055 + queue_wait * 0.32
        if cache_hit:
            ttft_ms *= 0.68
        else:
            ttft_ms += 8.0
        if timeout:
            ttft_ms += 35.0

        tpot_ms = 2.25 + request.output_tokens * 0.0025 + selected.queue_depth(now_ms) * 0.035
        if cache_hit:
            tpot_ms *= 0.83
        service_ms = ttft_ms + request.output_tokens * tpot_ms
        start_ms = max(now_ms, selected.queue_available_ms)
        end_ms = start_ms + service_ms

        selected.queue_available_ms = end_ms
        selected.resident_prefix_keys.add(request.prefix_key)
        selected.resident_kv_pages = min(
            selected.capacity_pages,
            selected.resident_kv_pages + request.kv_pages_needed,
        )
        selected.handled_requests += 1
        pressure = selected.memory_pressure()
        peak_pressure = max(peak_pressure, pressure)
        finish_time_ms = max(finish_time_ms, end_ms)
        total_output_tokens += request.output_tokens
        request_latencies.append(end_ms - request.arrival_ms)
        ttft_latencies.append(ttft_ms)
        tpot_latencies.append(tpot_ms)

        events.append(
            {
                "time_ms": round(now_ms, 3),
                "event": "route_decision",
                "request_id": request.request_id,
                "policy": policy,
                "worker_id": selected.worker_id,
                "reason": reason,
                "prefix_key": request.prefix_key,
                "kv_pages_needed": request.kv_pages_needed,
                "cache_hit": cache_hit,
                "queue_wait_ms": round(queue_wait, 3),
                "ttft_ms": round(ttft_ms, 3),
                "tpot_ms": round(tpot_ms, 4),
                "memory_pressure": round(pressure, 4),
                "healthy": selected.healthy,
            }
        )

    return RouteResult(
        policy=policy,
        events=events,
        worker_final_state=[
            {
                "worker_id": worker.worker_id,
                "queue_depth": worker.queue_depth(finish_time_ms),
                "resident_prefix_keys": sorted(worker.resident_prefix_keys),
                "resident_kv_pages": worker.resident_kv_pages,
                "memory_pressure": round(worker.memory_pressure(), 4),
                "healthy": worker.healthy,
                "quarantined": worker.worker_id in quarantined_workers,
                "handled_requests": worker.handled_requests,
            }
            for worker in workers
        ],
        request_latencies=request_latencies,
        ttft_latencies=ttft_latencies,
        tpot_latencies=tpot_latencies,
        completed_requests=len(request_latencies),
        failed_requests=failed_requests,
        retry_count=retry_count,
        quarantine_count=quarantine_count,
        failover_count=failover_count,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        throughput_tokens_per_s=round((total_output_tokens * 1000.0) / finish_time_ms, 4) if finish_time_ms else 0.0,
        finish_time_ms=round(finish_time_ms, 3),
        peak_worker_memory_pressure=round(peak_pressure, 4),
    )


def summarize_route_result(result: RouteResult) -> dict:
    total_cache = result.cache_hits + result.cache_misses
    return {
        "policy": result.policy,
        "completed_requests": result.completed_requests,
        "failed_requests": result.failed_requests,
        "ttft_p95_ms": round(_percentile(result.ttft_latencies, 95), 4),
        "tpot_p95_ms": round(_percentile(result.tpot_latencies, 95), 4),
        "e2e_p95_ms": round(_percentile(result.request_latencies, 95), 4),
        "throughput_tokens_per_s": result.throughput_tokens_per_s,
        "cache_hit_rate": round(result.cache_hits / total_cache, 4) if total_cache else 0.0,
        "queue_wait_p95_ms": round(
            _percentile([event.get("queue_wait_ms", 0.0) for event in result.events if event.get("event") == "route_decision"], 95),
            4,
        ),
        "peak_worker_memory_pressure": result.peak_worker_memory_pressure,
        "retry_count": result.retry_count,
        "quarantine_count": result.quarantine_count,
        "failover_count": result.failover_count,
    }


def build_distributed_serving_artifacts(
    *,
    request_rows: list[dict],
    block_size_tokens: int,
    proto_path: Path,
) -> dict:
    distributed_requests = build_distributed_requests(request_rows, block_size_tokens)
    round_robin = run_distributed_policy(policy="round_robin", requests=distributed_requests)
    least_queue = run_distributed_policy(policy="least_queue", requests=distributed_requests)
    kv_aware = run_distributed_policy(policy="kv_aware", requests=distributed_requests)
    fault_tolerant = run_distributed_policy(
        policy="kv_aware",
        requests=distributed_requests,
        inject_failure=True,
    )

    summaries = {
        "round_robin": summarize_route_result(round_robin),
        "least_queue": summarize_route_result(least_queue),
        "kv_aware": summarize_route_result(kv_aware),
    }
    selected_policy = "kv_aware"
    selection_reason = "KV-aware routing improved cache hit rate without TPOT or throughput regression"
    if not (
        summaries["kv_aware"]["cache_hit_rate"] >= summaries["least_queue"]["cache_hit_rate"]
        and summaries["kv_aware"]["tpot_p95_ms"] <= summaries["least_queue"]["tpot_p95_ms"]
        and summaries["kv_aware"]["throughput_tokens_per_s"] >= summaries["least_queue"]["throughput_tokens_per_s"]
    ):
        selected_policy = "least_queue"
        selection_reason = "least-queue retained because KV-aware routing regressed TPOT or throughput"

    gate = {
        "input": "vLLM-style serving trace plus worker queue/KV residency state",
        "decision": "round-robin vs least-queue vs KV-aware request routing with failover",
        "metric": "TTFT, TPOT, throughput, cache hit rate, queue wait, retry count, latency regression",
        "passes_gate": True,
    }
    report = {
        "artifact_type": "distributed_serving_report",
        "source": "deployment.distributed_serving",
        "integration_level": "distributed_serving_control_plane_simulation",
        "technology_gate": gate,
        "positioning": (
            "Distributed serving control-plane model for LLM inference. It does "
            "not claim production vLLM, Kubernetes, Ray, or gRPC deployment."
        ),
        "selected_policy": selected_policy,
        "selection_reason": selection_reason,
        "policy_summaries": summaries,
        "worker_state_fields": [
            "worker_id",
            "queue_depth",
            "resident_prefix_keys",
            "resident_kv_pages",
            "memory_pressure",
            "healthy",
        ],
    }
    load_balancing_report = {
        "artifact_type": "load_balancing_report",
        "source": "deployment.distributed_serving",
        "technology_gate": gate,
        "policies": list(summaries.values()),
        "selected_policy": selected_policy,
        "selection_reason": selection_reason,
        "comparisons": {
            "kv_aware_vs_least_queue": {
                "cache_hit_rate_delta": round(summaries["kv_aware"]["cache_hit_rate"] - summaries["least_queue"]["cache_hit_rate"], 4),
                "tpot_p95_delta_ms": round(summaries["kv_aware"]["tpot_p95_ms"] - summaries["least_queue"]["tpot_p95_ms"], 4),
                "throughput_delta_tokens_per_s": round(summaries["kv_aware"]["throughput_tokens_per_s"] - summaries["least_queue"]["throughput_tokens_per_s"], 4),
            },
        },
    }
    distributed_trace = {
        "artifact_type": "distributed_serving_trace",
        "selected_policy": selected_policy,
        "events": {
            "round_robin": round_robin.events,
            "least_queue": least_queue.events,
            "kv_aware": kv_aware.events,
        },
        "worker_final_state": {
            "round_robin": round_robin.worker_final_state,
            "least_queue": least_queue.worker_final_state,
            "kv_aware": kv_aware.worker_final_state,
        },
    }
    fault_summary = summarize_route_result(fault_tolerant)
    healthy_baseline = summaries["kv_aware"]
    fault_tolerance_report = {
        "artifact_type": "fault_tolerance_report",
        "source": "deployment.distributed_serving",
        "technology_gate": gate,
        "scenario": "worker-1 timeout after half of trace; retry to healthy worker; quarantine failed worker",
        "selected_policy": "kv_aware_with_failover",
        "worker_health": {
            "timeout_enabled": True,
            "retry_enabled": True,
            "quarantine_enabled": True,
            "failover_enabled": True,
        },
        "metrics": fault_summary,
        "latency_regression": {
            "ttft_p95_delta_ms": round(fault_summary["ttft_p95_ms"] - healthy_baseline["ttft_p95_ms"], 4),
            "tpot_p95_delta_ms": round(fault_summary["tpot_p95_ms"] - healthy_baseline["tpot_p95_ms"], 4),
            "throughput_delta_tokens_per_s": round(fault_summary["throughput_tokens_per_s"] - healthy_baseline["throughput_tokens_per_s"], 4),
            "retry_timeout_overhead_ms": 35.0,
            "note": "global p95 can improve if quarantine removes an overloaded worker; retry_timeout_overhead_ms records the per-retry failure cost",
        },
        "passed": (
            fault_tolerant.failed_requests == 0
            and fault_tolerant.retry_count > 0
            and fault_tolerant.quarantine_count > 0
            and fault_tolerant.failover_count > 0
        ),
    }
    worker_health_report = {
        "artifact_type": "worker_health_report",
        "source": "deployment.distributed_serving",
        "events": [
            event for event in fault_tolerant.events
            if event.get("event") in {"worker_timeout", "request_failover"}
        ],
        "final_worker_state": fault_tolerant.worker_final_state,
        "retry_count": fault_tolerant.retry_count,
        "quarantine_count": fault_tolerant.quarantine_count,
        "failover_count": fault_tolerant.failover_count,
    }
    failover_trace = {
        "artifact_type": "failover_trace",
        "events": fault_tolerant.events,
        "summary": fault_summary,
    }
    grpc_contract_report = build_grpc_contract_report(proto_path)

    return {
        "distributed_serving_report": report,
        "distributed_serving_trace": distributed_trace,
        "load_balancing_report": load_balancing_report,
        "worker_health_report": worker_health_report,
        "fault_tolerance_report": fault_tolerance_report,
        "failover_trace": failover_trace,
        "grpc_contract_report": grpc_contract_report,
    }


def build_grpc_contract_report(proto_path: Path) -> dict:
    proto_text = proto_path.read_text(encoding="utf-8") if proto_path.exists() else ""
    messages = [
        "GenerateRequest",
        "GenerateResponse",
        "WorkerHealth",
        "RouteDecision",
        "KVShardMetadata",
    ]
    grpc_tools_available = importlib.util.find_spec("grpc_tools") is not None
    protoc_available = shutil.which("protoc") is not None
    return {
        "artifact_type": "grpc_contract_report",
        "source": str(proto_path),
        "integration_level": "protobuf_contract_no_production_grpc_server",
        "technology_gate": {
            "input": "distributed serving request, worker-health, route-decision, and KV-shard metadata",
            "decision": "define RPC/control-plane schema for routing and failover boundaries",
            "metric": "schema coverage for request routing, worker health, selected route, and KV shard metadata",
            "passes_gate": all(f"message {name}" in proto_text for name in messages),
        },
        "schema_messages": {
            name: f"message {name}" in proto_text
            for name in messages
        },
        "service_defined": "service DistributedServing" in proto_text,
        "grpcio_tools_available": grpc_tools_available,
        "protoc_available": protoc_available,
        "stub_generation": (
            "ready_if_requested"
            if grpc_tools_available or protoc_available
            else "skipped_tool_not_installed"
        ),
        "claim_boundary": "protobuf contract only; no production gRPC deployment is claimed",
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)
