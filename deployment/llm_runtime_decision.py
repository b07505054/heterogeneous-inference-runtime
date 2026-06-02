import math
import random
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    arrival_ms: float


@dataclass(frozen=True)
class AdmissionDecision:
    action: str
    reason: str
    needed_blocks: int
    available_blocks: int
    pressure: float
    pressure_level: str
    effective_batch_cap: int


@dataclass
class CostModel:
    prefill_base_ms: float = 24.0
    prefill_ms_per_token: float = 0.155
    decode_base_ms: float = 2.15
    decode_ms_per_batch_item: float = 0.16
    decode_ms_per_1k_context: float = 0.55
    kv_update_base_ms: float = 0.18
    kv_update_ms_per_block: float = 0.011
    observed_decode_scale: float = 1.0
    observed_prefill_scale: float = 1.0

    def predict_prefill_ms(self, prompt_tokens: int) -> float:
        return (self.prefill_base_ms + prompt_tokens * self.prefill_ms_per_token) * self.observed_prefill_scale

    def predict_decode_step_ms(self, batch_size: int, avg_context_tokens: float) -> float:
        return (
            self.decode_base_ms
            + max(0, batch_size - 1) * self.decode_ms_per_batch_item
            + (avg_context_tokens / 1000.0) * self.decode_ms_per_1k_context
        ) * self.observed_decode_scale

    def predict_kv_update_ms(self, blocks: int) -> float:
        return self.kv_update_base_ms + blocks * self.kv_update_ms_per_block

    def calibrate(self, observed_prefill_ms: Iterable[float], observed_decode_ms: Iterable[float]) -> dict:
        prefill = list(observed_prefill_ms)
        decode = list(observed_decode_ms)
        report = {
            "prefill_scale_before": round(self.observed_prefill_scale, 4),
            "decode_scale_before": round(self.observed_decode_scale, 4),
        }
        if prefill:
            predicted = [self.predict_prefill_ms(512) for _ in prefill]
            avg_pred = sum(predicted) / len(predicted)
            avg_obs = sum(prefill) / len(prefill)
            if avg_pred > 0:
                self.observed_prefill_scale *= avg_obs / avg_pred
        if decode:
            avg_pred = sum(self.predict_decode_step_ms(1, 512) for _ in decode) / len(decode)
            avg_obs = sum(decode) / len(decode)
            if avg_pred > 0:
                self.observed_decode_scale *= avg_obs / avg_pred
        report.update(
            {
                "prefill_scale_after": round(self.observed_prefill_scale, 4),
                "decode_scale_after": round(self.observed_decode_scale, 4),
                "observed_prefill_samples": len(prefill),
                "observed_decode_samples": len(decode),
            }
        )
        return report


@dataclass
class MemoryPlanner:
    total_blocks: int
    block_size_tokens: int
    kv_mb_per_block: float
    free_blocks: list[int] = field(init=False)
    allocations: dict[str, list[int]] = field(default_factory=dict)
    peak_blocks: int = 0
    evictions: int = 0
    rejected: int = 0
    delayed: int = 0

    def __post_init__(self) -> None:
        self.free_blocks = list(range(self.total_blocks))

    def needed_blocks(self, request: Request) -> int:
        tokens = request.prompt_tokens + request.output_tokens
        return math.ceil(tokens / self.block_size_tokens)

    def can_allocate(self, request: Request) -> bool:
        return len(self.free_blocks) >= self.needed_blocks(request)

    def pressure(self) -> float:
        return 1.0 - (len(self.free_blocks) / self.total_blocks)

    def pressure_level(self) -> str:
        return self.pressure_level_for(self.pressure())

    def pressure_level_for(self, pressure: float) -> str:
        if pressure >= 0.90:
            return "critical"
        if pressure >= 0.75:
            return "high"
        if pressure >= 0.55:
            return "medium"
        return "low"

    def effective_batch_cap(self, max_decode_batch_size: int) -> int:
        return self.batch_cap_for_level(self.pressure_level(), max_decode_batch_size)

    def batch_cap_for_level(self, level: str, max_decode_batch_size: int) -> int:
        if level == "critical":
            return 1
        if level == "high":
            return max(1, min(max_decode_batch_size, 2))
        if level == "medium":
            return max(1, min(max_decode_batch_size, 4))
        return max_decode_batch_size

    def projected_pressure(self, additional_blocks: int) -> float:
        projected_free = max(0, len(self.free_blocks) - additional_blocks)
        return 1.0 - (projected_free / self.total_blocks)

    def admission_decision(
        self,
        request: Request,
        *,
        max_decode_batch_size: int,
        allow_delay: bool,
    ) -> AdmissionDecision:
        needed = self.needed_blocks(request)
        available = len(self.free_blocks)
        pressure = self.pressure()
        level = self.pressure_level()
        batch_cap = self.effective_batch_cap(max_decode_batch_size)

        if needed > available:
            return AdmissionDecision(
                action="reject",
                reason="insufficient_kv_blocks",
                needed_blocks=needed,
                available_blocks=available,
                pressure=pressure,
                pressure_level=level,
                effective_batch_cap=batch_cap,
            )

        if allow_delay and level == "critical" and request.prompt_tokens >= self.block_size_tokens * 48:
            self.delayed += 1
            return AdmissionDecision(
                action="delay",
                reason="critical_pressure_long_context",
                needed_blocks=needed,
                available_blocks=available,
                pressure=pressure,
                pressure_level=level,
                effective_batch_cap=batch_cap,
            )

        return AdmissionDecision(
            action="admit",
            reason="fits_kv_budget",
            needed_blocks=needed,
            available_blocks=available,
            pressure=pressure,
            pressure_level=level,
            effective_batch_cap=batch_cap,
        )

    def allocate(self, request: Request) -> list[int]:
        needed = self.needed_blocks(request)
        if len(self.free_blocks) < needed:
            self.rejected += 1
            raise MemoryError("insufficient KV blocks")
        blocks = [self.free_blocks.pop(0) for _ in range(needed)]
        self.allocations[request.request_id] = blocks
        self.peak_blocks = max(self.peak_blocks, self.used_blocks())
        return blocks

    def free(self, request_id: str) -> list[int]:
        blocks = self.allocations.pop(request_id, [])
        self.free_blocks.extend(blocks)
        self.free_blocks.sort()
        return blocks

    def used_blocks(self) -> int:
        return sum(len(blocks) for blocks in self.allocations.values())

    def peak_kv_mb(self) -> float:
        return round(self.peak_blocks * self.kv_mb_per_block, 3)


@dataclass
class SchedulerResult:
    policy: str
    scheduler_steps: list[dict]
    serving_events: list[dict]
    backend_placements: list[dict]
    kv_requests: list[dict]
    request_latencies: list[float]
    decode_step_latencies: list[float]
    prefill_latencies: list[float]
    completed_requests: int
    rejected_requests: int
    delayed_requests: int
    oom_events: int
    peak_allocated_blocks: int
    peak_kv_cache_mb: float
    avg_decode_batch_size: float
    decode_batch_efficiency: float
    tokens_per_second: float
    finish_time_ms: float
    pressure_limited_candidates: int


class RuntimeScheduler:
    def __init__(
        self,
        *,
        policy: str,
        cost_model: CostModel,
        memory: MemoryPlanner,
        rng: random.Random,
        max_decode_batch_size: int = 8,
    ) -> None:
        self.policy = policy
        self.cost_model = cost_model
        self.memory = memory
        self.rng = rng
        self.max_decode_batch_size = max_decode_batch_size
        self.pressure_limited_candidates = 0

    def run(self, requests: list[Request]) -> SchedulerResult:
        time_ms = 0.0
        scheduler_steps = []
        serving_events = []
        backend_placements = []
        kv_requests = []
        request_latencies = []
        decode_step_latencies = []
        prefill_latencies = []
        decode_batch_sizes = []
        completed_output_tokens = 0
        rejected = 0
        delayed = 0
        oom = 0
        delayed_once: set[str] = set()

        pending = sorted(requests, key=lambda req: req.arrival_ms)

        while pending:
            req = pending.pop(0)
            time_ms = max(time_ms, req.arrival_ms)
            queue_wait_ms = round(max(0.0, time_ms - req.arrival_ms), 3)
            decision = self.memory.admission_decision(
                req,
                max_decode_batch_size=self.max_decode_batch_size,
                allow_delay=self.policy == "cost_aware_memory_pressure",
            )
            serving_events.append(
                {
                    "time_ms": round(time_ms, 3),
                    "event": (
                        "request_admitted"
                        if decision.action == "admit"
                        else f"request_{decision.action}ed"
                    ),
                    "request_id": req.request_id,
                    "prompt_tokens": req.prompt_tokens,
                    "generated_tokens_target": req.output_tokens,
                    "queue_wait_ms": queue_wait_ms,
                    "needed_blocks": decision.needed_blocks,
                    "available_blocks": decision.available_blocks,
                    "memory_pressure": round(decision.pressure, 4),
                    "pressure_level": decision.pressure_level,
                    "admission_action": decision.action,
                    "admission_reason": decision.reason,
                    "effective_batch_cap": decision.effective_batch_cap,
                }
            )
            if decision.action == "delay" and req.request_id not in delayed_once:
                delayed += 1
                delayed_once.add(req.request_id)
                pending.append(
                    Request(
                        request_id=req.request_id,
                        prompt_tokens=req.prompt_tokens,
                        output_tokens=req.output_tokens,
                        arrival_ms=round(time_ms + 32.0, 3),
                    )
                )
                pending.sort(key=lambda item: item.arrival_ms)
                continue

            if decision.action != "admit":
                rejected += 1
                oom += 1
                continue

            allocated = self.memory.allocate(req)
            kv_mb = round(len(allocated) * self.memory.kv_mb_per_block, 3)
            kv_requests.append(
                {
                    "request_id": req.request_id,
                    "allocated_blocks": allocated,
                    "context_tokens": req.prompt_tokens,
                    "generated_tokens": req.output_tokens,
                    "kv_cache_mb": kv_mb,
                }
            )

            prefill_ms = self._observed_prefill(req.prompt_tokens)
            prefill_latencies.append(prefill_ms)
            scheduler_steps.append(
                {"time_ms": round(time_ms, 3), "event": "prefill_start", "request_id": req.request_id, "batch_size": 1}
            )
            serving_events.append(
                {
                    "time_ms": round(time_ms, 3),
                    "event": "prefill_start",
                    "request_id": req.request_id,
                    "backend": "gpu",
                    "allocated_blocks": allocated,
                }
            )
            backend_placements.append(
                {
                    "request_id": req.request_id,
                    "op": "attention_prefill",
                    "backend": "gpu",
                    "latency_ms": round(prefill_ms * 0.41, 3),
                }
            )
            backend_placements.append(
                {
                    "request_id": req.request_id,
                    "op": "kv_cache_update",
                    "backend": "cpu",
                    "latency_ms": round(self.cost_model.predict_kv_update_ms(len(allocated)), 3),
                }
            )
            time_ms += prefill_ms
            scheduler_steps.append(
                {"time_ms": round(time_ms, 3), "event": "prefill_end", "request_id": req.request_id, "batch_size": 1}
            )
            serving_events.append(
                {
                    "time_ms": round(time_ms, 3),
                    "event": "prefill_end",
                    "request_id": req.request_id,
                    "prefill_latency_ms": prefill_ms,
                }
            )

            decode_batch = self._choose_decode_batch(req, pending, time_ms)
            time_ms = max(time_ms, max(item.arrival_ms for item in decode_batch))
            batch_size = len(decode_batch)
            batch_queue_waits = {
                item.request_id: round(max(0.0, time_ms - item.arrival_ms), 3)
                for item in decode_batch
            }
            for item in decode_batch:
                if item.request_id == req.request_id:
                    continue
                item_blocks = self.memory.allocations.get(item.request_id, [])
                serving_events.append(
                    {
                        "time_ms": round(time_ms, 3),
                        "event": "request_admitted",
                        "request_id": item.request_id,
                        "prompt_tokens": item.prompt_tokens,
                        "generated_tokens_target": item.output_tokens,
                        "queue_wait_ms": batch_queue_waits[item.request_id],
                        "needed_blocks": self.memory.needed_blocks(item),
                        "memory_pressure": round(self.memory.pressure(), 4),
                        "pressure_level": self.memory.pressure_level(),
                        "admission_action": "admit",
                        "admission_reason": "batched_with_decode_peer",
                        "effective_batch_cap": self.memory.effective_batch_cap(self.max_decode_batch_size),
                    }
                )
                kv_requests.append(
                    {
                        "request_id": item.request_id,
                        "allocated_blocks": item_blocks,
                        "context_tokens": item.prompt_tokens,
                        "generated_tokens": item.output_tokens,
                        "kv_cache_mb": round(len(item_blocks) * self.memory.kv_mb_per_block, 3),
                    }
                )
            decode_batch_sizes.append(batch_size)
            scheduler_steps.append(
                {
                    "time_ms": round(time_ms, 3),
                    "event": "decode_batch",
                    "active_requests": [item.request_id for item in decode_batch],
                    "batch_size": batch_size,
                    "memory_pressure": round(self.memory.pressure(), 4),
                    "pressure_level": self.memory.pressure_level(),
                    "effective_batch_cap": self.memory.effective_batch_cap(self.max_decode_batch_size),
                }
            )
            decode_total = 0.0
            max_steps = max(item.output_tokens for item in decode_batch)
            avg_context = sum(item.prompt_tokens for item in decode_batch) / batch_size
            for step in range(max_steps):
                step_active = [item for item in decode_batch if step < item.output_tokens]
                step_ms = self._observed_decode(len(step_active), avg_context)
                decode_step_latencies.append(step_ms)
                decode_total += step_ms
                if step % 16 == 0:
                    serving_events.append(
                        {
                            "time_ms": round(time_ms, 3),
                            "event": "decode_step",
                            "request_id": req.request_id,
                            "step": step,
                            "active_requests": [item.request_id for item in step_active],
                            "batch_size": len(step_active),
                            "backend": "gpu",
                        }
                    )
                time_ms += step_ms

            for done in decode_batch:
                latency_ms = round(batch_queue_waits[done.request_id] + prefill_ms + decode_total, 3)
                request_latencies.append(latency_ms)
                completed_output_tokens += done.output_tokens
                serving_events.append(
                    {
                        "time_ms": round(time_ms, 3),
                        "event": "tokens_generated",
                        "request_id": done.request_id,
                        "tokens_generated": done.output_tokens,
                        "decode_latency_ms": round(decode_total, 3),
                    }
                )
                freed = self.memory.free(done.request_id)
                serving_events.append(
                    {
                        "time_ms": round(time_ms, 3),
                        "event": "kv_blocks_freed",
                        "request_id": done.request_id,
                        "freed_blocks": freed,
                    }
                )

        avg_batch = sum(decode_batch_sizes) / len(decode_batch_sizes) if decode_batch_sizes else 0.0
        return SchedulerResult(
            policy=self.policy,
            scheduler_steps=scheduler_steps,
            serving_events=serving_events,
            backend_placements=backend_placements,
            kv_requests=kv_requests,
            request_latencies=request_latencies,
            decode_step_latencies=decode_step_latencies,
            prefill_latencies=prefill_latencies,
            completed_requests=len(request_latencies),
            rejected_requests=rejected,
            delayed_requests=delayed,
            oom_events=oom,
            peak_allocated_blocks=self.memory.peak_blocks,
            peak_kv_cache_mb=self.memory.peak_kv_mb(),
            avg_decode_batch_size=round(avg_batch, 4),
            decode_batch_efficiency=round(avg_batch / self.max_decode_batch_size, 4) if self.max_decode_batch_size else 0.0,
            tokens_per_second=round((completed_output_tokens * 1000.0) / time_ms, 3) if time_ms else 0.0,
            finish_time_ms=round(time_ms, 3),
            pressure_limited_candidates=self.pressure_limited_candidates,
        )

    def _choose_decode_batch(self, req: Request, pending: list[Request], current_time_ms: float) -> list[Request]:
        if self.policy == "fcfs_fixed_batch":
            return [req]

        candidates = [req]
        remaining = []
        budget_blocks = max(0, int(self.memory.total_blocks * 0.82) - self.memory.used_blocks())
        for item in pending:
            inside_lookahead = item.arrival_ms <= current_time_ms + 48.0
            needed = self.memory.needed_blocks(item)
            projected_pressure = self.memory.projected_pressure(needed)
            projected_level = self.memory.pressure_level_for(projected_pressure)
            effective_cap = self.memory.batch_cap_for_level(
                projected_level,
                self.max_decode_batch_size,
            )
            if len(candidates) >= effective_cap:
                if projected_level in {"medium", "high", "critical"}:
                    self.pressure_limited_candidates += 1
                remaining.append(item)
                continue
            similar = abs(item.prompt_tokens - req.prompt_tokens) <= 512
            decision = self.memory.admission_decision(
                item,
                max_decode_batch_size=self.max_decode_batch_size,
                allow_delay=False,
            )
            if (
                inside_lookahead
                and similar
                and needed <= budget_blocks
                and decision.action == "admit"
            ):
                try:
                    self.memory.allocate(item)
                    candidates.append(item)
                    budget_blocks -= needed
                except MemoryError:
                    remaining.append(item)
            else:
                remaining.append(item)
        pending[:] = remaining
        return candidates

    def _observed_prefill(self, prompt_tokens: int) -> float:
        predicted = self.cost_model.predict_prefill_ms(prompt_tokens)
        return round(predicted * self.rng.uniform(0.92, 1.08), 3)

    def _observed_decode(self, batch_size: int, avg_context_tokens: float) -> float:
        predicted = self.cost_model.predict_decode_step_ms(batch_size, avg_context_tokens)
        return round(predicted * self.rng.uniform(0.86, 1.14), 3)


def build_requests(count: int, rng: random.Random) -> list[Request]:
    requests = []
    arrival_ms = 0.0
    for idx in range(count):
        arrival_ms += rng.uniform(0.0, 16.0)
        requests.append(
            Request(
                request_id=f"req-{idx + 1:03d}",
                prompt_tokens=rng.choice([64, 128, 256, 512, 1024]),
                output_tokens=rng.choice([32, 64, 96, 128]),
                arrival_ms=round(arrival_ms, 3),
            )
        )
    return requests


def summarize_policy(result: SchedulerResult) -> dict:
    pressure_counts = {}
    cap_reductions = 0
    for step in result.scheduler_steps:
        level = step.get("pressure_level")
        if level:
            pressure_counts[level] = pressure_counts.get(level, 0) + 1
        if step.get("effective_batch_cap", 8) < 8:
            cap_reductions += 1

    return {
        "policy": result.policy,
        "completed_requests": result.completed_requests,
        "rejected_requests": result.rejected_requests,
        "delayed_requests": result.delayed_requests,
        "p95_latency_ms": round(_percentile(result.request_latencies, 95), 3),
        "tokens_per_second": result.tokens_per_second,
        "peak_kv_cache_mb": result.peak_kv_cache_mb,
        "avg_decode_batch_size": result.avg_decode_batch_size,
        "decode_batch_efficiency": result.decode_batch_efficiency,
        "pressure_level_counts": pressure_counts,
        "batch_cap_reductions": cap_reductions,
        "pressure_limited_candidates": result.pressure_limited_candidates,
        "finish_time_ms": result.finish_time_ms,
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
