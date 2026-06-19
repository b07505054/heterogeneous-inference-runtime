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


@dataclass
class RuntimeRequestState:
    request: Request
    state: str = "waiting"
    prompt_tokens_total: int = 0
    prompt_tokens_prefilled: int = 0
    output_tokens_generated: int = 0
    kv_pages: list[int] = field(default_factory=list)
    ttft_step: int | None = None
    finish_step: int | None = None
    admitted_step: int | None = None
    rejected_reason: str | None = None

    def __post_init__(self) -> None:
        if self.prompt_tokens_total == 0:
            self.prompt_tokens_total = self.request.prompt_tokens


@dataclass
class KVPage:
    page_id: int
    owner_request_id: str
    token_begin: int
    token_end: int
    state: str
    last_access_step: int


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
class PagePrefetchPlanner:
    lookahead_steps: int = 1
    max_prefetch_blocks_per_step: int = 12
    pressure_disable_threshold: float = 0.82
    hit_latency_saving_ms: float = 0.075
    wasted_prefetch_penalty_ms: float = 0.015
    warmed_pages: set[tuple[str, int]] = field(default_factory=set)
    attempts: int = 0
    hits: int = 0
    misses: int = 0
    wasted: int = 0
    pressure_skips: int = 0
    events: list[dict] = field(default_factory=list)

    def pages_for_step(
        self,
        request: Request,
        blocks: list[int],
        step: int,
        block_size_tokens: int,
    ) -> list[int]:
        if not blocks:
            return []
        context_tokens = request.prompt_tokens + max(0, step)
        tail_page = min(len(blocks) - 1, context_tokens // block_size_tokens)
        pages = {blocks[tail_page]}
        if tail_page > 0:
            pages.add(blocks[tail_page - 1])
        return sorted(pages)

    def consume(
        self,
        *,
        active_requests: list[Request],
        allocations: dict[str, list[int]],
        step: int,
        block_size_tokens: int,
    ) -> dict:
        pages = []
        hits = 0
        misses = 0
        for request in active_requests:
            for block in self.pages_for_step(
                request,
                allocations.get(request.request_id, []),
                step,
                block_size_tokens,
            ):
                key = (request.request_id, block)
                pages.append({"request_id": request.request_id, "block": block})
                if key in self.warmed_pages:
                    hits += 1
                    self.warmed_pages.remove(key)
                else:
                    misses += 1
        self.hits += hits
        self.misses += misses
        return {
            "step": step,
            "pages": pages,
            "hits": hits,
            "misses": misses,
            "latency_saving_ms": round(hits * self.hit_latency_saving_ms, 4),
        }

    def plan(
        self,
        *,
        active_requests: list[Request],
        allocations: dict[str, list[int]],
        step: int,
        block_size_tokens: int,
        pressure: float,
    ) -> dict:
        if pressure >= self.pressure_disable_threshold:
            self.pressure_skips += 1
            event = {
                "event": "page_prefetch_skipped",
                "reason": "memory_pressure_above_prefetch_budget",
                "memory_pressure": round(pressure, 4),
                "pressure_disable_threshold": self.pressure_disable_threshold,
                "active_requests": [request.request_id for request in active_requests],
                "step": step,
            }
            self.events.append(event)
            return event

        planned = []
        for request in active_requests:
            for block in self.pages_for_step(
                request,
                allocations.get(request.request_id, []),
                step + self.lookahead_steps,
                block_size_tokens,
            ):
                if len(planned) >= self.max_prefetch_blocks_per_step:
                    break
                key = (request.request_id, block)
                if key in self.warmed_pages:
                    continue
                self.warmed_pages.add(key)
                planned.append({"request_id": request.request_id, "block": block})
            if len(planned) >= self.max_prefetch_blocks_per_step:
                break

        self.attempts += len(planned)
        event = {
            "event": "page_prefetch",
            "reason": "prefetch_next_decode_kv_pages",
            "memory_pressure": round(pressure, 4),
            "active_requests": [request.request_id for request in active_requests],
            "step": step,
            "lookahead_steps": self.lookahead_steps,
            "prefetched_pages": planned,
            "prefetched_blocks": len(planned),
        }
        self.events.append(event)
        return event

    def release_request(self, request_id: str) -> None:
        before = len(self.warmed_pages)
        self.warmed_pages = {
            key for key in self.warmed_pages
            if key[0] != request_id
        }
        self.wasted += before - len(self.warmed_pages)

    def summary(self) -> dict:
        total_accesses = self.hits + self.misses
        return {
            "enabled": True,
            "policy": "vllm_style_allocated_kv_page_prefetch",
            "attempts": self.attempts,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total_accesses, 4) if total_accesses else 0.0,
            "wasted_prefetch_blocks": self.wasted,
            "pressure_skips": self.pressure_skips,
            "max_prefetch_blocks_per_step": self.max_prefetch_blocks_per_step,
            "pressure_disable_threshold": self.pressure_disable_threshold,
            "hit_latency_saving_ms": self.hit_latency_saving_ms,
            "wasted_prefetch_penalty_ms": self.wasted_prefetch_penalty_ms,
        }


@dataclass
class PagedKVLifecycle:
    total_pages: int
    page_size_tokens: int
    kv_mb_per_page: float
    free_pages: list[int] = field(init=False)
    pages: dict[int, KVPage] = field(default_factory=dict)
    request_pages: dict[str, list[int]] = field(default_factory=dict)
    peak_pages: int = 0
    allocated_pages: int = 0
    freed_pages: int = 0
    prefetch_attempts: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    prefetch_waste: int = 0
    pressure_prefetch_skips: int = 0

    def __post_init__(self) -> None:
        self.free_pages = list(range(self.total_pages))

    def pressure(self) -> float:
        return 1.0 - (len(self.free_pages) / self.total_pages)

    def pressure_level(self) -> str:
        pressure = self.pressure()
        if pressure >= 0.90:
            return "critical"
        if pressure >= 0.75:
            return "high"
        if pressure >= 0.55:
            return "medium"
        return "low"

    def used_pages(self) -> int:
        return self.total_pages - len(self.free_pages)

    def peak_kv_mb(self) -> float:
        return round(self.peak_pages * self.kv_mb_per_page, 3)

    def pages_needed_for_tokens(self, tokens: int) -> int:
        return math.ceil(max(0, tokens) / self.page_size_tokens)

    def can_cover(self, additional_tokens: int) -> bool:
        return len(self.free_pages) >= self.pages_needed_for_tokens(additional_tokens)

    def allocate_range(
        self,
        *,
        request_id: str,
        token_begin: int,
        token_end: int,
        step: int,
    ) -> list[int]:
        needed = self.pages_needed_for_tokens(token_end - token_begin)
        if needed > len(self.free_pages):
            raise MemoryError("insufficient KV pages")

        allocated = []
        cursor = token_begin
        for _ in range(needed):
            page_id = self.free_pages.pop(0)
            page_end = min(token_end, cursor + self.page_size_tokens)
            self.pages[page_id] = KVPage(
                page_id=page_id,
                owner_request_id=request_id,
                token_begin=cursor,
                token_end=page_end,
                state="resident",
                last_access_step=step,
            )
            self.request_pages.setdefault(request_id, []).append(page_id)
            allocated.append(page_id)
            cursor = page_end

        self.allocated_pages += len(allocated)
        self.peak_pages = max(self.peak_pages, self.used_pages())
        return allocated

    def access_current_page(self, request_id: str, step: int) -> dict:
        pages = self.request_pages.get(request_id, [])
        if not pages:
            return {"hit": False, "page_id": None, "reason": "no_resident_pages"}

        page_id = pages[-1]
        page = self.pages[page_id]
        hit = page.state == "prefetched"
        if hit:
            self.prefetch_hits += 1
            page.state = "resident"
        else:
            self.prefetch_misses += 1
        page.last_access_step = step
        return {"hit": hit, "page_id": page_id, "reason": "accessed_current_page"}

    def prefetch_next_decode_page(
        self,
        *,
        request_id: str,
        step: int,
        pressure_disable_threshold: float,
    ) -> dict:
        pressure = self.pressure()
        if pressure >= pressure_disable_threshold:
            self.pressure_prefetch_skips += 1
            return {
                "event": "page_prefetch_skipped",
                "request_id": request_id,
                "reason": "memory_pressure_above_prefetch_budget",
                "memory_pressure": round(pressure, 4),
            }

        pages = self.request_pages.get(request_id, [])
        if not pages:
            return {
                "event": "page_prefetch_skipped",
                "request_id": request_id,
                "reason": "no_resident_pages",
                "memory_pressure": round(pressure, 4),
            }

        page = self.pages[pages[-1]]
        if page.state == "resident":
            page.state = "prefetched"
            page.last_access_step = step
            self.prefetch_attempts += 1
            return {
                "event": "page_prefetch",
                "request_id": request_id,
                "page_id": page.page_id,
                "memory_pressure": round(pressure, 4),
            }

        return {
            "event": "page_prefetch_skipped",
            "request_id": request_id,
            "reason": "already_prefetched",
            "memory_pressure": round(pressure, 4),
        }

    def release_request(self, request_id: str) -> list[int]:
        pages = self.request_pages.pop(request_id, [])
        for page_id in pages:
            page = self.pages.pop(page_id, None)
            if page and page.state == "prefetched":
                self.prefetch_waste += 1
            self.free_pages.append(page_id)
        self.free_pages.sort()
        self.freed_pages += len(pages)
        return pages

    def pages_for_request(self, request_id: str) -> list[dict]:
        return [
            {
                "page_id": page_id,
                "owner_request_id": self.pages[page_id].owner_request_id,
                "token_begin": self.pages[page_id].token_begin,
                "token_end": self.pages[page_id].token_end,
                "state": self.pages[page_id].state,
                "last_access_step": self.pages[page_id].last_access_step,
            }
            for page_id in self.request_pages.get(request_id, [])
            if page_id in self.pages
        ]

    def summary(self) -> dict:
        accesses = self.prefetch_hits + self.prefetch_misses
        return {
            "enabled": True,
            "policy": "paged_kv_lifecycle_with_pressure_aware_prefetch",
            "total_pages": self.total_pages,
            "page_size_tokens": self.page_size_tokens,
            "resident_pages": self.used_pages(),
            "peak_pages": self.peak_pages,
            "allocated_pages": self.allocated_pages,
            "freed_pages": self.freed_pages,
            "prefetch_attempts": self.prefetch_attempts,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "prefetch_hit_rate": round(self.prefetch_hits / accesses, 4) if accesses else 0.0,
            "prefetch_waste": self.prefetch_waste,
            "pressure_prefetch_skips": self.pressure_prefetch_skips,
            "page_leak_count": self.used_pages(),
        }


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
    page_prefetch: dict = field(default_factory=lambda: {"enabled": False})
    lifecycle: dict = field(default_factory=dict)
    kv_page_lifecycle: dict = field(default_factory=dict)
    paged_attention: dict = field(default_factory=lambda: {"enabled": False})


class RuntimeScheduler:
    def __init__(
        self,
        *,
        policy: str,
        cost_model: CostModel,
        memory: MemoryPlanner,
        rng: random.Random,
        max_decode_batch_size: int = 8,
        paged_attention_cost_enabled: bool = True,
    ) -> None:
        self.policy = policy
        self.cost_model = cost_model
        self.memory = memory
        self.rng = rng
        self.max_decode_batch_size = max_decode_batch_size
        self.paged_attention_cost_enabled = paged_attention_cost_enabled
        self.pressure_limited_candidates = 0
        self.page_prefetch = (
            PagePrefetchPlanner()
            if policy == "cost_aware_memory_pressure_page_prefetch"
            else None
        )

    def run(self, requests: list[Request]) -> SchedulerResult:
        if self.policy == "inflight_paged_kv_continuous_batching":
            return self._run_inflight_paged_kv(requests)

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
        paged_attention = PagedAttentionCostModel(
            enabled=self.paged_attention_cost_enabled,
        )

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
                prefetch_consumed = None
                if self.page_prefetch:
                    prefetch_consumed = self.page_prefetch.consume(
                        active_requests=step_active,
                        allocations=self.memory.allocations,
                        step=step,
                        block_size_tokens=self.memory.block_size_tokens,
                    )
                active_pages = {}
                for item in step_active:
                    blocks = self.memory.allocations.get(item.request_id, [])
                    context_tokens = item.prompt_tokens + step
                    pages_needed = (
                        min(
                            len(blocks),
                            max(1, math.ceil(context_tokens / self.memory.block_size_tokens)),
                        )
                        if blocks
                        else 0
                    )
                    active_pages[item.request_id] = blocks[:pages_needed]
                attention_event = paged_attention.estimate_step(
                    step=step,
                    active_page_ids=active_pages,
                    prefetch_hits=prefetch_consumed["hits"] if prefetch_consumed else 0,
                    prefetch_misses=prefetch_consumed["misses"] if prefetch_consumed else 0,
                )
                step_ms = round(
                    self._observed_decode(len(step_active), avg_context)
                    + attention_event["latency_ms"],
                    3,
                )
                if prefetch_consumed:
                    saving = prefetch_consumed["latency_saving_ms"]
                    warmed_miss_penalty = 0.0
                    if prefetch_consumed["misses"] and step > 0:
                        warmed_miss_penalty = min(
                            0.08,
                            prefetch_consumed["misses"] * self.page_prefetch.wasted_prefetch_penalty_ms,
                        )
                    step_ms = round(max(0.05, step_ms - saving + warmed_miss_penalty), 3)
                decode_step_latencies.append(step_ms)
                decode_total += step_ms
                if self.page_prefetch:
                    event = self.page_prefetch.plan(
                        active_requests=step_active,
                        allocations=self.memory.allocations,
                        step=step,
                        block_size_tokens=self.memory.block_size_tokens,
                        pressure=self.memory.pressure(),
                    )
                    if step % 16 == 0 or event["event"] == "page_prefetch_skipped":
                        scheduler_steps.append(
                            {
                                "time_ms": round(time_ms, 3),
                                **event,
                            }
                        )
                        serving_events.append(
                            {
                                "time_ms": round(time_ms, 3),
                                "event": event["event"],
                                "request_id": req.request_id,
                                "step": step,
                                "active_requests": event.get("active_requests", []),
                                "prefetched_blocks": event.get("prefetched_blocks", 0),
                                "memory_pressure": event.get("memory_pressure"),
                                "prefetch_hits": prefetch_consumed["hits"] if prefetch_consumed else 0,
                                "prefetch_misses": prefetch_consumed["misses"] if prefetch_consumed else 0,
                                "latency_saving_ms": prefetch_consumed["latency_saving_ms"] if prefetch_consumed else 0.0,
                            }
                        )
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
                            "paged_attention": attention_event,
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
                if self.page_prefetch:
                    self.page_prefetch.release_request(done.request_id)
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
            page_prefetch=self.page_prefetch.summary() if self.page_prefetch else {"enabled": False},
            paged_attention=paged_attention.summary(),
        )

    def _run_inflight_paged_kv(self, requests: list[Request]) -> SchedulerResult:
        prefill_chunk_tokens = 256
        max_batched_tokens_per_step = 768
        max_prefill_chunks_per_step = 2
        soft_limit = 0.72
        hard_limit = 0.88
        prefetch_disable_threshold = 0.78

        kv = PagedKVLifecycle(
            total_pages=self.memory.total_blocks,
            page_size_tokens=self.memory.block_size_tokens,
            kv_mb_per_page=self.memory.kv_mb_per_block,
        )
        states = {req.request_id: RuntimeRequestState(request=req) for req in requests}
        future = sorted(requests, key=lambda req: req.arrival_ms)
        waiting: list[str] = []
        prefill_queue: list[str] = []
        decode_queue: list[str] = []
        finished: set[str] = set()
        rejected_ids: set[str] = set()

        time_ms = 0.0
        step = 0
        scheduler_steps: list[dict] = []
        serving_events: list[dict] = []
        backend_placements: list[dict] = []
        request_latencies: list[float] = []
        decode_step_latencies: list[float] = []
        prefill_latencies: list[float] = []
        decode_batch_sizes: list[int] = []
        ttft_latencies: list[float] = []
        tpot_latencies: list[float] = []
        lifecycle_transitions: list[dict] = []
        completed_output_tokens = 0
        rejected = 0
        oom = 0
        delayed = 0
        pressure_limited_ticks = 0
        prefill_chunk_count = 0
        paged_attention = PagedAttentionCostModel(
            enabled=self.paged_attention_cost_enabled,
        )

        def transition(request_id: str, new_state: str, reason: str) -> None:
            state = states[request_id]
            old_state = state.state
            state.state = new_state
            row = {
                "time_ms": round(time_ms, 3),
                "step": step,
                "request_id": request_id,
                "from": old_state,
                "to": new_state,
                "reason": reason,
            }
            lifecycle_transitions.append(row)
            serving_events.append({"event": "request_state_transition", **row})

        def admit_waiting_requests() -> tuple[list[str], list[str]]:
            nonlocal rejected, oom, delayed
            admitted: list[str] = []
            held: list[str] = []
            remaining: list[str] = []
            for request_id in waiting:
                state = states[request_id]
                total_tokens = state.request.prompt_tokens + state.request.output_tokens
                if kv.pages_needed_for_tokens(total_tokens) > kv.total_pages:
                    state.rejected_reason = "request_exceeds_total_kv_pages"
                    transition(request_id, "rejected", state.rejected_reason)
                    rejected += 1
                    oom += 1
                    rejected_ids.add(request_id)
                    continue
                if kv.pressure() >= hard_limit:
                    delayed += 1
                    held.append(request_id)
                    remaining.append(request_id)
                    continue
                transition(request_id, "prefill", "admitted_under_kv_pressure_budget")
                state.admitted_step = step
                admitted.append(request_id)
                prefill_queue.append(request_id)
            waiting[:] = remaining
            return admitted, held

        def decode_cap_for_pressure() -> int:
            level = kv.pressure_level()
            if level == "critical":
                return 1
            if level == "high":
                return min(2, self.max_decode_batch_size)
            if level == "medium":
                return min(4, self.max_decode_batch_size)
            return self.max_decode_batch_size

        def choose_action() -> str:
            if kv.pressure() >= hard_limit:
                return "drain_decode" if decode_queue else "reject_or_delay"
            if prefill_queue and decode_queue and kv.pressure() < soft_limit:
                return "mixed_step"
            if decode_queue:
                return "decode_batch"
            if prefill_queue:
                return "prefill_chunk"
            if waiting:
                return "reject_or_delay"
            return "idle"

        def run_prefill_chunks(limit: int) -> tuple[list[str], int, list[int], float]:
            nonlocal prefill_chunk_count, oom, rejected
            active: list[str] = []
            allocated_pages: list[int] = []
            chunk_tokens_total = 0
            latency = 0.0
            token_budget = max_batched_tokens_per_step
            chunks = 0
            remaining: list[str] = []
            for request_id in prefill_queue:
                state = states[request_id]
                if chunks >= limit or token_budget <= 0 or kv.pressure() >= hard_limit:
                    remaining.append(request_id)
                    continue
                chunk = min(
                    prefill_chunk_tokens,
                    token_budget,
                    state.prompt_tokens_total - state.prompt_tokens_prefilled,
                )
                if chunk <= 0:
                    transition(request_id, "decode", "prefill_complete")
                    decode_queue.append(request_id)
                    continue
                try:
                    pages = kv.allocate_range(
                        request_id=request_id,
                        token_begin=state.prompt_tokens_prefilled,
                        token_end=state.prompt_tokens_prefilled + chunk,
                        step=step,
                    )
                except MemoryError:
                    released = kv.release_request(request_id)
                    state.kv_pages = []
                    state.rejected_reason = "oom_during_prefill_chunk"
                    transition(request_id, "rejected", state.rejected_reason)
                    serving_events.append(
                        {
                            "time_ms": round(time_ms, 3),
                            "event": "kv_pages_freed",
                            "request_id": request_id,
                            "freed_pages": released,
                            "reason": "prefill_oom_cleanup",
                        }
                    )
                    rejected += 1
                    oom += 1
                    rejected_ids.add(request_id)
                    continue
                state.prompt_tokens_prefilled += chunk
                state.kv_pages.extend(pages)
                active.append(request_id)
                allocated_pages.extend(pages)
                chunk_tokens_total += chunk
                token_budget -= chunk
                chunks += 1
                prefill_chunk_count += 1
                chunk_latency = self._observed_prefill(chunk)
                latency = max(latency, chunk_latency)
                prefill_latencies.append(chunk_latency)
                backend_placements.append(
                    {
                        "request_id": request_id,
                        "op": "attention_prefill_chunk",
                        "backend": "gpu",
                        "latency_ms": round(chunk_latency * 0.41, 3),
                    }
                )
                backend_placements.append(
                    {
                        "request_id": request_id,
                        "op": "kv_page_update",
                        "backend": "cpu",
                        "latency_ms": round(self.cost_model.predict_kv_update_ms(len(pages)), 3),
                    }
                )
                serving_events.append(
                    {
                        "time_ms": round(time_ms, 3),
                        "event": "prefill_chunk",
                        "request_id": request_id,
                        "chunk_tokens": chunk,
                        "prompt_tokens_prefilled": state.prompt_tokens_prefilled,
                        "allocated_pages": pages,
                        "memory_pressure": round(kv.pressure(), 4),
                    }
                )
                if state.prompt_tokens_prefilled >= state.prompt_tokens_total:
                    transition(request_id, "decode", "prefill_complete")
                    decode_queue.append(request_id)
                else:
                    remaining.append(request_id)
            prefill_queue[:] = remaining
            return active, chunk_tokens_total, allocated_pages, latency

        def run_decode_batch(force_drain: bool) -> tuple[list[str], list[int], list[int], list[dict], float]:
            nonlocal completed_output_tokens, rejected, oom
            cap = 1 if force_drain else decode_cap_for_pressure()
            active = decode_queue[:cap]
            decode_queue[:] = decode_queue[cap:]
            if not active:
                return [], [], [], [], 0.0

            decode_batch_sizes.append(len(active))
            avg_context = (
                sum(
                    states[request_id].request.prompt_tokens
                    + states[request_id].output_tokens_generated
                    for request_id in active
                )
                / len(active)
            )
            latency = self._observed_decode(len(active), avg_context)
            access_events: list[dict] = []
            allocated_pages: list[int] = []
            freed_pages: list[int] = []
            still_decoding: list[str] = []
            prefetch_hits = 0
            prefetch_misses = 0
            for request_id in active:
                state = states[request_id]
                token_index = state.prompt_tokens_total + state.output_tokens_generated
                if token_index % kv.page_size_tokens == 0:
                    token_end = min(
                        state.prompt_tokens_total + state.request.output_tokens,
                        token_index + kv.page_size_tokens,
                    )
                    try:
                        pages = kv.allocate_range(
                            request_id=request_id,
                            token_begin=token_index,
                            token_end=token_end,
                            step=step,
                        )
                    except MemoryError:
                        released = kv.release_request(request_id)
                        state.kv_pages = []
                        state.rejected_reason = "oom_during_decode_kv_growth"
                        transition(request_id, "rejected", state.rejected_reason)
                        rejected += 1
                        oom += 1
                        rejected_ids.add(request_id)
                        freed_pages.extend(released)
                        continue
                    state.kv_pages.extend(pages)
                    allocated_pages.extend(pages)
                access = kv.access_current_page(request_id, step)
                access_events.append({"request_id": request_id, **access})
                if access["hit"]:
                    prefetch_hits += 1
                else:
                    prefetch_misses += 1
                if access["page_id"] is None:
                    released = kv.release_request(request_id)
                    state.kv_pages = []
                    state.rejected_reason = "decode_without_resident_page"
                    transition(request_id, "rejected", state.rejected_reason)
                    rejected += 1
                    oom += 1
                    rejected_ids.add(request_id)
                    freed_pages.extend(released)
                    continue
                if state.output_tokens_generated == 0:
                    state.ttft_step = step
                    ttft = round(max(0.0, time_ms - state.request.arrival_ms), 3)
                    ttft_latencies.append(ttft)
                    serving_events.append(
                        {
                            "time_ms": round(time_ms, 3),
                            "event": "first_token",
                            "request_id": request_id,
                            "ttft_ms": ttft,
                        }
                    )
                state.output_tokens_generated += 1
                completed_output_tokens += 1
                tpot_latencies.append(latency)
                if state.output_tokens_generated >= state.request.output_tokens:
                    state.finish_step = step
                    transition(request_id, "finished", "output_tokens_complete")
                    released = kv.release_request(request_id)
                    state.kv_pages = []
                    finished.add(request_id)
                    freed_pages.extend(released)
                    request_latencies.append(round(time_ms + latency - state.request.arrival_ms, 3))
                    serving_events.append(
                        {
                            "time_ms": round(time_ms + latency, 3),
                            "event": "kv_pages_freed",
                            "request_id": request_id,
                            "freed_pages": released,
                            "reason": "request_finished",
                        }
                    )
                else:
                    prefetch = kv.prefetch_next_decode_page(
                        request_id=request_id,
                        step=step,
                        pressure_disable_threshold=prefetch_disable_threshold,
                    )
                    access_events.append(prefetch)
                    still_decoding.append(request_id)
            active_pages = {
                request_id: list(kv.request_pages.get(request_id, []))
                for request_id in active
                if request_id not in rejected_ids
            }
            attention_event = paged_attention.estimate_step(
                step=step,
                active_page_ids=active_pages,
                prefetch_hits=prefetch_hits,
                prefetch_misses=prefetch_misses,
            )
            latency = round(latency + attention_event["latency_ms"], 3)
            decode_step_latencies.append(latency)
            for idx in range(1, min(len(active_pages), len(tpot_latencies)) + 1):
                tpot_latencies[-idx] = latency
            access_events.append(attention_event)
            decode_queue.extend(still_decoding)
            return active, allocated_pages, freed_pages, access_events, latency

        while len(finished) + len(rejected_ids) < len(requests):
            if not (waiting or prefill_queue or decode_queue) and future:
                time_ms = max(time_ms, future[0].arrival_ms)
            while future and future[0].arrival_ms <= time_ms:
                req = future.pop(0)
                waiting.append(req.request_id)
                serving_events.append(
                    {
                        "time_ms": round(time_ms, 3),
                        "event": "request_arrived",
                        "request_id": req.request_id,
                        "prompt_tokens": req.prompt_tokens,
                        "generated_tokens_target": req.output_tokens,
                    }
                )

            pressure_before = kv.pressure()
            pressure_level_before = kv.pressure_level()
            admitted, held = admit_waiting_requests()
            action = choose_action()
            if pressure_level_before in {"high", "critical"} or held:
                pressure_limited_ticks += 1

            active_prefill: list[str] = []
            active_decode: list[str] = []
            prefill_tokens = 0
            pages_allocated: list[int] = []
            pages_freed: list[int] = []
            page_events: list[dict] = []
            step_latency = 0.0

            if action in {"prefill_chunk", "mixed_step"}:
                limit = 1 if kv.pressure_level() == "high" else max_prefill_chunks_per_step
                active_prefill, prefill_tokens, allocated, latency = run_prefill_chunks(limit)
                pages_allocated.extend(allocated)
                step_latency = max(step_latency, latency)
            if action in {"decode_batch", "mixed_step", "drain_decode"}:
                active_decode, allocated, freed, events, latency = run_decode_batch(action == "drain_decode")
                pages_allocated.extend(allocated)
                pages_freed.extend(freed)
                page_events.extend(events)
                step_latency = max(step_latency, latency)
            if action == "reject_or_delay" and waiting and kv.pressure() >= hard_limit:
                pressure_limited_ticks += 1
            if action == "reject_or_delay" and prefill_queue and kv.pressure() >= hard_limit:
                request_id = prefill_queue.pop()
                state = states[request_id]
                released = kv.release_request(request_id)
                state.kv_pages = []
                state.rejected_reason = "hard_pressure_without_decode_drain_capacity"
                transition(request_id, "rejected", state.rejected_reason)
                rejected += 1
                oom += 1
                rejected_ids.add(request_id)
                pages_freed.extend(released)
                serving_events.append(
                    {
                        "time_ms": round(time_ms, 3),
                        "event": "kv_pages_freed",
                        "request_id": request_id,
                        "freed_pages": released,
                        "reason": "hard_pressure_cleanup",
                    }
                )
                step_latency = max(step_latency, 0.001)

            scheduler_steps.append(
                {
                    "time_ms": round(time_ms, 3),
                    "event": "scheduler_tick",
                    "step": step,
                    "selected_action": action,
                    "active_request_ids": sorted(set(active_prefill + active_decode)),
                    "prefill_request_ids": active_prefill,
                    "decode_request_ids": active_decode,
                    "prefill_chunk_tokens": prefill_tokens,
                    "decode_batch_size": len(active_decode),
                    "kv_pages_allocated": pages_allocated,
                    "kv_pages_freed": pages_freed,
                    "kv_page_events": page_events,
                    "memory_pressure_before": round(pressure_before, 4),
                    "memory_pressure_after": round(kv.pressure(), 4),
                    "pressure_level": pressure_level_before,
                    "effective_batch_cap": decode_cap_for_pressure(),
                    "waiting": len(waiting),
                    "prefill_queue": len(prefill_queue),
                    "decode_queue": len(decode_queue),
                    "admitted_request_ids": admitted,
                    "delayed_request_ids": held,
                    "worker_id": "local-worker-0",
                    "device_id": "local-gpu-sim-0",
                }
            )

            if step_latency <= 0.0:
                if future:
                    time_ms = max(time_ms + 0.001, future[0].arrival_ms)
                else:
                    break
            else:
                time_ms += step_latency
            step += 1
            if step > 20000:
                raise RuntimeError("in-flight scheduler exceeded deterministic tick budget")

        avg_batch = sum(decode_batch_sizes) / len(decode_batch_sizes) if decode_batch_sizes else 0.0
        page_summary = kv.summary()
        finished_leaks = {
            request_id: kv.pages_for_request(request_id)
            for request_id, state in states.items()
            if state.state == "finished" and kv.pages_for_request(request_id)
        }
        rejected_leaks = {
            request_id: kv.pages_for_request(request_id)
            for request_id, state in states.items()
            if state.state == "rejected" and kv.pages_for_request(request_id)
        }
        complete_lifecycle = all(state.state in {"finished", "rejected"} for state in states.values())
        lifecycle = {
            "policy": self.policy,
            "positioning": (
                "Implemented a TensorRT-LLM-aligned local runtime policy with "
                "in-flight batching, paged KV cache orchestration, "
                "memory-pressure-aware admission, and TTFT/TPOT validation. "
                "This does not modify TensorRT-LLM internals."
            ),
            "request_states": [
                {
                    "request_id": state.request.request_id,
                    "state": state.state,
                    "prompt_tokens_total": state.prompt_tokens_total,
                    "prompt_tokens_prefilled": state.prompt_tokens_prefilled,
                    "output_tokens_generated": state.output_tokens_generated,
                    "kv_pages": state.kv_pages,
                    "ttft_step": state.ttft_step,
                    "finish_step": state.finish_step,
                    "rejected_reason": state.rejected_reason,
                }
                for state in states.values()
            ],
            "transitions": lifecycle_transitions,
            "invariants": {
                "finished_request_releases_pages": not finished_leaks,
                "rejected_request_owns_no_pages": not rejected_leaks,
                "decode_request_has_resident_page": all(
                    event.get("page_id") is not None
                    for tick in scheduler_steps
                    for event in tick.get("kv_page_events", [])
                    if event.get("reason") == "accessed_current_page"
                ),
                "hard_limit_forbids_new_prefill": all(
                    not tick["prefill_request_ids"]
                    for tick in scheduler_steps
                    if tick["memory_pressure_before"] >= hard_limit
                ),
                "every_admitted_request_reaches_terminal_state": complete_lifecycle,
                "page_leak_count": page_summary["page_leak_count"],
            },
            "config": {
                "prefill_chunk_tokens": prefill_chunk_tokens,
                "max_batched_tokens_per_step": max_batched_tokens_per_step,
                "max_decode_batch_size": self.max_decode_batch_size,
                "max_prefill_chunks_per_step": max_prefill_chunks_per_step,
                "memory_pressure_soft_limit": soft_limit,
                "memory_pressure_hard_limit": hard_limit,
            },
            "ttft_ms": {
                "p50": round(_percentile(ttft_latencies, 50), 3),
                "p95": round(_percentile(ttft_latencies, 95), 3),
            },
            "tpot_ms": {
                "p50": round(_percentile(tpot_latencies, 50), 3),
                "p95": round(_percentile(tpot_latencies, 95), 3),
            },
            "pressure_limited_ticks": pressure_limited_ticks,
            "prefill_chunk_count": prefill_chunk_count,
            "paged_attention": paged_attention.summary(),
        }
        kv_requests = [
            {
                "request_id": state.request.request_id,
                "logical_pages_observed": state.kv_pages,
                "context_tokens": state.request.prompt_tokens,
                "generated_tokens": state.request.output_tokens,
                "terminal_state": state.state,
                "kv_cache_mb": round(len(state.kv_pages) * kv.kv_mb_per_page, 3),
            }
            for state in states.values()
        ]

        return SchedulerResult(
            policy=self.policy,
            scheduler_steps=scheduler_steps,
            serving_events=serving_events,
            backend_placements=backend_placements,
            kv_requests=kv_requests,
            request_latencies=request_latencies,
            decode_step_latencies=decode_step_latencies,
            prefill_latencies=ttft_latencies or prefill_latencies,
            completed_requests=len(finished),
            rejected_requests=len(rejected_ids),
            delayed_requests=delayed,
            oom_events=oom,
            peak_allocated_blocks=kv.peak_pages,
            peak_kv_cache_mb=kv.peak_kv_mb(),
            avg_decode_batch_size=round(avg_batch, 4),
            decode_batch_efficiency=round(avg_batch / self.max_decode_batch_size, 4) if self.max_decode_batch_size else 0.0,
            tokens_per_second=round((completed_output_tokens * 1000.0) / time_ms, 3) if time_ms else 0.0,
            finish_time_ms=round(time_ms, 3),
            pressure_limited_candidates=pressure_limited_ticks,
            page_prefetch={
                "enabled": True,
                "policy": "pressure_aware_page_prefetch_inside_inflight_scheduler",
                "attempts": page_summary["prefetch_attempts"],
                "hits": page_summary["prefetch_hits"],
                "misses": page_summary["prefetch_misses"],
                "hit_rate": page_summary["prefetch_hit_rate"],
                "wasted_prefetch_blocks": page_summary["prefetch_waste"],
                "pressure_skips": page_summary["pressure_prefetch_skips"],
            },
            lifecycle=lifecycle,
            kv_page_lifecycle=page_summary,
            paged_attention=paged_attention.summary(),
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


@dataclass
class PagedAttentionCostModel:
    enabled: bool = True
    page_table_lookup_base_ms: float = 0.035
    page_read_ms: float = 0.0045
    non_contiguous_segment_penalty_ms: float = 0.018
    prefetch_miss_penalty_ms: float = 0.012
    prefetch_hit_saving_ms: float = 0.01
    decode_events: list[dict] = field(default_factory=list)

    def estimate_step(
        self,
        *,
        step: int,
        active_page_ids: dict[str, list[int]],
        prefetch_hits: int = 0,
        prefetch_misses: int = 0,
    ) -> dict:
        if not self.enabled:
            event = {
                "event": "paged_attention_read",
                "step": step,
                "active_request_ids": sorted(active_page_ids),
                "requests": {
                    request_id: {
                        "page_ids": pages,
                        "page_count": len(pages),
                    }
                    for request_id, pages in active_page_ids.items()
                },
                "pages_read": 0,
                "non_contiguous_segments": 0,
                "prefetch_hits": prefetch_hits,
                "prefetch_misses": prefetch_misses,
                "page_table_lookup_base_ms": 0.0,
                "page_read_cost_ms": 0.0,
                "non_contiguous_penalty_ms": 0.0,
                "prefetch_hit_saving_ms": 0.0,
                "prefetch_miss_penalty_ms": 0.0,
                "latency_ms": 0.0,
            }
            self.decode_events.append(event)
            return event

        total_pages = sum(len(pages) for pages in active_page_ids.values())
        non_contiguous_segments = 0
        for pages in active_page_ids.values():
            if not pages:
                continue
            ordered = sorted(pages)
            non_contiguous_segments += 1
            non_contiguous_segments += sum(
                1
                for left, right in zip(ordered, ordered[1:])
                if right != left + 1
            )

        non_contiguous_penalty = (
            non_contiguous_segments
            * self.non_contiguous_segment_penalty_ms
        )
        read_cost = total_pages * self.page_read_ms
        hit_saving = prefetch_hits * self.prefetch_hit_saving_ms
        miss_penalty = prefetch_misses * self.prefetch_miss_penalty_ms
        latency_ms = max(
            0.0,
            self.page_table_lookup_base_ms
            + read_cost
            + non_contiguous_penalty
            + miss_penalty
            - hit_saving,
        )
        event = {
            "event": "paged_attention_read",
            "step": step,
            "active_request_ids": sorted(active_page_ids),
            "requests": {
                request_id: {
                    "page_ids": pages,
                    "page_count": len(pages),
                }
                for request_id, pages in active_page_ids.items()
            },
            "pages_read": total_pages,
            "non_contiguous_segments": non_contiguous_segments,
            "prefetch_hits": prefetch_hits,
            "prefetch_misses": prefetch_misses,
            "page_table_lookup_base_ms": self.page_table_lookup_base_ms,
            "page_read_cost_ms": round(read_cost, 4),
            "non_contiguous_penalty_ms": round(non_contiguous_penalty, 4),
            "prefetch_hit_saving_ms": round(hit_saving, 4),
            "prefetch_miss_penalty_ms": round(miss_penalty, 4),
            "latency_ms": round(latency_ms, 4),
        }
        self.decode_events.append(event)
        return event

    def summary(self) -> dict:
        total_steps = len(self.decode_events)
        total_pages = sum(event["pages_read"] for event in self.decode_events)
        total_latency = sum(event["latency_ms"] for event in self.decode_events)
        total_hits = sum(event["prefetch_hits"] for event in self.decode_events)
        total_misses = sum(event["prefetch_misses"] for event in self.decode_events)
        total_non_contiguous = sum(
            event["non_contiguous_segments"]
            for event in self.decode_events
        )
        return {
            "enabled": self.enabled,
            "policy": (
                "paged_attention_read_cost_model"
                if self.enabled
                else "disabled_for_scheduler_focused_artifact"
            ),
            "positioning": (
                "Local paged-attention execution model over RuntimeScheduler KV "
                "page tables; this is not a TensorRT-LLM or vLLM kernel."
                if self.enabled
                else "Disabled to reproduce the scheduler-focused artifact before paged-attention read-cost accounting was added."
            ),
            "input": "decode batch request ids plus resident KV page table",
            "decision": (
                "model page-table reads, contiguous page segments, prefetch "
                "hit/miss effects, and memory-pressure skips before TPOT scoring"
            ),
            "metric": (
                "decode attention latency, pages read per decode step, "
                "non-contiguous segment penalty, prefetch hit/miss impact, TPOT"
            ),
            "decode_steps": total_steps,
            "pages_read": total_pages,
            "avg_pages_per_step": round(total_pages / total_steps, 4) if total_steps else 0.0,
            "non_contiguous_segments": total_non_contiguous,
            "prefetch_hits": total_hits,
            "prefetch_misses": total_misses,
            "total_latency_ms": round(total_latency, 4),
            "avg_latency_ms": round(total_latency / total_steps, 4) if total_steps else 0.0,
            "p95_latency_ms": round(
                _percentile([event["latency_ms"] for event in self.decode_events], 95),
                4,
            ) if self.decode_events else 0.0,
        }


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

    lifecycle_metrics = result.lifecycle or {}
    kv_page_lifecycle = result.kv_page_lifecycle or {}
    paged_attention = result.paged_attention or {}
    return {
        "policy": result.policy,
        "completed_requests": result.completed_requests,
        "rejected_requests": result.rejected_requests,
        "delayed_requests": result.delayed_requests,
        "p95_latency_ms": round(_percentile(result.request_latencies, 95), 3),
        "ttft_p50_ms": lifecycle_metrics.get("ttft_ms", {}).get(
            "p50",
            round(_percentile(result.prefill_latencies, 50), 3),
        ),
        "ttft_p95_ms": lifecycle_metrics.get("ttft_ms", {}).get(
            "p95",
            round(_percentile(result.prefill_latencies, 95), 3),
        ),
        "tpot_p50_ms": lifecycle_metrics.get("tpot_ms", {}).get(
            "p50",
            round(_percentile(result.decode_step_latencies, 50), 3),
        ),
        "tpot_p95_ms": lifecycle_metrics.get("tpot_ms", {}).get(
            "p95",
            round(_percentile(result.decode_step_latencies, 95), 3),
        ),
        "tokens_per_second": result.tokens_per_second,
        "peak_kv_cache_mb": result.peak_kv_cache_mb,
        "avg_decode_batch_size": result.avg_decode_batch_size,
        "decode_batch_efficiency": result.decode_batch_efficiency,
        "pressure_level_counts": pressure_counts,
        "batch_cap_reductions": cap_reductions,
        "pressure_limited_candidates": result.pressure_limited_candidates,
        "finish_time_ms": result.finish_time_ms,
        "page_prefetch": result.page_prefetch,
        "prefill_chunk_count": lifecycle_metrics.get("prefill_chunk_count", 0),
        "kv_page_hit_rate": result.page_prefetch.get("hit_rate", 0.0),
        "paged_attention_latency_p95_ms": paged_attention.get("p95_latency_ms", 0.0),
        "paged_attention_pages_read": paged_attention.get("pages_read", 0),
        "paged_attention_avg_pages_per_step": paged_attention.get("avg_pages_per_step", 0.0),
        "paged_attention_non_contiguous_segments": paged_attention.get("non_contiguous_segments", 0),
        "pages_allocated": kv_page_lifecycle.get("allocated_pages", 0),
        "pages_freed": kv_page_lifecycle.get("freed_pages", 0),
        "pressure_limited_ticks": lifecycle_metrics.get(
            "pressure_limited_ticks",
            result.pressure_limited_candidates,
        ),
        "reject_oom_count": result.oom_events,
        "invariants": lifecycle_metrics.get("invariants", {}),
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
