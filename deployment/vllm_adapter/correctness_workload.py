"""D4B Part K/L/M: deterministic correctness workload and TP1-vs-TP2 comparison.

Uses the real OpenAI-compatible `/v1/completions` endpoint with greedy
(temperature=0) decoding as the primary correctness signal. Token IDs are
derived deterministically from vLLM's own `logprobs.tokens` field (the
literal BPE token strings the model actually emitted) via the real
tokenizer's vocabulary, not by re-tokenizing the returned text -- avoiding
any re-tokenization ambiguity.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_MAX_TOKENS = 24
DEFAULT_SEED = 1234


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    category: str
    prompt: str

    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"prompt_id": self.prompt_id, "category": self.category, "prompt": self.prompt,
                "prompt_sha256": self.prompt_sha256()}


def build_prompt_corpus() -> list[PromptSpec]:
    """8-16 deterministic prompts spanning the required categories."""
    repeated_prefix = "The quick brown fox jumps over the lazy dog. " * 3
    return [
        PromptSpec("short_1", "very_short", "Hello"),
        PromptSpec("short_2", "very_short", "The capital of France is"),
        PromptSpec("medium_1", "medium", "Explain in one sentence why the sky appears blue during the day."),
        PromptSpec("long_prefill_1", "longer_prefill",
                    "Summarize the following in one sentence: Distributed tensor parallelism splits "
                    "individual weight matrices of a neural network across multiple accelerators so that "
                    "each device holds only a shard of every layer, requiring collective communication "
                    "operations such as all-reduce or all-gather to reconstruct full activations at "
                    "specific points in the forward pass, in contrast to pipeline parallelism which "
                    "assigns whole layers to distinct devices instead."),
        PromptSpec("code_1", "code", "Write a Python function that returns the factorial of a non-negative integer n."),
        PromptSpec("numeric_1", "numeric_reasoning", "What is 17 multiplied by 6? Answer with just the number."),
        PromptSpec("repeated_prefix_1", "repeated_prefix", repeated_prefix + "What animal is mentioned?"),
        PromptSpec("short_3", "very_short", "Good morning"),
        PromptSpec("medium_2", "medium", "List three primary colors."),
        PromptSpec("numeric_2", "numeric_reasoning", "If a train travels 60 miles in 1.5 hours, what is its average speed in miles per hour?"),
    ]


@dataclass(frozen=True)
class CompletionRequestParams:
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = DEFAULT_SEED
    logprobs: int = 5
    echo: bool = False
    n: int = 1

    def to_payload(self, *, model: str, prompt: str) -> dict[str, Any]:
        return {
            "model": model, "prompt": prompt, "max_tokens": self.max_tokens,
            "temperature": self.temperature, "top_p": self.top_p, "seed": self.seed,
            "logprobs": self.logprobs, "echo": self.echo, "n": self.n,
        }


@dataclass(frozen=True)
class CompletionResult:
    prompt_id: str
    http_status: int
    latency_s: float
    raw_response: dict[str, Any] | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id, "http_status": self.http_status, "latency_s": self.latency_s,
            "raw_response": self.raw_response, "error": self.error,
        }


def send_completion(
    base_url: str, model: str, spec: PromptSpec, params: CompletionRequestParams, *, timeout_s: float = 60.0,
) -> CompletionResult:
    payload = params.to_payload(model=model, prompt=spec.prompt)
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{base_url}/v1/completions", json=payload, timeout=timeout_s)
        latency_s = time.perf_counter() - t0
        body = resp.json() if resp.content else None
        return CompletionResult(
            prompt_id=spec.prompt_id, http_status=resp.status_code, latency_s=latency_s,
            raw_response=body, error=None if resp.status_code == 200 else str(body),
        )
    except requests.RequestException as exc:
        return CompletionResult(
            prompt_id=spec.prompt_id, http_status=-1, latency_s=time.perf_counter() - t0,
            raw_response=None, error=str(exc),
        )


_DECODED_VOCAB_CACHE: dict[int, dict[str, int | None]] = {}


def _decoded_surface_to_id_map(tokenizer) -> dict[str, int | None]:
    """vLLM's OpenAI-compatible `logprobs.tokens` field returns each token's
    already-decoded human-readable surface string (e.g. " am"), not the raw
    BPE vocabulary entry (e.g. "Ġam") -- so `convert_tokens_to_ids` cannot
    map it directly. Build a one-time reverse map from decoded surface
    string -> vocab id by decoding every vocab entry individually. A
    surface string reachable by more than one distinct token id is marked
    ambiguous (None) so callers fail closed rather than guess."""
    cached = _DECODED_VOCAB_CACHE.get(id(tokenizer))
    if cached is not None:
        return cached
    mapping: dict[str, int | None] = {}
    vocab = tokenizer.get_vocab()
    for tok_str, tok_id in vocab.items():
        decoded = tokenizer.convert_tokens_to_string([tok_str])
        if decoded in mapping and mapping[decoded] != tok_id:
            mapping[decoded] = None  # ambiguous surface string -> multiple ids
        else:
            mapping[decoded] = tok_id
    _DECODED_VOCAB_CACHE[id(tokenizer)] = mapping
    return mapping


def extract_token_ids(result: CompletionResult, tokenizer) -> list[int] | None:
    """Derive exact generated token IDs from vLLM's logprobs.tokens field
    (the literal emitted tokens, in their decoded surface form) via a
    reverse lookup against the real tokenizer vocabulary -- not by
    re-tokenizing the full surface text. Fails closed (returns None) on any
    unmapped or ambiguous token, or if a round-trip decode check disagrees
    with vLLM's own reported text for these tokens."""
    if not result.raw_response:
        return None
    choices = result.raw_response.get("choices") or []
    if not choices:
        return None
    logprobs = choices[0].get("logprobs") or {}
    tokens = logprobs.get("tokens")
    if not tokens:
        return None

    surface_to_id = _decoded_surface_to_id_map(tokenizer)
    ids: list[int] = []
    for tok in tokens:
        tid = surface_to_id.get(tok)
        if tid is None:
            return None  # fail closed: unmapped or ambiguous surface token
        ids.append(tid)

    # Round-trip integrity check: decoding the derived ids must reproduce
    # the exact concatenation of vLLM's own reported token strings.
    if tokenizer.decode(ids) != "".join(tokens):
        return None
    return ids


@dataclass(frozen=True)
class PromptComparison:
    prompt_id: str
    tp1_status: int
    tp2_status: int
    status_match: bool
    text_match: bool
    finish_reason_match: bool
    token_ids_match: bool | None
    prompt_tokens_match: bool
    completion_tokens_match: bool
    tp1_text: str | None
    tp2_text: str | None
    tp1_finish_reason: str | None
    tp2_finish_reason: str | None
    tp1_token_ids: list[int] | None
    tp2_token_ids: list[int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id, "tp1_status": self.tp1_status, "tp2_status": self.tp2_status,
            "status_match": self.status_match, "text_match": self.text_match,
            "finish_reason_match": self.finish_reason_match, "token_ids_match": self.token_ids_match,
            "prompt_tokens_match": self.prompt_tokens_match, "completion_tokens_match": self.completion_tokens_match,
            "tp1_text": self.tp1_text, "tp2_text": self.tp2_text,
            "tp1_finish_reason": self.tp1_finish_reason, "tp2_finish_reason": self.tp2_finish_reason,
            "tp1_token_ids": self.tp1_token_ids, "tp2_token_ids": self.tp2_token_ids,
        }


def compare_completions(prompt_id: str, tp1: CompletionResult, tp2: CompletionResult, tokenizer) -> PromptComparison:
    tp1_choice = (tp1.raw_response or {}).get("choices", [{}])[0] if tp1.raw_response else {}
    tp2_choice = (tp2.raw_response or {}).get("choices", [{}])[0] if tp2.raw_response else {}
    tp1_usage = (tp1.raw_response or {}).get("usage", {}) if tp1.raw_response else {}
    tp2_usage = (tp2.raw_response or {}).get("usage", {}) if tp2.raw_response else {}

    tp1_ids = extract_token_ids(tp1, tokenizer)
    tp2_ids = extract_token_ids(tp2, tokenizer)
    token_ids_match = (tp1_ids == tp2_ids) if (tp1_ids is not None and tp2_ids is not None) else None

    return PromptComparison(
        prompt_id=prompt_id, tp1_status=tp1.http_status, tp2_status=tp2.http_status,
        status_match=tp1.http_status == tp2.http_status == 200,
        text_match=tp1_choice.get("text") == tp2_choice.get("text"),
        finish_reason_match=tp1_choice.get("finish_reason") == tp2_choice.get("finish_reason"),
        token_ids_match=token_ids_match,
        prompt_tokens_match=tp1_usage.get("prompt_tokens") == tp2_usage.get("prompt_tokens"),
        completion_tokens_match=tp1_usage.get("completion_tokens") == tp2_usage.get("completion_tokens"),
        tp1_text=tp1_choice.get("text"), tp2_text=tp2_choice.get("text"),
        tp1_finish_reason=tp1_choice.get("finish_reason"), tp2_finish_reason=tp2_choice.get("finish_reason"),
        tp1_token_ids=tp1_ids, tp2_token_ids=tp2_ids,
    )


@dataclass(frozen=True)
class LogprobComparison:
    prompt_id: str
    selected_token_ids_match: bool
    max_abs_logprob_error: float
    mean_abs_logprob_error: float
    topk_id_agreement_rate: float
    positions_compared: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id, "selected_token_ids_match": self.selected_token_ids_match,
            "max_abs_logprob_error": self.max_abs_logprob_error, "mean_abs_logprob_error": self.mean_abs_logprob_error,
            "topk_id_agreement_rate": self.topk_id_agreement_rate, "positions_compared": self.positions_compared,
        }


def compare_logprobs(prompt_id: str, tp1: CompletionResult, tp2: CompletionResult, tokenizer) -> LogprobComparison | None:
    if not tp1.raw_response or not tp2.raw_response:
        return None
    tp1_lp = (tp1.raw_response.get("choices") or [{}])[0].get("logprobs") or {}
    tp2_lp = (tp2.raw_response.get("choices") or [{}])[0].get("logprobs") or {}
    tp1_tokens = tp1_lp.get("tokens") or []
    tp2_tokens = tp2_lp.get("tokens") or []
    tp1_top = tp1_lp.get("top_logprobs") or []
    tp2_top = tp2_lp.get("top_logprobs") or []
    if not tp1_tokens or not tp2_tokens:
        return None

    n = min(len(tp1_tokens), len(tp2_tokens), len(tp1_top), len(tp2_top))
    selected_match = tp1_tokens[:n] == tp2_tokens[:n]
    abs_errors: list[float] = []
    agreement_hits = 0
    agreement_total = 0
    for i in range(n):
        d1 = tp1_top[i] or {}
        d2 = tp2_top[i] or {}
        common_keys = set(d1) & set(d2)
        for k in common_keys:
            try:
                abs_errors.append(abs(float(d1[k]) - float(d2[k])))
            except (TypeError, ValueError):
                continue
        agreement_total += 1
        top1_ids = set(sorted(d1, key=lambda k: -d1[k])[:5])
        top2_ids = set(sorted(d2, key=lambda k: -d2[k])[:5])
        if top1_ids and top2_ids:
            overlap = len(top1_ids & top2_ids) / max(1, len(top1_ids | top2_ids))
            agreement_hits += overlap

    return LogprobComparison(
        prompt_id=prompt_id, selected_token_ids_match=selected_match,
        max_abs_logprob_error=max(abs_errors) if abs_errors else 0.0,
        mean_abs_logprob_error=(sum(abs_errors) / len(abs_errors)) if abs_errors else 0.0,
        topk_id_agreement_rate=(agreement_hits / agreement_total) if agreement_total else 0.0,
        positions_compared=n,
    )
