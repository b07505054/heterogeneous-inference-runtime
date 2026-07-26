"""Metadata-only physical KV page ownership for multi-request serving."""
from __future__ import annotations

import math
from dataclasses import dataclass


class KVPageAllocationError(RuntimeError):
    pass


class KVPageStateError(RuntimeError):
    pass


@dataclass
class _RequestKVState:
    physical_pages: list[int]
    valid_token_count: int


@dataclass(frozen=True)
class KVPageAppendReservation:
    request_id: str
    logical_token_index: int
    physical_page: int
    block_table: tuple[int, ...]
    allocated_new_page: bool


class KVPageManager:
    def __init__(self, total_pages: int, tokens_per_page: int) -> None:
        self._require_positive_int(total_pages, "total_pages")
        self._require_positive_int(tokens_per_page, "tokens_per_page")
        self._total_pages = total_pages
        self._tokens_per_page = tokens_per_page
        self._free_pages = list(range(total_pages))
        self._requests: dict[str, _RequestKVState] = {}
        self._page_owner: dict[int, str] = {}
        self._pending_appends: dict[str, KVPageAppendReservation] = {}

    @property
    def total_pages(self) -> int:
        return self._total_pages

    @property
    def tokens_per_page(self) -> int:
        return self._tokens_per_page

    def num_free_pages(self) -> int:
        return len(self._free_pages)

    def num_allocated_pages(self) -> int:
        return self._total_pages - len(self._free_pages)

    def has_request(self, request_id: str) -> bool:
        return request_id in self._requests

    def required_pages_for_tokens(self, token_count: int) -> int:
        self._require_non_negative_int(token_count, "token_count")
        return math.ceil(token_count / self._tokens_per_page)

    def additional_pages_required(
        self,
        request_id: str,
        target_token_count: int,
    ) -> int:
        self._require_request_id(request_id)
        self._require_positive_int(target_token_count, "target_token_count")
        state = self._requests.get(request_id)
        current_tokens = state.valid_token_count if state else 0
        if target_token_count < current_tokens:
            raise KVPageStateError("target_token_count_smaller_than_current")
        current_pages = len(state.physical_pages) if state else 0
        target_pages = self.required_pages_for_tokens(target_token_count)
        return max(0, target_pages - current_pages)

    def can_reserve(
        self,
        request_id: str,
        target_token_count: int,
    ) -> bool:
        return (
            self.additional_pages_required(request_id, target_token_count)
            <= len(self._free_pages)
        )

    def reserve_prefill(
        self,
        request_id: str,
        token_count: int,
    ) -> tuple[int, ...]:
        self._require_request_id(request_id)
        self._require_positive_int(token_count, "token_count")
        if request_id in self._requests:
            raise KVPageStateError("duplicate_request_id")
        if request_id in self._pending_appends:
            raise KVPageStateError("pending_append_exists")

        needed = self.required_pages_for_tokens(token_count)
        if needed > len(self._free_pages):
            raise KVPageAllocationError("out_of_kv_pages")

        allocated = self._free_pages[:needed]
        self._free_pages = self._free_pages[needed:]
        self._requests[request_id] = _RequestKVState(
            physical_pages=list(allocated),
            valid_token_count=token_count,
        )
        for page_id in allocated:
            self._page_owner[page_id] = request_id
        self.validate_invariants()
        return tuple(allocated)

    def append_token(
        self,
        request_id: str,
        logical_token_index: int,
    ) -> int:
        reservation = self.begin_append_token(request_id, logical_token_index)
        self.commit_append_token(reservation)
        return reservation.physical_page

    def begin_append_token(
        self,
        request_id: str,
        logical_token_index: int,
    ) -> KVPageAppendReservation:
        self._require_request_id(request_id)
        self._require_non_negative_int(logical_token_index, "logical_token_index")
        if request_id in self._pending_appends:
            raise KVPageStateError("pending_append_exists")
        state = self._request_state(request_id)
        if logical_token_index != state.valid_token_count:
            raise KVPageStateError("non_sequential_logical_token")

        block_index = logical_token_index // self._tokens_per_page
        if block_index < len(state.physical_pages):
            return KVPageAppendReservation(
                request_id=request_id,
                logical_token_index=logical_token_index,
                physical_page=state.physical_pages[block_index],
                block_table=tuple(state.physical_pages),
                allocated_new_page=False,
            )

        if not self._free_pages:
            raise KVPageAllocationError("out_of_kv_pages")

        page_id = self._free_pages[0]
        self._free_pages = self._free_pages[1:]
        self._page_owner[page_id] = request_id
        reservation = KVPageAppendReservation(
            request_id=request_id,
            logical_token_index=logical_token_index,
            physical_page=page_id,
            block_table=tuple(state.physical_pages + [page_id]),
            allocated_new_page=True,
        )
        self._pending_appends[request_id] = reservation
        self.validate_invariants()
        return reservation

    def commit_append_token(self, reservation: KVPageAppendReservation) -> int:
        state = self._request_state(reservation.request_id)
        self._validate_append_reservation(reservation, state)
        if reservation.allocated_new_page:
            state.physical_pages.append(reservation.physical_page)
            self._pending_appends.pop(reservation.request_id)
        state.valid_token_count += 1
        self.validate_invariants()
        return reservation.physical_page

    def rollback_append_token(self, reservation: KVPageAppendReservation) -> None:
        state = self._request_state(reservation.request_id)
        self._validate_append_reservation(reservation, state)
        if reservation.allocated_new_page:
            self._pending_appends.pop(reservation.request_id)
            self._page_owner.pop(reservation.physical_page)
            self._free_pages = sorted(self._free_pages + [reservation.physical_page])
        self.validate_invariants()

    def block_table(
        self,
        request_id: str,
    ) -> tuple[int, ...]:
        return tuple(self._request_state(request_id).physical_pages)

    def valid_token_count(
        self,
        request_id: str,
    ) -> int:
        return self._request_state(request_id).valid_token_count

    def release(
        self,
        request_id: str,
    ) -> tuple[int, ...]:
        self._require_request_id(request_id)
        state = self._request_state(request_id)
        if request_id in self._pending_appends:
            raise KVPageStateError("pending_append_exists")
        released = tuple(state.physical_pages)
        for page_id in released:
            if self._page_owner.get(page_id) != request_id:
                raise KVPageStateError("page_owner_mismatch")

        del self._requests[request_id]
        for page_id in released:
            self._page_owner.pop(page_id)
        self._free_pages = sorted(self._free_pages + list(released))
        self.validate_invariants()
        return released

    def validate_invariants(self) -> None:
        free_set = set(self._free_pages)
        if len(free_set) != len(self._free_pages):
            raise KVPageStateError("duplicate_free_page")
        if any(page_id < 0 or page_id >= self._total_pages for page_id in free_set):
            raise KVPageStateError("free_page_out_of_range")

        allocated_pages: list[int] = []
        expected_owner: dict[int, str] = {}
        for request_id, state in self._requests.items():
            self._require_request_id(request_id)
            self._require_non_negative_int(
                state.valid_token_count,
                "valid_token_count",
            )
            pages = state.physical_pages
            if len(set(pages)) != len(pages):
                raise KVPageStateError("duplicate_request_page")
            if any(page_id < 0 or page_id >= self._total_pages for page_id in pages):
                raise KVPageStateError("request_page_out_of_range")

            expected_pages = self.required_pages_for_tokens(state.valid_token_count)
            if len(pages) != expected_pages:
                raise KVPageStateError("block_table_length_mismatch")
            if state.valid_token_count > len(pages) * self._tokens_per_page:
                raise KVPageStateError("valid_tokens_exceed_capacity")

            allocated_pages.extend(pages)
            for page_id in pages:
                if page_id in expected_owner:
                    raise KVPageStateError("page_owned_by_multiple_requests")
                expected_owner[page_id] = request_id

        for request_id, reservation in self._pending_appends.items():
            state = self._requests.get(request_id)
            if state is None:
                raise KVPageStateError("pending_append_unknown_request")
            self._validate_append_reservation(reservation, state)
            if reservation.allocated_new_page:
                page_id = reservation.physical_page
                if page_id in expected_owner:
                    raise KVPageStateError("page_owned_by_multiple_requests")
                expected_owner[page_id] = request_id
                allocated_pages.append(page_id)

        allocated_set = set(allocated_pages)
        if free_set & allocated_set:
            raise KVPageStateError("page_both_free_and_allocated")
        if len(free_set) + len(allocated_set) != self._total_pages:
            raise KVPageStateError("page_accounting_mismatch")
        if set(self._page_owner) != allocated_set:
            raise KVPageStateError("page_owner_keys_mismatch")
        for page_id, owner in self._page_owner.items():
            if expected_owner.get(page_id) != owner:
                raise KVPageStateError("page_owner_value_mismatch")

    def _request_state(self, request_id: str) -> _RequestKVState:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise KVPageStateError("unknown_request_id") from exc

    def _validate_append_reservation(
        self,
        reservation: KVPageAppendReservation,
        state: _RequestKVState,
    ) -> None:
        if reservation.logical_token_index != state.valid_token_count:
            raise KVPageStateError("stale_append_reservation")
        current_pages = tuple(state.physical_pages)
        if reservation.allocated_new_page:
            expected_table = current_pages + (reservation.physical_page,)
            if self._pending_appends.get(reservation.request_id) != reservation:
                raise KVPageStateError("stale_append_reservation")
        else:
            expected_table = current_pages
        if reservation.block_table != expected_table:
            raise KVPageStateError("append_reservation_table_mismatch")
        block_index = reservation.logical_token_index // self._tokens_per_page
        if block_index >= len(reservation.block_table):
            raise KVPageStateError("append_reservation_table_mismatch")
        if reservation.block_table[block_index] != reservation.physical_page:
            raise KVPageStateError("append_reservation_page_mismatch")

    @staticmethod
    def _require_request_id(request_id: str) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise KVPageStateError("invalid_request_id")

    @staticmethod
    def _require_positive_int(value: int, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise KVPageStateError(f"{name}_must_be_positive")

    @staticmethod
    def _require_non_negative_int(value: int, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise KVPageStateError(f"{name}_must_be_non_negative")
