import pytest

from deployment.execution_plan.kv_page_manager import (
    KVPageAllocationError,
    KVPageManager,
    KVPageStateError,
)


def test_construction_validation():
    with pytest.raises(KVPageStateError, match="total_pages_must_be_positive"):
        KVPageManager(total_pages=0, tokens_per_page=4)
    with pytest.raises(KVPageStateError, match="tokens_per_page_must_be_positive"):
        KVPageManager(total_pages=4, tokens_per_page=0)


def test_one_page_prefill_allocation():
    manager = KVPageManager(total_pages=4, tokens_per_page=8)

    assert manager.reserve_prefill("r1", 3) == (0,)
    assert manager.block_table("r1") == (0,)
    assert manager.valid_token_count("r1") == 3
    assert manager.num_free_pages() == 3
    assert manager.num_allocated_pages() == 1


def test_exact_page_fill_does_not_allocate_extra_page():
    manager = KVPageManager(total_pages=4, tokens_per_page=8)

    assert manager.reserve_prefill("r1", 8) == (0,)
    assert manager.num_allocated_pages() == 1
    assert manager.additional_pages_required("r1", 8) == 0


def test_prefill_spanning_multiple_pages():
    manager = KVPageManager(total_pages=4, tokens_per_page=8)

    assert manager.reserve_prefill("r1", 17) == (0, 1, 2)
    assert manager.valid_token_count("r1") == 17
    assert manager.num_free_pages() == 1


def test_append_inside_existing_page():
    manager = KVPageManager(total_pages=4, tokens_per_page=8)
    manager.reserve_prefill("r1", 3)

    assert manager.append_token("r1", 3) == 0
    assert manager.block_table("r1") == (0,)
    assert manager.valid_token_count("r1") == 4
    assert manager.num_free_pages() == 3


def test_append_crossing_one_page_boundary():
    manager = KVPageManager(total_pages=4, tokens_per_page=4)
    manager.reserve_prefill("r1", 4)

    assert manager.append_token("r1", 4) == 1
    assert manager.block_table("r1") == (0, 1)
    assert manager.valid_token_count("r1") == 5
    assert manager.num_free_pages() == 2


def test_two_requests_never_receive_same_page():
    manager = KVPageManager(total_pages=4, tokens_per_page=4)

    first = set(manager.reserve_prefill("r1", 5))
    second = set(manager.reserve_prefill("r2", 4))

    assert first.isdisjoint(second)
    manager.validate_invariants()


def test_release_returns_every_page():
    manager = KVPageManager(total_pages=4, tokens_per_page=4)
    manager.reserve_prefill("r1", 9)

    assert manager.release("r1") == (0, 1, 2)
    assert manager.num_free_pages() == 4
    assert not manager.has_request("r1")


def test_released_pages_are_reused_deterministically():
    manager = KVPageManager(total_pages=4, tokens_per_page=4)
    manager.reserve_prefill("r1", 8)
    manager.reserve_prefill("r2", 4)

    assert manager.release("r1") == (0, 1)
    assert manager.reserve_prefill("r3", 4) == (0,)


def test_oom_during_prefill_causes_no_partial_mutation():
    manager = KVPageManager(total_pages=2, tokens_per_page=4)
    manager.reserve_prefill("r1", 4)
    before = manager.block_table("r1"), manager.num_free_pages()

    with pytest.raises(KVPageAllocationError, match="out_of_kv_pages"):
        manager.reserve_prefill("r2", 8)

    assert manager.block_table("r1") == before[0]
    assert manager.num_free_pages() == before[1]
    assert not manager.has_request("r2")
    manager.validate_invariants()


def test_oom_during_boundary_append_causes_no_partial_mutation():
    manager = KVPageManager(total_pages=2, tokens_per_page=4)
    manager.reserve_prefill("r1", 4)
    manager.reserve_prefill("r2", 4)
    before_table = manager.block_table("r1")
    before_tokens = manager.valid_token_count("r1")
    before_free = manager.num_free_pages()

    with pytest.raises(KVPageAllocationError, match="out_of_kv_pages"):
        manager.append_token("r1", 4)

    assert manager.block_table("r1") == before_table
    assert manager.valid_token_count("r1") == before_tokens
    assert manager.num_free_pages() == before_free
    manager.validate_invariants()


def test_duplicate_request_id_rejection():
    manager = KVPageManager(total_pages=2, tokens_per_page=4)
    manager.reserve_prefill("r1", 1)

    with pytest.raises(KVPageStateError, match="duplicate_request_id"):
        manager.reserve_prefill("r1", 1)


def test_unknown_request_append_rejection():
    manager = KVPageManager(total_pages=2, tokens_per_page=4)

    with pytest.raises(KVPageStateError, match="unknown_request_id"):
        manager.append_token("missing", 0)


def test_non_sequential_logical_token_append_rejection():
    manager = KVPageManager(total_pages=2, tokens_per_page=4)
    manager.reserve_prefill("r1", 2)

    with pytest.raises(KVPageStateError, match="non_sequential_logical_token"):
        manager.append_token("r1", 3)


def test_unknown_request_release_rejection():
    manager = KVPageManager(total_pages=2, tokens_per_page=4)

    with pytest.raises(KVPageStateError, match="unknown_request_id"):
        manager.release("missing")


def test_double_release_rejection():
    manager = KVPageManager(total_pages=2, tokens_per_page=4)
    manager.reserve_prefill("r1", 1)
    manager.release("r1")

    with pytest.raises(KVPageStateError, match="unknown_request_id"):
        manager.release("r1")


def test_block_table_stability():
    manager = KVPageManager(total_pages=4, tokens_per_page=4)
    manager.reserve_prefill("r1", 4)
    initial = manager.block_table("r1")
    manager.reserve_prefill("r2", 4)
    manager.append_token("r1", 4)

    assert manager.block_table("r1")[: len(initial)] == initial


def test_can_reserve_does_not_mutate_state():
    manager = KVPageManager(total_pages=3, tokens_per_page=4)
    manager.reserve_prefill("r1", 4)
    before = (
        manager.block_table("r1"),
        manager.valid_token_count("r1"),
        manager.num_free_pages(),
    )

    assert manager.can_reserve("r1", 8) is True
    assert manager.can_reserve("r2", 12) is False
    assert (
        manager.block_table("r1"),
        manager.valid_token_count("r1"),
        manager.num_free_pages(),
    ) == before


def test_invalid_target_token_count_rejection():
    manager = KVPageManager(total_pages=3, tokens_per_page=4)
    manager.reserve_prefill("r1", 4)

    with pytest.raises(KVPageStateError, match="target_token_count_must_be_positive"):
        manager.can_reserve("r2", 0)
    with pytest.raises(KVPageStateError, match="target_token_count_smaller_than_current"):
        manager.can_reserve("r1", 3)


def test_invariant_validation_detects_corrupted_internal_state():
    manager = KVPageManager(total_pages=4, tokens_per_page=4)
    manager.reserve_prefill("r1", 4)
    manager.reserve_prefill("r2", 4)

    manager._requests["r2"].physical_pages[0] = manager.block_table("r1")[0]

    with pytest.raises(KVPageStateError, match="page_owned_by_multiple_requests"):
        manager.validate_invariants()
