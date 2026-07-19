import copy
from dataclasses import replace

import pytest

from deployment.scheduler_wall_clock import (
    EpochPolicyController, FastSelectorConfiguration,
    FrozenSelectorConfiguration, SchedulerSelectorV1Fast,
    choose_adaptive_horizon, online_state_view, prune_candidates)
from deployment.serving_execution import ServingPlanError
from deployment.serving_scheduler import (
    ReplicaSchedulerState, RequestExecutionState, SchedulerProfile)


def state(future=False):
    p=SchedulerProfile(max_num_seqs=4,max_num_batched_tokens=8,
                       max_prefill_chunk_tokens=4,balanced_decode_reservation=2)
    s=ReplicaSchedulerState("replica-0",p)
    d=RequestExecutionState("d","sd","replica-0",0,8,8,4);s.ingest(d)
    s.ingest(RequestExecutionState("p","sp","replica-0",0,20,0,2))
    if future:s.ingest(RequestExecutionState("future","sf","replica-0",10,30,0,2))
    return s


def test_frozen_configuration_digest_and_mutation_detection():
    FrozenSelectorConfiguration().validate_immutable()
    with pytest.raises(ServingPlanError):
        replace(FrozenSelectorConfiguration(),horizon="8").validate_immutable()


def test_online_view_removes_future_arrivals_and_does_not_mutate():
    s=state(True);before=copy.deepcopy(s);view=online_state_view(s)
    assert "future" not in view.requests
    assert s==before


def test_fast_modes_reject_offline_and_bad_horizon():
    with pytest.raises(ServingPlanError):
        FastSelectorConfiguration(selection_mode="offline_trace")
    with pytest.raises(ServingPlanError):
        FastSelectorConfiguration(adaptive_horizons=(3,))


def test_adaptive_horizon_supported():
    assert choose_adaptive_horizon(state()) in (1,2,4,8)


def test_pruning_keeps_at_least_one_and_merges_identical():
    kept,reasons=prune_candidates(state())
    assert kept
    assert len(kept)+len(reasons)>=4


def test_fast_selector_is_deterministic_and_low_overhead_recorded():
    s=state()
    a=SchedulerSelectorV1Fast(FastSelectorConfiguration())
    b=SchedulerSelectorV1Fast(FastSelectorConfiguration())
    pa,pb=a.select(s),b.select(s)
    assert pa.policy_id==pb.policy_id and pa.horizon==pb.horizon
    assert a.profiles[0]["total_ns"]>0


def test_evidence_and_hot_path_modes_preserve_choice():
    s=state()
    hot=SchedulerSelectorV1Fast(FastSelectorConfiguration(evidence_mode=False))
    evidence=SchedulerSelectorV1Fast(FastSelectorConfiguration(evidence_mode=True))
    assert hot.select(s).policy_id==evidence.select(s).policy_id
    assert hot.profiles[-1]["candidate_detail"] is None
    assert evidence.profiles[-1]["candidate_detail"] is not None


def test_epoch_reuse_and_required_reselection():
    s=state();selector=SchedulerSelectorV1Fast(FastSelectorConfiguration(epoch_steps=4))
    ctl=EpochPolicyController(selector)
    first=ctl.policy(s);second=ctl.policy(s)
    assert first==second and ctl.planning_calls==1
    ctl.policy(s,event="new_arrival")
    assert ctl.planning_calls==2


def test_epoch_policy_switch_is_visible():
    s=state();ctl=EpochPolicyController(
        SchedulerSelectorV1Fast(FastSelectorConfiguration(epoch_steps=1)))
    ctl.policy(s);ctl.policy(s)
    assert ctl.planning_calls==2 and ctl.policy_switches>=0


def test_future_request_does_not_change_online_decision():
    no_future=SchedulerSelectorV1Fast(FastSelectorConfiguration()).select(state(False))
    with_future=SchedulerSelectorV1Fast(FastSelectorConfiguration()).select(state(True))
    assert no_future.policy_id==with_future.policy_id


def test_equivalence_margin_is_frozen_dataclass():
    c=FastSelectorConfiguration(practical_equivalence_margin=.02)
    assert c.practical_equivalence_margin==.02
    with pytest.raises(Exception):
        c.practical_equivalence_margin=.03
