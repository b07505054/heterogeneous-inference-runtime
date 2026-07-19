from deployment.scheduler_rollout_v2 import clone_scheduler_state,rollout_policy,state_hash
from deployment.serving_scheduler import ReplicaSchedulerState,RequestExecutionState,SchedulerProfile

def sample():
 s=ReplicaSchedulerState("replica-0",SchedulerProfile(4,10,6,1,2))
 s.ingest(RequestExecutionState("a","sp-a","replica-0",0,15,0,3))
 s.ingest(RequestExecutionState("b","sp-b","replica-0",0,5,0,2))
 return s

def test_rollout_does_not_mutate_live_state():
 s=sample();before=state_hash(s);rollout_policy(s,"decode_first",8)
 assert state_hash(s)==before

def test_rollout_is_deterministic():
 s=sample()
 assert rollout_policy(s,"chunked_balanced",8)==rollout_policy(s,"chunked_balanced",8)

def test_rollout_preserves_request_identity_and_ranges():
 rows=rollout_policy(sample(),"prefill_first",8)
 assert rows and all(x[0] in {"a","b"} and x[3]>0 for r in rows for x in r.items)

def test_candidate_clones_are_isolated():
 s=sample();a=clone_scheduler_state(s);b=clone_scheduler_state(s)
 a.requests["a"].prefill_completed_tokens=3
 assert b.requests["a"].prefill_completed_tokens==0

def test_rollout_rejects_invalid_horizon():
 import pytest
 with pytest.raises(ValueError):rollout_policy(sample(),"decode_first",0)
