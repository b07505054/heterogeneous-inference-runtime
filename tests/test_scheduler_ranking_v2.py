import copy
import hashlib
import json
import unittest

from deployment.scheduler_ranking_v2 import (
    IncrementalSchedulerState, RankingModel, RankingSelectorV2,
    SummaryDelta, reference_summary)
from deployment.serving_execution import ServingPlanError
from deployment.serving_scheduler import (
    ReplicaSchedulerState, RequestExecutionState, SchedulerProfile)


def state(n=8):
    s=ReplicaSchedulerState("r0",SchedulerProfile(max_num_seqs=16,
        max_num_batched_tokens=128,max_prefill_chunk_tokens=64),clock_ms=10)
    for i in range(n):
        r=RequestExecutionState(f"q{i}",f"p{i}","r0",0,32+i,i%4,4,
                                phase="PREFILL")
        s.ingest(r)
    return s


def model(coeff=None, **kw):
    payload=dict(version="ranking_v2_test",
        coefficients={p:dict(coeff or {}) for p in
                      ("decode_first","prefill_first","chunked_balanced","slo_aware")},
        intercepts={"decode_first":2.0,"prefill_first":3.0,
                    "chunked_balanced":1.0,"slo_aware":4.0},
        feature_means={},feature_scales={},
        training_provenance="independent_wall_clock_test",
        equivalence_margin=.02,hysteresis_margin=.01,
        default_policy="chunked_balanced")
    payload.update(kw)
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,
        separators=(",",":")).encode()).hexdigest()
    return RankingModel(**payload,frozen_digest=digest)


class RankingV2Tests(unittest.TestCase):
    def test_incremental_matches_reference(self):
        s=state();inc=IncrementalSchedulerState(s,frontier_size=3)
        self.assertEqual(inc.snapshot(),reference_summary(s,frontier_size=3))

    def test_progress_update_matches_reference(self):
        s=state();inc=IncrementalSchedulerState(s)
        s.requests["q0"].prefill_completed_tokens=7;s.version+=1
        inc.upsert(s.requests["q0"],s.clock_ms)
        a,b=inc.snapshot(),reference_summary(s)
        for name in ("active_count","total_remaining_prefill",
                     "total_remaining_decode","prefix_hit_tokens"):
            self.assertEqual(getattr(a,name),getattr(b,name))

    def test_completion_removed_from_hot_state(self):
        s=state(1);inc=IncrementalSchedulerState(s)
        r=s.requests["q0"];r.prefill_completed_tokens=r.uncached_prompt_tokens
        r.decode_completed_tokens=r.expected_output_tokens;r.phase="FINISHED"
        inc.upsert(r,11)
        x=inc.snapshot();self.assertEqual(x.active_count,0)
        self.assertEqual(x.completed_count,1)

    def test_future_request_rejected(self):
        s=state(1);inc=IncrementalSchedulerState(s)
        r=copy.deepcopy(s.requests["q0"]);r.request_id="future";r.arrival_time_ms=20
        with self.assertRaisesRegex(ServingPlanError,"future"):
            inc.upsert(r,10)

    def test_cross_replica_rejected(self):
        s=state(1);inc=IncrementalSchedulerState(s)
        r=copy.deepcopy(s.requests["q0"]);r.replica_id="r1"
        with self.assertRaisesRegex(ServingPlanError,"cross-replica"):
            inc.upsert(r,10)

    def test_summary_delta_does_not_mutate_base(self):
        base=IncrementalSchedulerState(state()).snapshot();before=copy.deepcopy(base)
        changed=SummaryDelta(remaining_prefill_delta=-4,elapsed_ms=2).apply(base)
        self.assertEqual(base,before);self.assertNotEqual(changed,base)

    def test_candidate_deltas_are_isolated(self):
        base=IncrementalSchedulerState(state()).snapshot()
        a=SummaryDelta(remaining_prefill_delta=-4).apply(base)
        b=SummaryDelta(remaining_decode_delta=-2).apply(base)
        self.assertNotEqual(a.total_remaining_prefill,b.total_remaining_prefill)
        self.assertEqual(base.total_remaining_prefill,
                         IncrementalSchedulerState(state()).snapshot().total_remaining_prefill)

    def test_frozen_model_detects_mutation(self):
        m=model()
        object.__setattr__(m,"default_policy","decode_first")
        with self.assertRaisesRegex(ServingPlanError,"changed after freeze"):
            m.validate()

    def test_unknown_feature_rejected(self):
        m=model({"future_arrival_count":1})
        with self.assertRaisesRegex(ServingPlanError,"unknown"):
            m.validate()

    def test_selector_uses_existing_policy(self):
        plan=RankingSelectorV2(model()).select(
            IncrementalSchedulerState(state()).snapshot())
        self.assertEqual(plan.policy_id,"chunked_balanced")

    def test_confidence_tie_uses_stable_default(self):
        m=model(intercepts={p:1.0 for p in
            ("decode_first","prefill_first","chunked_balanced","slo_aware")})
        plan=RankingSelectorV2(m).select(
            IncrementalSchedulerState(state()).snapshot())
        self.assertTrue(plan.uncertain);self.assertTrue(plan.used_default)

    def test_hysteresis_retains_near_tie(self):
        m=model(intercepts={"decode_first":1.0,"prefill_first":4.0,
            "chunked_balanced":1.03,"slo_aware":5.0},
            equivalence_margin=0.0,hysteresis_margin=.05)
        sel=RankingSelectorV2(m);sel.current_policy="chunked_balanced"
        plan=sel.select(IncrementalSchedulerState(state()).snapshot())
        self.assertTrue(plan.retained_by_hysteresis)

    def test_latency_observations_are_past_only(self):
        inc=IncrementalSchedulerState(state())
        inc.latency.observe("mixed",3.5)
        self.assertEqual(inc.snapshot().recent_mixed_ms,3.5)

    def test_invalid_latency_rejected(self):
        inc=IncrementalSchedulerState(state())
        with self.assertRaises(ServingPlanError):
            inc.latency.observe("decode",-1)

    def test_completed_history_not_in_frontier(self):
        s=state(1);r=s.requests["q0"];r.phase="FINISHED"
        r.prefill_completed_tokens=r.uncached_prompt_tokens
        r.decode_completed_tokens=r.expected_output_tokens
        x=IncrementalSchedulerState(s).snapshot()
        self.assertFalse(x.frontier);self.assertEqual(x.history_record_count,1)


if __name__=="__main__":
    unittest.main()
