import unittest
from deployment.scheduler_observability import (
  RankingSelectorV3,StepInstrumentation,freeze_v3)
from deployment.scheduler_ranking_v2 import IncrementalSchedulerState
from deployment.serving_execution import ServingPlanError
from deployment.serving_scheduler import (
  ReplicaSchedulerState,RequestExecutionState,SchedulerProfile)

def summary():
 s=ReplicaSchedulerState("r",SchedulerProfile(),clock_ms=10)
 s.ingest(RequestExecutionState("q","p","r",0,16,0,4,phase="PREFILL"))
 return IncrementalSchedulerState(s).snapshot()
class ObservabilityTests(unittest.TestCase):
 def test_accounting(self):
  x=StepInstrumentation("t","r","p","v","r",0,1,0,{},{});x.add_region("model_execution",0,900_000)
  self.assertLess(x.finish(1_000_000)["unaccounted_ms"],.11)
 def test_overlap_rejected(self):
  x=StepInstrumentation("t","r","p","v","r",0,1,0,{},{});x.add_region("model_setup",0,10);x.add_region("model_execution",5,20)
  with self.assertRaisesRegex(ServingPlanError,"overlap"):x.finish(20)
 def test_unaccounted_rejected(self):
  x=StepInstrumentation("t","r","p","v","r",0,1,0,{},{});x.add_region("model_execution",0,1)
  with self.assertRaisesRegex(ServingPlanError,"unaccounted"):x.finish(2_000_000,.01,.01)
 def test_missing_version_is_explicit(self):
  with self.assertRaises(TypeError):StepInstrumentation("t","r","p","v","r",0)
 def test_freeze_mutation(self):
  c=freeze_v3();object.__setattr__(c,"default_policy","decode_first")
  with self.assertRaisesRegex(ServingPlanError,"changed"):c.validate()
 def test_static_decision(self):
  self.assertIn(RankingSelectorV3(freeze_v3()).select(summary()).policy_id,
                ("decode_first","prefill_first","chunked_balanced","slo_aware"))
 def test_adaptive_does_not_mutate_static(self):
  c=freeze_v3();a=RankingSelectorV3(c,True);a.select(summary());c.validate()
 def test_hysteresis_stable(self):
  s=RankingSelectorV3(freeze_v3());a=s.select(summary());b=s.select(summary())
  self.assertEqual(a.policy_id,b.policy_id)
 def test_future_measurement_not_an_input(self):
  self.assertNotIn("future",RankingSelectorV3.select.__code__.co_varnames)

if __name__=="__main__":unittest.main()
