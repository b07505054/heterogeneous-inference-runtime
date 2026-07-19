import unittest
from deployment.scheduler_policy_v4 import (
 RankingSelectorV4,freeze_v4,PAIRWISE_V4,RISK_V4)
from deployment.scheduler_ranking_v2 import IncrementalSchedulerState
from deployment.serving_execution import ServingPlanError
from deployment.serving_scheduler import ReplicaSchedulerState,RequestExecutionState,SchedulerProfile
def summary(n=8):
 s=ReplicaSchedulerState("r",SchedulerProfile(),clock_ms=1)
 for i in range(n):s.ingest(RequestExecutionState(str(i),str(i),"r",0,32,0,4,phase="PREFILL"))
 return IncrementalSchedulerState(s,frontier_size=8).snapshot()
class V4Tests(unittest.TestCase):
 def test_freeze(self):
  c=freeze_v4(PAIRWISE_V4);c.validate()
 def test_mutation(self):
  c=freeze_v4(PAIRWISE_V4);object.__setattr__(c,"decision_margin",0)
  with self.assertRaisesRegex(ServingPlanError,"changed"):c.validate()
 def test_pairwise_legal(self):
  self.assertIn(RankingSelectorV4(freeze_v4(PAIRWISE_V4)).select(summary()).policy_id,
   ("decode_first","prefill_first","chunked_balanced","slo_aware"))
 def test_risk_default(self):
  x=RankingSelectorV4(freeze_v4(RISK_V4),True).select(summary(32))
  self.assertTrue(x.default_used);self.assertEqual(x.policy_id,"decode_first")
 def test_uncertainty_present(self):
  self.assertEqual(len(RankingSelectorV4(freeze_v4(PAIRWISE_V4)).select(summary()).uncertainty),4)
 def test_no_adaptive_state(self):
  self.assertFalse(hasattr(RankingSelectorV4(freeze_v4(PAIRWISE_V4)),"update"))
if __name__=="__main__":unittest.main()
