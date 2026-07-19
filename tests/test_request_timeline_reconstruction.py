import unittest
from deployment.request_timeline_reconstruction import reconstruct_run,error_summary
from deployment.serving_execution import ServingPlanError
def run():
 return {"run_id":"r","trace_id":"t","host_wall_start_ns":1000,
  "steps":[{"step_id":0,"step_start_ns":1000,"step_end_ns":3000,
   "execution":{"scheduled_items":[{"request_id":"q","phase":"prefill",
    "token_start":0,"token_count":2,"callback_start_ns":1200,"commit_ns":1500}]}},
   {"step_id":1,"step_start_ns":3000,"step_end_ns":5000,
   "execution":{"scheduled_items":[{"request_id":"q","phase":"decode",
    "token_start":0,"token_count":1,"callback_start_ns":3200,"commit_ns":4000}]}},
   {"step_id":2,"step_start_ns":5000,"step_end_ns":7000,
   "execution":{"scheduled_items":[{"request_id":"q","phase":"decode",
    "token_start":1,"token_count":1,"callback_start_ns":5200,"commit_ns":6000}]}}],
  "direct_request_timestamps":{"q":{"arrival_ms":0.0,"first_token_ms":.003,
   "completion_ms":.005,"decode_token_times_ms":[.003,.005]}}}
class ReconstructionTests(unittest.TestCase):
 def test_exact(self):
  x=reconstruct_run(run());self.assertTrue(error_summary([x]).get("passed"))
 def test_duplicate_step(self):
  x=run();x["steps"].append(x["steps"][0])
  with self.assertRaisesRegex(ServingPlanError,"double-counts"):reconstruct_run(x)
 def test_missing_origin(self):
  x=run();del x["host_wall_start_ns"]
  with self.assertRaisesRegex(ServingPlanError,"origin"):reconstruct_run(x)
 def test_item_outside_step(self):
  x=run();x["steps"][0]["execution"]["scheduled_items"][0]["commit_ns"]=4000
  with self.assertRaisesRegex(ServingPlanError,"outside"):reconstruct_run(x)
 def test_missing_request(self):
  x=run();x["steps"]=[]
  with self.assertRaisesRegex(ServingPlanError,"omits request"):reconstruct_run(x)
 def test_frozen_tolerance(self):
  with self.assertRaisesRegex(ServingPlanError,"tolerance"):error_summary([],0)
if __name__=="__main__":unittest.main()
