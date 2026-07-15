import copy,json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from deployment.execution_plan.executorch_candidate_integration import *
from deployment.execution_plan.slice3c_target_selection import *
REPO=Path(__file__).resolve().parents[1];SH=['37x41x29','64x64x64','128x128x128','256x256x256']
class T(unittest.TestCase):
 @classmethod
 def setUpClass(c):
  c.tmp=tempfile.TemporaryDirectory();c.root=Path(c.tmp.name);(c.root/'artifacts').mkdir();(c.root/'proofs').mkdir()
  c.caps=load_codegen_capabilities(json.load(open(REPO/'configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json')));c.all=enumerate_complete_candidates();c.et=[x for x in c.all if x.backend=='executorch_xnnpack'];c.by={x.candidate_id:x for x in c.all};c.shape={'M':64,'N':64,'K':64}
  for name,data in [('runner',b'runner'),('fp32.pte',b'fp32'),('int8.pte',b'int8'),('manifest.json',b'manifest')]: (c.root/'artifacts'/name).write_bytes(data)
  for precision in ('fp32','int8'):
   p={'delegate':'xnnpack','precision':precision,'delegated_node_count':1,'portable_fallback_nodes':[]};p['proof_sha256']=canonical_hash(p);(c.root/'proofs'/f'{precision}.json').write_text(json.dumps(p))
  c.e=c.make_evidence(c.shape,{'fp32':.016019,'int8_t1':.009204,'int8_t4':.008583,'custom_fp32':.20024,'custom_int8':.041473575})
 @classmethod
 def tearDownClass(c):c.tmp.cleanup()
 @classmethod
 def make_evidence(c,shape,lat):
  def h(p):return sha256_file(c.root/p)
  out={FP32_CANDIDATE_ID:{'candidate_id':FP32_CANDIDATE_ID,'shape':shape,'steady_state_invoke_median_ms':lat['custom_fp32'],'correctness_metrics':{'cosine_similarity':1,'relative_l2_error':0},'thread_count':1,'artifact_bytes':79960,'runtime_package_bytes':79960,'peak_rss_kib':3984},INT8_PACKED_A76_DOTPROD_CANDIDATE_ID:{'candidate_id':INT8_PACKED_A76_DOTPROD_CANDIDATE_ID,'shape':shape,'steady_state_invoke_median_ms':lat['custom_int8'],'correctness_metrics':{'cosine_similarity':.99998,'relative_l2_error':.006},'thread_count':1,'artifact_bytes':75896,'runtime_package_bytes':75896,'peak_rss_kib':3984}}
  for candidate,precision,threads,key in [(c.et[0],'fp32',1,'fp32'),(c.et[1],'int8',1,'int8_t1'),(c.et[2],'int8',4,'int8_t4')]:
   proof=json.load(open(c.root/'proofs'/f'{precision}.json'));e={'candidate_id':candidate.candidate_id,'backend':'executorch_xnnpack','runtime':'executorch','delegate':'xnnpack','precision':precision,'quantization_scheme':candidate.quantization_scheme,'thread_count':threads,'shape':shape,'target_id':'raspberry-pi5-cortex-a76-cpu','runner_ref':'artifacts/runner','runner_sha256':h('artifacts/runner'),'pte_ref':f'artifacts/{precision}.pte','pte_sha256':h(f'artifacts/{precision}.pte'),'workload_manifest_ref':'artifacts/manifest.json','workload_manifest_sha256':h('artifacts/manifest.json'),'delegation_proof_ref':f'proofs/{precision}.json','delegation_proof_sha256':proof['proof_sha256'],'measurement_ref':f'measurements/{candidate.candidate_id}.json','correctness_metrics':{'cosine_similarity':.99997 if precision=='int8' else 1,'relative_l2_error':.007 if precision=='int8' else 0},'steady_state_invoke_median_ms':lat[key],'session_count':5,'sample_count_per_session':100,'between_session_cv':.01,'variability_threshold':.12,'variability_gate_passed':True,'randomized_order_provenance':[1,2,3,4,5],'boundary_classification':'slice3f_already_loaded_boundary','runtime_package_bytes':9454800,'artifact_bytes':100,'peak_rss_kib':6448,'artifact_root':str(c.root)};e['evidence_sha256']=canonical_hash(e);out[candidate.candidate_id]=e
  return out
 def seal(self,e):e.pop('evidence_sha256',None);e['evidence_sha256']=canonical_hash(e);return e
 def test_enumerated_active_path(self):self.assertEqual(len(self.et),3)
 def test_candidate_ids_unique(self):self.assertEqual(len({x.candidate_id for x in self.et}),3)
 def test_runtime_package_legal(self):self.assertEqual(validate_external_evidence(self.et[1],self.caps,self.e[self.et[1].candidate_id],shape=self.shape,policy=SelectionPolicy()),[])
 def test_runner_hash_mismatch(self):
  e=self.seal(copy.deepcopy(self.e[self.et[1].candidate_id]));e['runner_sha256']='0'*64;e=self.seal(e);self.assertIn('runner_hash_mismatch',validate_external_evidence(self.et[1],self.caps,e,shape=self.shape,policy=SelectionPolicy()))
 def test_pte_hash_mismatch(self):
  e=self.seal(copy.deepcopy(self.e[self.et[1].candidate_id]));e['pte_sha256']='0'*64;e=self.seal(e);self.assertIn('pte_hash_mismatch',validate_external_evidence(self.et[1],self.caps,e,shape=self.shape,policy=SelectionPolicy()))
 def test_delegation_proof_mismatch(self):
  e=self.seal(copy.deepcopy(self.e[self.et[1].candidate_id]));e['delegation_proof_sha256']='0'*64;e=self.seal(e);self.assertIn('delegation_proof_mismatch',validate_external_evidence(self.et[1],self.caps,e,shape=self.shape,policy=SelectionPolicy()))
 def test_missing_xnnpack(self):self.assertIn('missing_xnnpack_capability',validate_external_evidence(self.et[1],replace(self.caps,supports_xnnpack_delegate=False),self.e[self.et[1].candidate_id],shape=self.shape,policy=SelectionPolicy()))
 def test_thread_budget(self):self.assertIn('thread_budget_exceeded',validate_external_evidence(self.et[2],self.caps,self.e[self.et[2].candidate_id],shape=self.shape,policy=SelectionPolicy(max_threads=1)))
 def test_shape_mismatch(self):self.assertIn('shape_mismatch',validate_external_evidence(self.et[1],self.caps,self.e[self.et[1].candidate_id],shape={'M':1,'N':1,'K':1},policy=SelectionPolicy()))
 def test_measurement_identity(self):self.assertNotIn('measurement_identity_mismatch',validate_external_evidence(self.et[1],self.caps,self.e[self.et[1].candidate_id],shape=self.shape,policy=SelectionPolicy()))
 def test_correctness_gate(self):
  e=copy.deepcopy(self.e[self.et[1].candidate_id]);e['correctness_metrics']['cosine_similarity']=0;e=self.seal(e);self.assertIn('correctness_gate_failed',validate_external_evidence(self.et[1],self.caps,e,shape=self.shape,policy=SelectionPolicy()))
 def candidates(self):return [x for x in self.all if x.candidate_id in {FP32_CANDIDATE_ID,INT8_PACKED_A76_DOTPROD_CANDIDATE_ID,*[e.candidate_id for e in self.et]}]
 def test_default_expected_all_shapes(self):
  expected=[EXECUTORCH_XNNPACK_INT8_T1_CANDIDATE_ID]+[EXECUTORCH_XNNPACK_INT8_T4_CANDIDATE_ID]*3
  latencies=[(.006537,.0035375,.005704,.0352035,.024256935),(.016019,.009204,.008583,.20024,.041473575),(.119,.048,.022778,1.56514,.1935629),(.9176665,.307028,.1007595,12.47145,1.115453)]
  for s,x,ls in zip(SH,expected,latencies):m,n,k=map(int,s.split('x'));e=self.make_evidence({'M':m,'N':n,'K':k},dict(zip(('fp32','int8_t1','int8_t4','custom_fp32','custom_int8'),ls)));self.assertEqual(select_complete_candidate(self.candidates(),self.caps,e,shape={'M':m,'N':n,'K':k})['selected_candidate_id'],x)
 def test_one_thread_policy(self):self.assertEqual(select_complete_candidate(self.candidates(),self.caps,self.e,shape=self.shape,policy=SelectionPolicy(max_threads=1))['selected_candidate_id'],EXECUTORCH_XNNPACK_INT8_T1_CANDIDATE_ID)
 def test_unavailable_fallback(self):self.assertEqual(select_complete_candidate(self.candidates(),replace(self.caps,supports_executorch_runtime=False),self.e,shape=self.shape)['selected_candidate_id'],INT8_PACKED_A76_DOTPROD_CANDIDATE_ID)
 def test_size_budget_fallback(self):self.assertEqual(select_complete_candidate(self.candidates(),self.caps,self.e,shape=self.shape,policy=SelectionPolicy(max_runtime_package_size=1_000_000))['selected_candidate_id'],INT8_PACKED_A76_DOTPROD_CANDIDATE_ID)
 def test_plan_contract(self):
  sel=select_complete_candidate(self.candidates(),self.caps,self.e,shape=self.shape);candidate=self.by[sel['selected_candidate_id']];p=build_executorch_execution_plan(candidate,self.e[candidate.candidate_id],sel);self.assertEqual((p['backend'],p['runtime'],p['delegate'],p['thread_count']),('executorch_xnnpack','executorch','xnnpack',4));self.assertIn('runner_sha256',p);self.assertIn('pte_sha256',p)
 def test_plan_stages(self):
  sel=select_complete_candidate(self.candidates(),self.caps,self.e,shape=self.shape);candidate=self.by[sel['selected_candidate_id']];p=build_executorch_execution_plan(candidate,self.e[candidate.candidate_id],sel);self.assertEqual([x['stage_id'] for x in p['execution_stages']],['load_executorch_program','bind_input','execute_xnnpack_delegate','return_output'])
 def test_thread_identity_not_changed(self):self.assertNotEqual(self.et[1].candidate_id,self.et[2].candidate_id)
 def test_precision_identity_not_changed(self):self.assertNotEqual(self.et[0].candidate_id,self.et[1].candidate_id)
 def test_selection_agreement_and_regret(self):
  selected=select_complete_candidate(self.candidates(),self.caps,self.e,shape=self.shape);legal=[x for x in selected['considered_candidates'] if x['legal']];oracle=min(legal,key=lambda x:(x['latency_median_ms'],x['candidate_id']));self.assertEqual(selected['selected_candidate_id'],oracle['candidate_id']);self.assertAlmostEqual((selected['selected_latency_median_ms']-oracle['latency_median_ms'])/oracle['latency_median_ms'],0)
 def test_stable_rejection_vocabulary(self):self.assertIn('pte_hash_mismatch',REJECTION_REASONS);self.assertIn('missing_runtime_package',REJECTION_REASONS)
if __name__=='__main__':unittest.main()
