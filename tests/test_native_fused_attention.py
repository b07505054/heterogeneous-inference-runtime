from pathlib import Path
import subprocess
import pytest, torch
from deployment.native_fused_attention import ABI_VERSION, NativeFusedAttentionLibrary, sha256

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope="module")
def lib(tmp_path_factory):
 d=tmp_path_factory.mktemp("native_fused");so=d/"lib.so"
 subprocess.run(["g++","-O3","-std=c++17","-fPIC","-shared","-fno-tree-vectorize",
  "-mavx2","-mfma",str(ROOT/"native/fused_online_attention.cpp"),"-o",str(so)],check=True)
 return NativeFusedAttentionLibrary(so,sha256(so),ABI_VERSION)
@pytest.mark.parametrize("impl",["native_scalar","native_avx2"])
@pytest.mark.parametrize("q,k",[(1,7),(7,7),(11,37),(37,63)])
def test_native_matches_reference(lib,impl,q,k):
 torch.manual_seed(q*100+k);Q=torch.randn(1,14,q,64);K=torch.randn(1,2,k,64);V=torch.randn_like(K)
 mask=torch.tril(torch.ones(q,k,dtype=torch.bool),diagonal=k-q)
 ref=torch.nn.functional.scaled_dot_product_attention(Q,K.repeat_interleave(7,1),V.repeat_interleave(7,1),attn_mask=mask)
 out,stats=lib.run(impl,Q,K,V,64**-.5,1,32,total_query_heads=14)
 torch.testing.assert_close(out,ref,atol=2e-5,rtol=2e-5)
 assert stats["temporary_bytes"]==384 and stats["allocations_per_query_row"]==0
 assert not stats["full_score_materialized"] and not stats["full_probability_materialized"]
def test_invalid_gqa_fails(lib):
 q=torch.randn(1,13,1,64);k=torch.randn(1,2,7,64)
 with pytest.raises(RuntimeError,match="invalid_gqa"):
  lib.run("native_scalar",q,k,k,64**-.5,1,32,total_query_heads=13)
def test_artifact_hash_is_fail_closed(tmp_path):
 p=tmp_path/"bad.so";p.write_bytes(b"x")
 with pytest.raises(RuntimeError,match="hash"):
  NativeFusedAttentionLibrary(p,"0"*64,ABI_VERSION)
