#include "attention_fp32.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
HirAttentionStatus err(int code, const char *msg) { return {code, msg}; }
bool mul(size_t a, size_t b, size_t &out) {
  if (a && b > std::numeric_limits<size_t>::max() / a) return false;
  out = a * b; return true;
}
HirAttentionStatus run(const float *q, size_t qc, const float *k, size_t kc,
                       const float *v, size_t vc, float *o, size_t oc,
                       float *ws, size_t wc, int64_t b, int64_t h, int64_t ql,
                       int64_t cl, int64_t d, bool prefill) {
  if (!q || !k || !v || !o || !ws) return err(1, "null_buffer");
  if (b <= 0 || h <= 0 || ql <= 0 || cl <= 0 || d <= 0) return err(2, "invalid_dimension");
  if ((prefill && (ql <= 1 || ql != cl)) || (!prefill && ql != 1)) return err(3, "phase_shape_mismatch");
  size_t bh, qn, kn;
  if (!mul(size_t(b), size_t(h), bh) || !mul(bh, size_t(ql), qn) ||
      !mul(qn, size_t(d), qn) || !mul(bh, size_t(cl), kn) || !mul(kn, size_t(d), kn))
    return err(4, "size_overflow");
  if (qc < qn || oc < qn || kc < kn || vc < kn) return err(5, "insufficient_buffer");
  if (wc < size_t(cl)) return err(6, "insufficient_workspace");
  const float scale = 1.0f / std::sqrt(float(d));
  for (int64_t bi = 0; bi < b; ++bi) for (int64_t hi = 0; hi < h; ++hi)
    for (int64_t qi = 0; qi < ql; ++qi) {
      const int64_t valid = prefill ? qi + 1 : cl;
      float mx = -std::numeric_limits<float>::infinity();
      const size_t qbase = (((size_t(bi)*h+hi)*ql+qi)*d);
      for (int64_t ci=0; ci<valid; ++ci) {
        const size_t kbase=(((size_t(bi)*h+hi)*cl+ci)*d); float s=0;
        for (int64_t di=0; di<d; ++di) s += q[qbase+di]*k[kbase+di];
        ws[ci]=s*scale; mx=std::max(mx,ws[ci]);
      }
      float sum=0; for(int64_t ci=0;ci<valid;++ci){ws[ci]=std::exp(ws[ci]-mx);sum+=ws[ci];}
      for(int64_t di=0;di<d;++di){float x=0;for(int64_t ci=0;ci<valid;++ci){
        size_t vb=(((size_t(bi)*h+hi)*cl+ci)*d);x+=(ws[ci]/sum)*v[vb+di];}o[qbase+di]=x;}
    }
  return {0, "ok"};
}
}
extern "C" const char *hir_attention_artifact_version(){return "hir.cpu_attention.v1";}
extern "C" const char *hir_contiguous_kv_artifact_version(){return "hir.contiguous_kv.v1";}
extern "C" HirAttentionStatus hir_contiguous_kv_initialize(float*k,size_t kc,float*v,size_t vc,int64_t b,int64_t h,int64_t c,int64_t d){
  size_t n,x;if(!k||!v)return err(1,"null_buffer");if(b<=0||h<=0||c<=0||d<=0)return err(2,"invalid_dimension");
  if(!mul(size_t(b),size_t(h),n)||!mul(n,size_t(c),n)||!mul(n,size_t(d),n))return err(4,"size_overflow");
  if(kc<n||vc<n)return err(5,"insufficient_buffer");std::fill(k,k+n,0.0f);std::fill(v,v+n,0.0f);return {0,"ok"};}
extern "C" HirAttentionStatus hir_contiguous_kv_prefill_write(float*kc,size_t kcc,float*vc,size_t vcc,const float*k,size_t kn,const float*v,size_t vn,int64_t b,int64_t h,int64_t t,int64_t c,int64_t d){
  if(!kc||!vc||!k||!v)return err(1,"null_buffer");if(t<=0||t>c)return err(2,"invalid_prefill_tokens");size_t cacheN,inputN;
  if(!mul(size_t(b),size_t(h),cacheN)||!mul(cacheN,size_t(c),cacheN)||!mul(cacheN,size_t(d),cacheN)||!mul(size_t(b),size_t(h),inputN)||!mul(inputN,size_t(t),inputN)||!mul(inputN,size_t(d),inputN))return err(4,"size_overflow");
  if(kcc<cacheN||vcc<cacheN||kn<inputN||vn<inputN)return err(5,"insufficient_buffer");
  for(int64_t bi=0;bi<b;++bi)for(int64_t hi=0;hi<h;++hi)for(int64_t ti=0;ti<t;++ti)for(int64_t di=0;di<d;++di){size_t src=(((size_t(bi)*h+hi)*t+ti)*d+di),dst=(((size_t(bi)*h+hi)*c+ti)*d+di);kc[dst]=k[src];vc[dst]=v[src];}return {0,"ok"};}
extern "C" HirAttentionStatus hir_contiguous_kv_append(float*kc,size_t kcc,float*vc,size_t vcc,const float*k,size_t kn,const float*v,size_t vn,int64_t b,int64_t h,int64_t idx,int64_t c,int64_t d){
  if(!kc||!vc||!k||!v)return err(1,"null_buffer");if(idx<0||idx>=c)return err(2,"append_out_of_capacity");size_t cacheN,inputN;
  if(!mul(size_t(b),size_t(h),cacheN)||!mul(cacheN,size_t(c),cacheN)||!mul(cacheN,size_t(d),cacheN)||!mul(size_t(b),size_t(h),inputN)||!mul(inputN,size_t(d),inputN))return err(4,"size_overflow");
  if(kcc<cacheN||vcc<cacheN||kn<inputN||vn<inputN)return err(5,"insufficient_buffer");for(int64_t bi=0;bi<b;++bi)for(int64_t hi=0;hi<h;++hi)for(int64_t di=0;di<d;++di){size_t src=((size_t(bi)*h+hi)*d+di),dst=(((size_t(bi)*h+hi)*c+idx)*d+di);kc[dst]=k[src];vc[dst]=v[src];}return {0,"ok"};}
extern "C" HirAttentionStatus hir_contiguous_kv_reset(float*k,size_t kc,float*v,size_t vc,int64_t b,int64_t h,int64_t c,int64_t d){return hir_contiguous_kv_initialize(k,kc,v,vc,b,h,c,d);}
extern "C" HirAttentionStatus hir_cpu_attention_prefill_fp32(const float*q,size_t qc,const float*k,size_t kc,const float*v,size_t vc,float*o,size_t oc,float*w,size_t wc,int64_t b,int64_t h,int64_t ql,int64_t cl,int64_t d){return run(q,qc,k,kc,v,vc,o,oc,w,wc,b,h,ql,cl,d,true);}
extern "C" HirAttentionStatus hir_cpu_attention_decode_fp32(const float*q,size_t qc,const float*k,size_t kc,const float*v,size_t vc,float*o,size_t oc,float*w,size_t wc,int64_t b,int64_t h,int64_t ql,int64_t cl,int64_t d){return run(q,qc,k,kc,v,vc,o,oc,w,wc,b,h,ql,cl,d,false);}
extern "C" HirAttentionStatus hir_cpu_attention_decode_contiguous_kv_fp32(const float*q,size_t qc,const float*k,size_t kc,const float*v,size_t vc,float*o,size_t oc,float*w,size_t wc,int64_t b,int64_t h,int64_t valid,int64_t capacity,int64_t d){
  if(valid<=0||valid>capacity)return err(2,"invalid_valid_tokens");
  // run() receives the physical capacity as its K/V stride; a dedicated loop
  // is required because its logical context stride otherwise equals valid.
  if(!q||!k||!v||!o||!w)return err(1,"null_buffer");size_t cacheN,qn;
  if(!mul(size_t(b),size_t(h),cacheN)||!mul(cacheN,size_t(capacity),cacheN)||!mul(cacheN,size_t(d),cacheN)||!mul(size_t(b),size_t(h),qn)||!mul(qn,size_t(d),qn))return err(4,"size_overflow");
  if(kc<cacheN||vc<cacheN||qc<qn||oc<qn||wc<size_t(valid))return err(5,"insufficient_buffer");float scale=1.0f/std::sqrt(float(d));
  for(int64_t bi=0;bi<b;++bi)for(int64_t hi=0;hi<h;++hi){size_t qb=((size_t(bi)*h+hi)*d);float mx=-std::numeric_limits<float>::infinity();for(int64_t ci=0;ci<valid;++ci){size_t kb=(((size_t(bi)*h+hi)*capacity+ci)*d);float s=0;for(int64_t di=0;di<d;++di)s+=q[qb+di]*k[kb+di];w[ci]=s*scale;mx=std::max(mx,w[ci]);}float sum=0;for(int64_t ci=0;ci<valid;++ci){w[ci]=std::exp(w[ci]-mx);sum+=w[ci];}for(int64_t di=0;di<d;++di){float x=0;for(int64_t ci=0;ci<valid;++ci){size_t vb=(((size_t(bi)*h+hi)*capacity+ci)*d);x+=(w[ci]/sum)*v[vb+di];}o[qb+di]=x;}}return {0,"ok"};}
