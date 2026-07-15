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
extern "C" HirAttentionStatus hir_cpu_attention_decode_contiguous_kv_reordered_fp32(const float*q,size_t qc,const float*k,size_t kc,const float*v,size_t vc,float*o,size_t oc,float*w,size_t wc,int64_t b,int64_t h,int64_t valid,int64_t capacity,int64_t d){
  if(valid<=0||valid>capacity)return err(2,"invalid_valid_tokens");if(!q||!k||!v||!o||!w)return err(1,"null_buffer");size_t cacheN,qn;
  if(!mul(size_t(b),size_t(h),cacheN)||!mul(cacheN,size_t(capacity),cacheN)||!mul(cacheN,size_t(d),cacheN)||!mul(size_t(b),size_t(h),qn)||!mul(qn,size_t(d),qn))return err(4,"size_overflow");
  if(kc<cacheN||vc<cacheN||qc<qn||oc<qn||wc<size_t(valid))return err(5,"insufficient_buffer");float scale=1.0f/std::sqrt(float(d));
  for(int64_t bi=0;bi<b;++bi)for(int64_t hi=0;hi<h;++hi){size_t qb=((size_t(bi)*h+hi)*d),base=((size_t(bi)*h+hi)*capacity*d);float mx=-std::numeric_limits<float>::infinity();
    for(int64_t t=0;t<valid;++t){const float*kt=k+base+size_t(t)*d;float s=0;for(int64_t di=0;di<d;++di)s+=q[qb+di]*kt[di];w[t]=s*scale;mx=std::max(mx,w[t]);}
    float sum=0;for(int64_t t=0;t<valid;++t){w[t]=std::exp(w[t]-mx);sum+=w[t];}std::fill(o+qb,o+qb+d,0.0f);
    for(int64_t t=0;t<valid;++t){const float weight=w[t]/sum;const float*vt=v+base+size_t(t)*d;for(int64_t di=0;di<d;++di)o[qb+di]+=weight*vt[di];}}
  return {0,"ok"};}
// Retain the historical benchmark symbol so existing validation artifacts and
// runners remain reproducible. Production plans use the distinct entry point
// above and the Runtime never dispatches through this alias.
extern "C" HirAttentionStatus hir_cpu_attention_decode_contiguous_kv_reordered_control_fp32(const float*q,size_t qc,const float*k,size_t kc,const float*v,size_t vc,float*o,size_t oc,float*w,size_t wc,int64_t b,int64_t h,int64_t valid,int64_t capacity,int64_t d){
  return hir_cpu_attention_decode_contiguous_kv_reordered_fp32(q,qc,k,kc,v,vc,o,oc,w,wc,b,h,valid,capacity,d);
}
extern "C" const char *hir_paged_kv_artifact_version(){return "hir.paged_kv.v1";}
namespace { bool paged_addr(int64_t token,int64_t pages,int64_t heads,int64_t pt,int64_t d,const int32_t*bt,size_t btc,int32_t sentinel,int64_t head,size_t&addr){size_t block=size_t(token/pt);if(block>=btc)return false;int32_t page=bt[block];if(page==sentinel||page<0||page>=pages)return false;addr=(((size_t(page)*heads+head)*pt+size_t(token%pt))*d);return true;} }
extern "C" HirAttentionStatus hir_paged_kv_initialize(float*k,size_t kc,float*v,size_t vc,int32_t*bt,size_t btc,int64_t pages,int64_t h,int64_t pt,int64_t d,int32_t sentinel){size_t n;if(!k||!v||!bt)return err(1,"null_buffer");if(pages<=0||h<=0||pt<=0||d<=0)return err(2,"invalid_dimension");if(!mul(size_t(pages),size_t(h),n)||!mul(n,size_t(pt),n)||!mul(n,size_t(d),n))return err(4,"size_overflow");if(kc<n||vc<n)return err(5,"insufficient_buffer");std::fill(k,k+n,0);std::fill(v,v+n,0);std::fill(bt,bt+btc,sentinel);return {0,"ok"};}
extern "C" HirAttentionStatus hir_paged_kv_reset(float*k,size_t kc,float*v,size_t vc,int32_t*bt,size_t btc,int64_t pages,int64_t h,int64_t pt,int64_t d,int32_t sentinel){return hir_paged_kv_initialize(k,kc,v,vc,bt,btc,pages,h,pt,d,sentinel);}
extern "C" HirAttentionStatus hir_paged_kv_prefill_write(float*kp,size_t kc,float*vp,size_t vc,const int32_t*bt,size_t btc,const float*k,size_t kn,const float*v,size_t vn,int64_t tokens,int64_t pages,int64_t h,int64_t pt,int64_t d,int32_t sentinel){if(!kp||!vp||!bt||!k||!v)return err(1,"null_buffer");size_t poolN,inputN;if(!mul(size_t(pages),size_t(h),poolN)||!mul(poolN,size_t(pt),poolN)||!mul(poolN,size_t(d),poolN)||!mul(size_t(h),size_t(tokens),inputN)||!mul(inputN,size_t(d),inputN))return err(4,"size_overflow");if(kc<poolN||vc<poolN||kn<inputN||vn<inputN)return err(5,"insufficient_buffer");for(int64_t hi=0;hi<h;++hi)for(int64_t t=0;t<tokens;++t){size_t dst;if(!paged_addr(t,pages,h,pt,d,bt,btc,sentinel,hi,dst))return err(7,"invalid_block_table");size_t src=((size_t(hi)*tokens+t)*d);std::copy(k+src,k+src+d,kp+dst);std::copy(v+src,v+src+d,vp+dst);}return {0,"ok"};}
extern "C" HirAttentionStatus hir_paged_kv_append(float*kp,size_t kc,float*vp,size_t vc,const int32_t*bt,size_t btc,const float*k,size_t kn,const float*v,size_t vn,int64_t idx,int64_t pages,int64_t h,int64_t pt,int64_t d,int32_t sentinel){if(!kp||!vp||!bt||!k||!v)return err(1,"null_buffer");size_t poolN,inputN;if(!mul(size_t(pages),size_t(h),poolN)||!mul(poolN,size_t(pt),poolN)||!mul(poolN,size_t(d),poolN)||!mul(size_t(h),size_t(d),inputN))return err(4,"size_overflow");if(kc<poolN||vc<poolN||kn<inputN||vn<inputN)return err(5,"insufficient_buffer");for(int64_t hi=0;hi<h;++hi){size_t dst;if(!paged_addr(idx,pages,h,pt,d,bt,btc,sentinel,hi,dst))return err(7,"invalid_block_table");size_t src=size_t(hi)*d;std::copy(k+src,k+src+d,kp+dst);std::copy(v+src,v+src+d,vp+dst);}return {0,"ok"};}
extern "C" HirAttentionStatus hir_cpu_attention_decode_paged_kv_fp32(const float*q,size_t qc,const float*k,size_t kc,const float*v,size_t vc,const int32_t*bt,size_t btc,float*o,size_t oc,float*w,size_t wc,int64_t valid,int64_t pages,int64_t h,int64_t pt,int64_t d,int32_t sentinel){if(!q||!k||!v||!bt||!o||!w)return err(1,"null_buffer");size_t poolN,qn;if(valid<=0||!mul(size_t(pages),size_t(h),poolN)||!mul(poolN,size_t(pt),poolN)||!mul(poolN,size_t(d),poolN)||!mul(size_t(h),size_t(d),qn))return err(4,"size_overflow");if(kc<poolN||vc<poolN||qc<qn||oc<qn||wc<size_t(valid))return err(5,"insufficient_buffer");float scale=1/std::sqrt(float(d));for(int64_t hi=0;hi<h;++hi){size_t qb=size_t(hi)*d;float mx=-std::numeric_limits<float>::infinity();for(int64_t t=0;t<valid;++t){size_t kb;if(!paged_addr(t,pages,h,pt,d,bt,btc,sentinel,hi,kb))return err(7,"invalid_block_table");float s=0;for(int64_t di=0;di<d;++di)s+=q[qb+di]*k[kb+di];w[t]=s*scale;mx=std::max(mx,w[t]);}float sum=0;for(int64_t t=0;t<valid;++t){w[t]=std::exp(w[t]-mx);sum+=w[t];}for(int64_t di=0;di<d;++di){float x=0;for(int64_t t=0;t<valid;++t){size_t vb;paged_addr(t,pages,h,pt,d,bt,btc,sentinel,hi,vb);x+=(w[t]/sum)*v[vb+di];}o[qb+di]=x;}}return {0,"ok"};}

extern "C" HirAttentionStatus hir_cpu_attention_decode_paged_kv_page_major_fp32(
    const float*q,size_t qc,const float*k,size_t kc,const float*v,size_t vc,
    const int32_t*bt,size_t btc,int32_t*physical,size_t physicalCount,
    float*o,size_t oc,float*w,size_t wc,
    int64_t valid,int64_t pages,int64_t h,int64_t pt,int64_t d,int32_t sentinel) {
  if(!q||!k||!v||!bt||!physical||!o||!w)return err(1,"null_buffer");
  size_t poolN,qn;
  if(valid<=0||pt<=0||!mul(size_t(pages),size_t(h),poolN)||
     !mul(poolN,size_t(pt),poolN)||!mul(poolN,size_t(d),poolN)||
     !mul(size_t(h),size_t(d),qn))return err(4,"size_overflow");
  if(kc<poolN||vc<poolN||qc<qn||oc<qn||wc<size_t(valid))
    return err(5,"insufficient_buffer");
  const size_t logicalPages=(size_t(valid)+size_t(pt)-1)/size_t(pt);
  if(logicalPages>btc||logicalPages>physicalCount)return err(7,"invalid_block_table");
  // Cache the physical-page identity once. The timed arithmetic loops never
  // divide/modulo logical token indices and never reread the block table.
  for(size_t block=0;block<logicalPages;++block){
    int32_t page=bt[block];
    if(page==sentinel||page<0||page>=pages)return err(7,"invalid_block_table");
    physical[block]=page;
  }
  const float scale=1/std::sqrt(float(d));
  const size_t pageStride=size_t(h)*size_t(pt)*size_t(d);
  const size_t headStride=size_t(pt)*size_t(d);
  for(int64_t hi=0;hi<h;++hi){
    const size_t qb=size_t(hi)*size_t(d);float mx=-std::numeric_limits<float>::infinity();
    size_t logical=0;
    for(size_t block=0;block<logicalPages;++block){
      const float* kbase=k+size_t(physical[block])*pageStride+size_t(hi)*headStride;
      const size_t inPage=std::min(size_t(pt),size_t(valid)-logical);
      for(size_t offset=0;offset<inPage;++offset,++logical){
        const float* kt=kbase+offset*size_t(d);float s=0;
        for(int64_t di=0;di<d;++di)s+=q[qb+size_t(di)]*kt[di];
        w[logical]=s*scale;mx=std::max(mx,w[logical]);
      }
    }
    float sum=0;for(int64_t t=0;t<valid;++t){w[t]=std::exp(w[t]-mx);sum+=w[t];}
    std::fill(o+qb,o+qb+size_t(d),0.0f);logical=0;
    for(size_t block=0;block<logicalPages;++block){
      const float* vbase=v+size_t(physical[block])*pageStride+size_t(hi)*headStride;
      const size_t inPage=std::min(size_t(pt),size_t(valid)-logical);
      for(size_t offset=0;offset<inPage;++offset,++logical){
        const float weight=w[logical]/sum;const float* vt=vbase+offset*size_t(d);
        for(int64_t di=0;di<d;++di)o[qb+size_t(di)]+=weight*vt[di];
      }
    }
  }
  return {0,"ok"};
}
