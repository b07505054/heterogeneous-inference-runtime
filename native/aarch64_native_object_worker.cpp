// Static-shape worker for the Slice-18/19/20 mlir_ciface_memref_f32_v1 ABI.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>

extern "C" {
struct MemRef2D { float *allocated, *aligned; int64_t offset, sizes[2], strides[2]; };
#ifndef M_DIM
#define M_DIM 32
#endif
#ifndef N_DIM
#define N_DIM 32
#endif
#ifndef K_DIM
#define K_DIM 32
#endif
#ifndef ENTRY_POINT
#define ENTRY_POINT _mlir_ciface_matmul_bias_relu_tiled_32x32x32
#endif
void ENTRY_POINT(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
}
static constexpr int M=M_DIM, N=N_DIM, K=K_DIM, GUARD=16, MAX=20000;
static double samples[MAX];
static void fill(std::vector<float>& v, uint32_t s) {
  for (float& x:v) { s=s*1664525u+1013904223u; x=float((s>>8)&65535)/65536.f-.5f; }
}
static MemRef2D desc(std::vector<float>& v, int rows, int cols) {
  return {v.data(),v.data()+GUARD,0,{rows,cols},{cols,1}};
}
static double pct(int n,double p) {
  std::vector<double> x(samples,samples+n); std::sort(x.begin(),x.end());
  return x[size_t(p*double(n-1))];
}
int main(int argc,char**argv) {
  int warm=argc>1?std::atoi(argv[1]):20, n=argc>2?std::atoi(argv[2]):500;
  int batch=argc>3?std::atoi(argv[3]):100;
  if(n<1||n>MAX) return 2;
  const float sentinel=1234567.f;
  std::vector<float>a(M*K+2*GUARD,sentinel),b(K*N+2*GUARD,sentinel),
                    bias(M*N+2*GUARD,sentinel),ref(M*N);
  std::vector<float> av(M*K),bv(K*N),cv(M*N); fill(av,1);fill(bv,2);fill(cv,3);
  std::copy(av.begin(),av.end(),a.begin()+GUARD);
  std::copy(bv.begin(),bv.end(),b.begin()+GUARD);
  std::copy(cv.begin(),cv.end(),bias.begin()+GUARD);
  for(int i=0;i<M;i++)for(int j=0;j<N;j++){float z=0;for(int k=0;k<K;k++)z+=av[i*K+k]*bv[k*N+j];ref[i*N+j]=std::max(0.f,z+cv[i*N+j]);}
  auto ad=desc(a,M,K),bd=desc(b,K,N),cd=desc(bias,M,N); double maxerr=0; bool repeated=true;
  for(int it=0;it<warm+n+2;it++){auto t0=std::chrono::steady_clock::now();
    for(int bch=0;bch<batch;bch++){MemRef2D out{};
      ENTRY_POINT(&out,&ad,&bd,&cd);
      for(int q=0;q<M*N;q++){double e=std::fabs(out.aligned[q]-ref[q]);maxerr=std::max(maxerr,e);if(e>=1e-3)repeated=false;}
      std::free(out.allocated);
    }
    auto t1=std::chrono::steady_clock::now();
    if(it>=warm&&it<warm+n)samples[it-warm]=std::chrono::duration<double,std::milli>(t1-t0).count()/batch;
  }
  bool guards=true;
  for(int i=0;i<GUARD;i++) {
    guards &= a[i]==sentinel && a[M*K+GUARD+i]==sentinel;
    guards &= b[i]==sentinel && b[K*N+GUARD+i]==sentinel;
    guards &= bias[i]==sentinel && bias[M*N+GUARD+i]==sentinel;
  }
  double mean=std::accumulate(samples,samples+n,0.0)/n, var=0;
  for(int i=0;i<n;i++)var+=(samples[i]-mean)*(samples[i]-mean);
  double sd=std::sqrt(var/(n>1?n-1:1));
  auto mm=std::minmax_element(samples,samples+n);
  std::printf("{\"correct\":%s,\"repeated_call_correct\":%s,\"guard_buffers_intact\":%s,\"max_abs_error\":%.9g,\"warmup_samples\":%d,\"sample_count\":%d,\"calls_per_sample\":%d,\"measured_calls\":%d,\"p50_ms\":%.9g,\"p95_ms\":%.9g,\"mean_ms\":%.9g,\"stddev_ms\":%.9g,\"minimum_ms\":%.9g,\"maximum_ms\":%.9g}\n",
    repeated?"true":"false",repeated?"true":"false",guards?"true":"false",maxerr,warm,n,batch,n*batch,pct(n,.50),pct(n,.95),mean,sd,*mm.first,*mm.second);
  return repeated&&guards?0:1;
}
