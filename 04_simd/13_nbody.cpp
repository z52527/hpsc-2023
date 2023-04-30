#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], cmp[N], ans[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    cmp[i] = i;
  }
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 cmpvec = _mm256_load_ps(cmp);
  __m256 zeros = _mm256_setzero_ps();
  for(int i=0; i<N; i++) {
  /*
    for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
    */
    
    __m256 xi = _mm256_set1_ps(x[i]);
    __m256 yi = _mm256_set1_ps(y[i]);
    __m256 ivec = _mm256_set1_ps(i);
    //if(i!=j)
    __m256 mask = _mm256_cmp_ps(ivec, cmpvec, _CMP_NEQ_OQ);

    //float rx = x[i] - x[j];
    //float ry = y[i] - y[j];
    __m256 rxvec = _mm256_sub_ps(xi, xvec);
    __m256 ryvec = _mm256_sub_ps(yi, yvec);
    //float r = std::sqrt(rx * rx + ry * ry);
    __m256 r2vec = _mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec, ryvec));
    __m256 rrvec = _mm256_blendv_ps(zeros, _mm256_rsqrt_ps(r2vec), mask);

    //fx[i] -= rx * m[j] / (r * r * r);
    //fy[i] -= ry * m[j] / (r * r * r);
    __m256 fxivec = _mm256_mul_ps(_mm256_mul_ps(rxvec, mvec), _mm256_mul_ps(_mm256_mul_ps(rrvec, rrvec), rrvec));
    __m256 fyivec = _mm256_mul_ps(_mm256_mul_ps(ryvec, mvec), _mm256_mul_ps(_mm256_mul_ps(rrvec, rrvec), rrvec));

    __m256 tmp = _mm256_permute2f128_ps(fxivec, fxivec, 1);
    tmp = _mm256_add_ps(tmp, fxivec);
    tmp = _mm256_hadd_ps(tmp, tmp);
    tmp = _mm256_hadd_ps(tmp, tmp);
    _mm256_store_ps(ans, tmp);
    fx[i] = ans[0];
    tmp = _mm256_permute2f128_ps(fyivec, fyivec, 1);
    tmp = _mm256_add_ps(tmp, fyivec);
    tmp = _mm256_hadd_ps(tmp, tmp);
    tmp = _mm256_hadd_ps(tmp, tmp);
    _mm256_store_ps(ans, tmp);
    fy[i] = ans[0];
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
