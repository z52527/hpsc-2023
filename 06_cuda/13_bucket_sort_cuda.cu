#include <cstdio>
#include <cstdlib>
#include <vector>
#define BLOCK_SIZE 64
__global__ void init(int *key, int *bucket, int N){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid > N) return;
  if(tid < 5)bucket[tid] = 0;
  __syncthreads();
  atomicAdd(&bucket[key[tid]], 1);
  __syncthreads();
  int offset=0;
  for (int i = 0; i < tid; i++) {
    offset += bucket[i];
  }
  for (int i = 0; i < bucket[tid]; i++) {
    key[offset+i] = tid;
  }
}
__global__ void sort(int *key, int *bucket, int N){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid > N) return;
  int tmp = 0;
  for (int i = 0; i < tid; i++) {
    tmp += bucket[i];
  }
  for (int i = 0; i < bucket[tid]; i++) {
    key[tmp+i] = tid;
  }
}
int main() {
  int n = 50;
  int range = 5;
//  std::vector<int> key(n);
  int *key, *bucket;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

/*
  std::vector<int> bucket(range);
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
*/
  
  init<<<(n+256-1)/256, 256>>>(key, bucket, n);
  sort<<<(range+256-1)/256,256>>>(key, bucket, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
