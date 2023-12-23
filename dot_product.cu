#include <device_atomic_functions.h>
#include <math.h>

__global__ void dotProduct(int *a, int *b, int *c, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    atomicAdd(c, a[tid] * b[tid]);
  }
}

int main() {
  int n = 100000;
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;
  int size = n * sizeof(int);

  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, sizeof(int));

  a = (int *)malloc(size);
  b = (int *)malloc(size);
  c = (int *)malloc(sizeof(int));

  for (int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = i;
  }

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, sizeof(int), cudaMemcpyHostToDevice);

  dotProduct<<<ceil(n / 256.0), 256>>>(d_a, d_b, d_c, n);

  cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost);


  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(a);
  free(b);
  free(c);

  return 0;
}
