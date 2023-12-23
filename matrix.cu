__global__ void matrixMulKernel(int *d_M, int *d_N, int *d_P, int Width) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((Row < Width) && (Col < Width)) {
    int Pvalue = 0;
    for (int k = 0; k < Width; ++k) {
      Pvalue += d_M[Row * Width + k] * d_N[k * Width + Col];
    }
    d_P[Row * Width + Col] = Pvalue;
    
  }
}

int main(int argc, char **argv) {
  int Width = 5;
  int *h_M, *h_N, *h_P;
  int *d_M, *d_N, *d_P;
  int size = Width * Width * sizeof(int);
  h_M = (int *)malloc(size);
  h_N = (int *)malloc(size);
  h_P = (int *)malloc(size);
  cudaMalloc((void **)&d_M, size);
  cudaMalloc((void **)&d_N, size);
  cudaMalloc((void **)&d_P, size);
  for (int i = 0; i < Width * Width; i++) {
    h_M[i] = 1;
    h_N[i] = 2;
  }
  cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
  dim3 dimGrid(1, 1);
  dim3 dimBlock(Width, Width);
  matrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
  cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);
  free(h_M);
  free(h_N);
  free(h_P);
  return 0;
}