#include <cuda.h>
#include <iostream>

// For max thread num:
// 1024 -- 32
// 512  -- 16
// 256  -- 8
#define BLOCK_SIZE 32

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}



__global__ void cuda_mat_add(float* dst, float* mat1, float* mat2, int cols) {
    int idx = (blockIdx.y * blockDim.y + threadIdx.y) * cols + blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = mat1[idx] + mat2[idx];    
}


int GPU_mat_add(float* dst, float* mat1, float* mat2, int N, int M) {

    int sizeof_matrix = N * M * sizeof(float);

    int block_size_x = (BLOCK_SIZE > N) ? 1 : N / BLOCK_SIZE;
    int block_size_y = (BLOCK_SIZE > M) ? 1 : M / BLOCK_SIZE;

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(block_size_x, block_size_y);

    // Memory allocate
    cudaEvent_t start, stop;
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    cudaEventRecord(start);

    float* out_dev;
    float* in1_dev;
    float* in2_dev;
    {
        checkCudaErrors(cudaMalloc(&in1_dev, sizeof_matrix));
        checkCudaErrors(cudaMemcpy(in1_dev, mat1, sizeof_matrix, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc(&in2_dev, sizeof_matrix));
        checkCudaErrors(cudaMemcpy(in2_dev, mat2, sizeof_matrix, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc(&out_dev, sizeof_matrix));
    }
    
    {
        cuda_mat_add<<<blocks, threads>>>(out_dev, in1_dev, in2_dev, M);
        cudaThreadSynchronize();

        checkCudaErrors(cudaGetLastError());
    }

    // Memory deallocate
    {
        checkCudaErrors(cudaMemcpy(dst, out_dev, sizeof_matrix, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(out_dev));
        checkCudaErrors(cudaFree(in1_dev));
        checkCudaErrors(cudaFree(in2_dev));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    int microseconds = static_cast<int>(1000.f * milliseconds);

    return microseconds;

}
