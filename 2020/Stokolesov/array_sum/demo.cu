#include <stdio.h>

// by lectures and "CUDA by Example" book

// device code: array sum calculation: c = a + b
__global__ void sum_arrays_kernel(float* a, float* b, float* c, int array_len) {
    printf("blockId, threadId: %d, %d\n", blockIdx.x, threadIdx.x);
    // element index that corresponds to current thread
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    // process all indices that correspond to current thread
    while (ind < array_len) {
        printf("blockId, threadId, ind: %d, %d, %d\n", blockIdx.x, threadIdx.x, ind);
        c[ind] = a[ind] + b[ind];
        ind += blockDim.x * gridDim.x;
    }
}

// host code: preparation
void sum_arrays_gpu(float* host_a, float* host_b, float* host_c, int array_len) {
    // Step 1
    // size of memory to allocate on device for each array
    long mem_size = array_len * sizeof(float);
    // device memory allocation for arrays a, b and c
    // with copying data from host to device arrays
    float* dev_a;
    cudaMalloc((void**) &dev_a, mem_size);
    cudaMemcpy(dev_a, host_a, mem_size, cudaMemcpyHostToDevice);
    float* dev_b;
    cudaMalloc((void**) &dev_b, mem_size);
    cudaMemcpy(dev_b, host_b, mem_size, cudaMemcpyHostToDevice);
    float* dev_c;
    cudaMalloc((void**) &dev_c, mem_size);
    
    // Step 2
    // grid (of blocks) dimensions initialization
    dim3 DimGrid(2, 1, 1);
    // block (of threads) dimensions initialization
    dim3 DimBlock(4, 1, 1);
    // running kernel summation code
    sum_arrays_kernel<<<DimGrid, DimBlock>>>(dev_a, dev_b, dev_c, array_len);

    // Step 3
    // copying result from device to host array
    cudaMemcpy(host_c, dev_c, mem_size, cudaMemcpyDeviceToHost);
    // freeing device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

int main() {
    // array size
    int array_len = 10;
    // first array
    float* host_a = (float*)malloc(array_len * sizeof(float));
    // second array
    float* host_b = (float*)malloc(array_len * sizeof(float));
    for (int i = 0; i < array_len; i++) {
        host_a[i] = i;
        host_b[i] = (i + 1) * 2;
    }
    // result array
    float* host_c = (float*)malloc(array_len * sizeof(float));

    // summation
    sum_arrays_gpu(host_a, host_b, host_c, array_len);

    // showing result
    printf("host_a[i] + host_b[i] = host_c[i]:\n");
    for (int i = 0; i < array_len; i++)
        printf("%.2f + %.2f = %.2f\n", host_a[i], host_b[i], host_c[i]);

    return 0;
}