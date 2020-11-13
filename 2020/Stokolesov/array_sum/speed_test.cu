#include <stdio.h>
#include <time.h>

// by lectures and "CUDA by Example" book

// device code: array sum calculation: c = a + b
__global__ void sum_arrays_kernel(float* a, float* b, float* c, int array_len) {
    // element index that corresponds to current thread
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    // process all indices that correspond to current thread
    while (ind < array_len) {
        c[ind] = a[ind] + b[ind];
        ind += blockDim.x * gridDim.x;
    }
}

// host code: preparation
float sum_arrays_gpu(float* host_a, float* host_b, float* host_c, int array_len) {
    // Step 1
    // size of memory to allocate on device for each array
    long mem_size = array_len * sizeof(float);
    // device memory allocation for arrays a, b and c
    // copying data from host to device arrays
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
    dim3 DimGrid(512, 1, 1);
    // block (of threads) dimensions initialization
    dim3 DimBlock(256, 1, 1);
    // running kernel summation code
    clock_t start = clock();
    sum_arrays_kernel<<<DimGrid, DimBlock>>>(dev_a, dev_b, dev_c, array_len);
    cudaDeviceSynchronize();
    clock_t end = clock();
    float time = (float)(end - start) / CLOCKS_PER_SEC;

    // Step 3
    // copying result from device to host array
    cudaMemcpy(host_c, dev_c, mem_size, cudaMemcpyDeviceToHost);
    // freeing device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return time;
}

float sum_arrays_cpu(float* a, float* b, float* c, int array_len) {
    clock_t start = clock();
    for (int ind = 0; ind < array_len; ind++)
        c[ind] = a[ind] + b[ind];
    clock_t end = clock();
    float time = (float)(end - start) / CLOCKS_PER_SEC;
    return time;
}

int main() {
    // array lengths
    int num_lens = 7;
    double array_lens[num_lens] = { 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9 };
    for (int i = 0; i < num_lens; i++) {
        double array_len = array_lens[i];
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
        float cpu_time = sum_arrays_cpu(host_a, host_b, host_c, array_len);
        float gpu_time = sum_arrays_gpu(host_a, host_b, host_c, array_len);
        printf("Array length: %.0e\n", array_len);
        printf("CPU time: %.10f\n", cpu_time);
        printf("GPU time: %.10f\n", gpu_time);
        printf("CPU / GPU time: %.10f\n", cpu_time / gpu_time);
        printf("--------------------------\n");
    }

    return 0;
}