#include <stdio.h>
#include <time.h>

// by lectures and "CUDA by Example" book

// device code: array reversing
__global__ void rev_array_kernel(float* a, int array_len) {
    // printf("blockId, threadId: %d, %d\n", blockIdx.x, threadIdx.x);
    // swap area right bound
    int right = array_len / 2;
    // element index that corresponds to current thread
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    // process all indices that correspond to current thread
    while (ind < right) {
        // printf("blockId, threadId, ind: %d, %d, %d\n", blockIdx.x, threadIdx.x, ind);
        // TODO: move to separate function
        int swap_ind = array_len - 1 - ind;
        int temp = a[ind];
        a[ind] = a[swap_ind];
        a[swap_ind] = temp;
        ind += blockDim.x * gridDim.x;
    }
}

// host code: preparation
float rev_array_gpu(float* host_a, int array_len) {
    // Step 1
    float* dev_a;
    // size of memory to allocate on device for array
    long mem_size = array_len * sizeof(float);
    // device memory allocation for array
    cudaMalloc((void**) &dev_a, mem_size);
    // copying data from host to device array
    cudaMemcpy(dev_a, host_a, mem_size, cudaMemcpyHostToDevice);
    
    // Step 2
    // grid (of blocks) dimensions initialization
    // dim3 DimGrid(2, 1, 1);
    dim3 DimGrid(512, 1, 1);
    // block (of threads) dimensions initialization
    // dim3 DimBlock(4, 1, 1);
    dim3 DimBlock(256, 1, 1);
    // running kernel array reversing code
    clock_t start = clock();
    rev_array_kernel<<<DimGrid, DimBlock>>>(dev_a, array_len);
    cudaDeviceSynchronize();
    clock_t end = clock();
    float time = (float)(end - start) / CLOCKS_PER_SEC;

    // Step 3
    // copying result from device to host array
    // NOTE: time complexity?
    cudaMemcpy(host_a, dev_a, mem_size, cudaMemcpyDeviceToHost);
    // freeing device memory
    cudaFree(dev_a);

    return time;
}

float rev_array_cpu(float* a, int array_len) {
    clock_t start = clock();
    for (int ind = 0; ind < array_len / 2; ind++) {
        // TODO: move to separate function
        int swap_ind = array_len - 1 - ind;
        int temp = a[ind];
        a[ind] = a[swap_ind];
        a[swap_ind] = temp;
    }
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
        // initial array
        float* host_a = (float*)malloc(array_len * sizeof(float));
        for (int i = 0; i < array_len; i++)
            host_a[i] = i + 1;

        // reversing
        // TODO: make multiple tests
        // and separate fun for calc stats like mean and std
        float cpu_time = rev_array_cpu(host_a, array_len);
        float gpu_time = rev_array_gpu(host_a, array_len);
        printf("Array length: %.0e\n", array_len);
        printf("CPU time: %.10f\n", cpu_time);
        printf("GPU time: %.10f\n", gpu_time);
        printf("CPU / GPU time: %.10f\n", cpu_time / gpu_time);
        printf("--------------------------\n");
    }

    return 0;
}