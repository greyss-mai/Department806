#include <stdio.h>
#include <time.h>

// by lectures, "CUDA by Example"
// and "CUDA, Supercomputing for the Masses" books

const int numThreadsPerBlock = 256;
const int numBlocks = 128;

// Question: why this approach should have better performance
// than straightforward parallel algorithm?
// NOTE: array length should be divisible by numThreadsPerBlock * numBlocks for this code
__global__ void rev_array_kernel_shared(int *d_in, int *d_out, int d_len) {
    extern __shared__ int s_data[];

    // corresponding element index of input array
    int in_ind = blockDim.x * blockIdx.x + threadIdx.x;
    // offset in output array for current grid offset over input array
    int offset = d_len - blockDim.x * gridDim.x;
    // corresponding index for element in output array
    int out_ind = offset + (blockDim.x * (gridDim.x - 1 - blockIdx.x) + threadIdx.x);
    // grid offset step over input / output array
    int offset_step = blockDim.x * gridDim.x;
    while (in_ind < d_len) {
        // load one element per thread from device memory and store it
        // *in reversed order* into temporary shared memory
        s_data[blockDim.x - 1 - threadIdx.x] = d_in[in_ind];
    
        // wait until all threads in the block have written their data to shared memory
        __syncthreads();
    
        // write the data from shared memory in forward order,
        // but to the reversed block offset as before
        d_out[out_ind] = s_data[threadIdx.x];

        // proceed to next offset
        in_ind += offset_step;
        out_ind -= offset_step;
    }
}

// host code: preparation
float rev_array_gpu(int* host_a, int array_len) {
    // Step 1
    int* dev_a;
    // size of memory to allocate on device for array
    long mem_size = array_len * sizeof(int);
    // device memory allocation for array
    cudaMalloc((void**) &dev_a, mem_size);
    // copying data from host to device array
    cudaMemcpy(dev_a, host_a, mem_size, cudaMemcpyHostToDevice);
    // result
    int* dev_res;
    cudaMalloc((void**) &dev_res, mem_size);
    
    // Step 2
    int sharedMemSize = numThreadsPerBlock * sizeof(int);
    // grid (of blocks) dimensions initialization
    dim3 DimGrid(numBlocks, 1, 1);
    // block (of threads) dimensions initialization
    dim3 DimBlock(numThreadsPerBlock, 1, 1);
    // running kernel array reversing code
    clock_t start = clock();
    rev_array_kernel_shared<<<DimGrid, DimBlock, sharedMemSize>>>(dev_a, dev_res, array_len);
    cudaDeviceSynchronize();
    clock_t end = clock();
    float time = (float)(end - start) / CLOCKS_PER_SEC;

    // Step 3
    // copying result from device to host array
    cudaMemcpy(host_a, dev_res, mem_size, cudaMemcpyDeviceToHost);
    // freeing device memory
    cudaFree(dev_a);
    cudaFree(dev_res);

    return time;
}

float rev_array_cpu(int* a, int array_len) {
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
    int array_len_base = numBlocks * numThreadsPerBlock;
    int num_factors = 7;
    double array_len_factors[num_factors] = { 1, 10, 50, 100, 200, 500, 1000 };
    for (int i = 0; i < num_factors; i++) {
        double array_len = array_len_base * array_len_factors[i];
        // initial array
        int* host_a = (int*)malloc(array_len * sizeof(int));
        for (int i = 0; i < array_len; i++)
            host_a[i] = i + 1;

        // reversing
        float cpu_time = rev_array_cpu(host_a, array_len);
        float gpu_time = rev_array_gpu(host_a, array_len);
        printf("Array length: %.3e\n", array_len);
        printf("CPU time: %.10f\n", cpu_time);
        printf("GPU time: %.10f\n", gpu_time);
        printf("CPU / GPU time: %.10f\n", cpu_time / gpu_time);
        printf("--------------------------\n");
    }

    return 0;
}