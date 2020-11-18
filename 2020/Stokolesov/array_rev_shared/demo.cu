#include <stdio.h>
#include <time.h>

// by lectures, "CUDA by Example"
// and "CUDA, Supercomputing for the Masses" books

const int numThreadsPerBlock = 4;
const int numBlocks = 3;
int array_len = numBlocks * numThreadsPerBlock * 3;

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
void rev_array_gpu(int* host_a) {
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
    rev_array_kernel_shared<<<DimGrid, DimBlock, sharedMemSize>>>(dev_a, dev_res, array_len);

    // Step 3
    // copying result from device to host array
    cudaMemcpy(host_a, dev_res, mem_size, cudaMemcpyDeviceToHost);
    // freeing device memory
    cudaFree(dev_a);
    cudaFree(dev_res);
}

int main() {
    // array length
    printf("Array length: %d\n", array_len);

    // initial array
    int* host_a = (int*)malloc(array_len * sizeof(int));
    for (int i = 0; i < array_len; i++)
        host_a[i] = i + 1;

    // showing
    printf("initial:\n");
    for (int i = 0; i < array_len; i++)
        printf("a[%d] = %d\n", i, host_a[i]);

    // reversing    
    rev_array_gpu(host_a);

    // showing result
    printf("reversed:\n");
    for (int i = 0; i < array_len; i++)
        printf("a[%d] = %d\n", i, host_a[i]);

    printf("----------------------\n");

    return 0;
}