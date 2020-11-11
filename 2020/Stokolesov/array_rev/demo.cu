#include <stdio.h>
#include <time.h>

// by lectures and "CUDA by Example" book

// device code: array reversing
__global__ void rev_array_kernel(int* a, int array_len) {
    printf("blockId, threadId: %d, %d\n", blockIdx.x, threadIdx.x);
    // swap area right bound
    int right = array_len / 2;
    // element index that corresponds to current thread
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    // process all indices that correspond to current thread
    while (ind < right) {
        printf("blockId, threadId, ind: %d, %d, %d\n", blockIdx.x, threadIdx.x, ind);
        // TODO: move to separate function
        int swap_ind = array_len - 1 - ind;
        int temp = a[ind];
        a[ind] = a[swap_ind];
        a[swap_ind] = temp;
        ind += blockDim.x * gridDim.x;
    }
}

// host code: preparation
void rev_array_gpu(int* host_a, int array_len) {
    // Step 1
    int* dev_a;
    // size of memory to allocate on device for array
    long mem_size = array_len * sizeof(int);
    // device memory allocation for array
    cudaMalloc((void**) &dev_a, mem_size);
    // copying data from host to device array
    cudaMemcpy(dev_a, host_a, mem_size, cudaMemcpyHostToDevice);
    
    // Step 2
    // grid (of blocks) dimensions initialization
    dim3 DimGrid(2, 1, 1);
    // block (of threads) dimensions initialization
    dim3 DimBlock(4, 1, 1);
    // running kernel array reversing code
    rev_array_kernel<<<DimGrid, DimBlock>>>(dev_a, array_len);

    // Step 3
    // copying result from device to host array
    cudaMemcpy(host_a, dev_a, mem_size, cudaMemcpyDeviceToHost);
    // freeing device memory
    cudaFree(dev_a);
}

int main() {
    // array lengths
    int num_lens = 2;
    int array_lens[num_lens] = { 20, 21 };
    for (int i = 0; i < num_lens; i++) {
        int array_len = array_lens[i];
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
        rev_array_gpu(host_a, array_len);

        // showing result
        printf("reversed:\n");
        for (int i = 0; i < array_len; i++)
            printf("a[%d] = %d\n", i, host_a[i]);

        printf("----------------------\n");
    }

    return 0;
}