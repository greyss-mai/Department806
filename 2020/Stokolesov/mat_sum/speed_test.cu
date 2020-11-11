#include <stdio.h>
#include <time.h>

// by lectures and "CUDA by Example" book

#define ind(k, i, j, rows, cols) (k * (rows * cols) + i * cols + j)


// device code: matrices sum calculation
__global__ void sum_matrices_kernel(int* mat_stack, int* mat, int rows, int cols, int num) {
    // row and col that correspond to current thread
    // process all elements that correspond to current thread
    for (int i = blockIdx.y * blockDim.y + threadIdx.y;
            i < rows; i += blockDim.y * gridDim.y)
        for (int j = blockIdx.x * blockDim.x + threadIdx.x;
                j < cols; j += blockDim.x * gridDim.x) {
            int mat_ind = ind(0, i, j, rows, cols);
            mat[mat_ind] = 0;
            // iterating over all elements on (i, j) position
            for (int k = 0; k < num; k++) {
                int stack_ind = ind(k, i, j, rows, cols);
                mat[mat_ind] += mat_stack[stack_ind];
            }
        }
}

int* cuda_copy_tens(int* host_tensor, int rows, int cols, int num) {
    int* dev_tensor;
    // size of memory to allocate on device for tensor
    long mem_size = rows * cols * num * sizeof(int);
    // device memory allocation
    cudaMalloc((void**) &dev_tensor, mem_size);
    // copying data from host to device
    cudaMemcpy(dev_tensor, host_tensor, mem_size, cudaMemcpyHostToDevice);
    // returning pointer
    return dev_tensor;
}

// host code: preparation
float sum_matrices_gpu(int* host_mat_stack, int* host_m, int rows, int cols, int num) {
    // Step 1: moving data on device
    int* dev_mat_stack = cuda_copy_tens(host_mat_stack, rows, cols, num);
    int* dev_m = cuda_copy_tens(host_m, rows, cols, 1);
    
    // Step 2
    // grid (of blocks) dimensions
    dim3 grid_dim(128, 128, 1);
    // block (of threads) dimensions
    dim3 block_dim(32, 32, 1);
    // running kernel summation code
    clock_t start = clock();
    sum_matrices_kernel<<<grid_dim, block_dim>>>(dev_mat_stack, dev_m, rows, cols, num);
    cudaDeviceSynchronize();
    clock_t end = clock();
    float time = (float)(end - start) / CLOCKS_PER_SEC;

    // Step 3
    // copying result from device to host matrix
    cudaMemcpy(host_m, dev_m, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
    // freeing device memory
    cudaFree(dev_mat_stack);
    cudaFree(dev_m);

    return time;
}

float sum_matrices_cpu(int* mat_stack, int* mat, int rows, int cols, int num) {
    clock_t start = clock();
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            int mat_ind = ind(0, i, j, rows, cols);
            mat[mat_ind] = 0;
            for (int k = 0; k < num; k++) {
                int stack_ind = ind(k, i, j, rows, cols);
                mat[mat_ind] += mat_stack[stack_ind];
            }
        }
    clock_t end = clock();
    float time = (float)(end - start) / CLOCKS_PER_SEC;
    return time;
}

// initialize matrix stack
int* init_mat_stack(int* mat_stack, int rows, int cols, int num) {
    for (int k = 0; k < num; k++)
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                int index = ind(k, i, j, rows, cols);
                mat_stack[index] = rand() % 20;
            }
    return mat_stack;
}

// QUESTION #1: why gpu times differ despite elements of each line
// always processing by single thread and number of threads always more
// than number of lines?
// there should be time always equal to process time of single thread?
// possible answer: in case of multiple threads they must wait each other
// in cudaDeviceSynchronize point?
// QUESTION #2: why when we set dim of block equal to 64x64
// then cudaDeviceSynchronize doesn't work?
int main() {
    // matrix count
    int num = 1e6;

    for (int dim = 10; dim > 0; dim--) {
        int rows = dim;
        int cols = dim;

        // first matrix
        int* host_mat_stack = (int*) malloc(rows * cols * num * sizeof(int));
        init_mat_stack(host_mat_stack, rows, cols, num);

        // result matrix
        int* host_m = (int*) malloc(rows * cols * sizeof(int));

        // summation
        int num_measures = 10;
        float gpu_time_mean = 0;
        float cpu_time_mean = 0;
        for (int i = 0; i < num_measures; i++) {
            gpu_time_mean += sum_matrices_gpu(host_mat_stack, host_m, rows, cols, num);
            cpu_time_mean += sum_matrices_cpu(host_mat_stack, host_m, rows, cols, num);
        }
        gpu_time_mean /= num_measures;
        cpu_time_mean /= num_measures;
        printf("Matrix stack params: (%d, %d) x %.0e\n", rows, cols, double(num));
        printf("CPU time mean: %.10f\n", cpu_time_mean);
        printf("GPU time mean: %.10f\n", gpu_time_mean);
        printf("CPU / GPU time: %.10f\n", cpu_time_mean / gpu_time_mean);
        printf("--------------------------\n");
    }

    return 0;
}