#include <stdio.h>

// by lectures and "CUDA by Example" book

#define ind(k, i, j, rows, cols) (k * (rows * cols) + i * cols + j)


// device code: matrices sum calculation
__global__ void sum_matrices_kernel(int* mat_stack, int* mat, int rows, int cols, int num) {
    printf("blockId, threadId, dims: [%d, %d], [%d, %d], [%d, %d]\n",
           blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, rows, cols);
    // row and col that correspond to current thread
    // process all elements that correspond to current thread
    for (int i = blockIdx.y * blockDim.y + threadIdx.y;
            i < rows; i += blockDim.y * gridDim.y)
        for (int j = blockIdx.x * blockDim.x + threadIdx.x;
                j < cols; j += blockDim.x * gridDim.x) {
            printf("blockId, threadId, pos: [%d, %d], [%d, %d], [%d, %d]\n",
                blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, i, j);
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
void sum_matrices_gpu(int* host_mat_stack, int* host_m, int rows, int cols, int num) {
    // Step 1: moving data on device
    int* dev_mat_stack = cuda_copy_tens(host_mat_stack, rows, cols, num);
    int* dev_m = cuda_copy_tens(host_m, rows, cols, 1);
    
    // Step 2
    // grid (of blocks) dimensions
    dim3 grid_dim(3, 2, 1);
    // block (of threads) dimensions
    dim3 block_dim(2, 2, 1);
    // running kernel summation code
    sum_matrices_kernel<<<grid_dim, block_dim>>>(dev_mat_stack, dev_m, rows, cols, num);

    // Step 3
    // copying result from device to host matrix
    cudaMemcpy(host_m, dev_m, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
    // freeing device memory
    cudaFree(dev_mat_stack);
    cudaFree(dev_m);
}

void sum_matrices_cpu(int* mat_stack, int* mat, int rows, int cols, int num) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            int mat_ind = ind(0, i, j, rows, cols);
            mat[mat_ind] = 0;
            for (int k = 0; k < num; k++) {
                int stack_ind = ind(k, i, j, rows, cols);
                mat[mat_ind] += mat_stack[stack_ind];
            }
        }
}

// initialize matrix stack
int* init_mat_stack(int* mat_stack, int rows, int cols, int num) {
    for (int k = 0; k < num; k++)
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                int index = ind(k, i, j, rows, cols);
                int rel_index = ind(0, i, j, rows, cols);
                mat_stack[index] = (k + 1) * (rel_index + 1);
            }
    return mat_stack;
}

// print matrix
void print_mat(const char* header, int* mat, int rows, int cols) {
    printf("%s (%d, %d):\n", header, rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            int index = ind(0, i, j, rows, cols);
            printf("\t%d ", mat[index]);
            if (j == cols - 1)
                printf("\n");
    }
}

void print_mat_stack(int* mat_stack, int rows, int cols, int num) {
    printf("Matrix stack (%d, %d) x %d:\n", rows, cols, num);
    for (int k = 0; k < num; k++) {
        char *header = (char*) malloc(256 * sizeof(char));
        sprintf(header, "Matrix #%d", k + 1);
        int* matrix_offset = mat_stack + k * (rows * cols) * sizeof(char);
        print_mat(header, matrix_offset, rows, cols);
    }
}

int main() {
    // matrix params
    int rows = 6;
    int cols = 8;
    int num = 3;

    // first matrix
    int* host_mat_stack = (int*) malloc(rows * cols * num * sizeof(int));
    init_mat_stack(host_mat_stack, rows, cols, num);
    print_mat_stack(host_mat_stack, rows, cols, num);

    // result matrix
    int* host_m = (int*) malloc(rows * cols * sizeof(int));
    print_mat("Result matrix", host_m, rows, cols);

    // summation on device
    sum_matrices_gpu(host_mat_stack, host_m, rows, cols, num);
    
    // showing result
    print_mat("Result matrix", host_m, rows, cols);
    
    // summation on host
    sum_matrices_cpu(host_mat_stack, host_m, rows, cols, num);

    // showing result
    print_mat("Result matrix", host_m, rows, cols);
    
    return 0;
}