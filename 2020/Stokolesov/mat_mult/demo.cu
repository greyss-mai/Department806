#include <stdio.h>

// by lectures and "CUDA by Example" book

#define ind(i, j, cols) (i * cols + j)

struct dim2 {
    int rows;
    int cols;
};

// device code: matrices mult calculation
__global__ void mult_matrices_kernel(int* m1, int* m2, int* m3, dim2 m3_dims, int inner_dim) {
    int rows = m3_dims.rows;
    int cols = m3_dims.cols;
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
            int index_3 = ind(i, j, cols);
            m3[index_3] = 0;
            // iterating over row and down column
            for (int k = 0; k < inner_dim; k++) {
                int index_1 = ind(i, k, inner_dim);
                int index_2 = ind(k, j, cols);
                m3[index_3] += m1[index_1] * m2[index_2];
            }
        }
}

int* cuda_copy_mat(int* host_m, dim2 m_dims) {
    int* dev_m;
    // size of memory to allocate on device for matrix
    long mem_size = m_dims.rows * m_dims.cols * sizeof(int);
    // device memory allocation
    cudaMalloc((void**) &dev_m, mem_size);
    // copying data from host to device
    cudaMemcpy(dev_m, host_m, mem_size, cudaMemcpyHostToDevice);
    // returning pointer
    return dev_m;
}

// host code: preparation
void mult_matrices_gpu(int* host_m1, dim2 m1_dims,
                       int* host_m2, dim2 m2_dims,
                       int* host_m3, dim2 m3_dims) {
    // Step 1: moving data on device
    int* dev_m1 = cuda_copy_mat(host_m1, m1_dims);
    int* dev_m2 = cuda_copy_mat(host_m2, m3_dims);
    int* dev_m3 = cuda_copy_mat(host_m3, m3_dims);
    
    // Step 2
    // grid (of blocks) dimensions
    dim3 grid_dim(3, 2, 1);
    // block (of threads) dimensions
    dim3 block_dim(2, 2, 1);
    // running kernel multiplication code
    mult_matrices_kernel<<<grid_dim, block_dim>>>(dev_m1, dev_m2, dev_m3, m3_dims, m1_dims.cols);

    // Step 3
    // copying result from device to host matrix
    cudaMemcpy(host_m3, dev_m3, m3_dims.rows * m3_dims.cols * sizeof(int), cudaMemcpyDeviceToHost);
    // freeing device memory
    cudaFree(dev_m1);
    cudaFree(dev_m2);
    cudaFree(dev_m3);
}

void mult_matrices_cpu(int* m1, dim2 m1_dims,
                       int* m2, dim2 m2_dims,
                       int* m3, dim2 m3_dims) {
    for (int i = 0; i < m1_dims.rows; i++)
        for (int j = 0; j < m2_dims.cols; j++) {
            int index_3 = ind(i, j, m3_dims.cols);
            m3[index_3] = 0;
            for (int k = 0; k < m1_dims.cols; k++) {
                    int index_1 = ind(i, k, m1_dims.cols);
                    int index_2 = ind(k, j, m2_dims.cols);
                    m3[index_3] += m1[index_1] * m2[index_2];
            }
        }
}

// create matrix (array representation)
int* create_mat(dim2 dims, int k) {
    int rows = dims.rows;
    int cols = dims.cols;
    int* mat = (int*)malloc(rows * cols * sizeof(int));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            mat[ind(i, j, cols)] = k * (ind(i, j, cols) + 1);
    return mat;
}

// print matrix
void print_mat(const char* header, int* mat, dim2 dims) {
    int rows = dims.rows;
    int cols = dims.cols;
    printf("%s (%d, %d):\n", header, rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            printf("\t%d ", mat[ind(i, j, cols)]);
            if (j == cols - 1)
                printf("\n");
    }
}

int main() {
    // first matrix
    struct dim2 m1_dims = {6, 4};
    int* host_m1 = create_mat(m1_dims, 1);
    print_mat("First matrix", host_m1, m1_dims);

    // second matrix
    struct dim2 m2_dims = {4, 8};
    int* host_m2 = create_mat(m2_dims, 2);
    print_mat("Second matrix", host_m2, m2_dims);

    // dimensionality validation
    if (m1_dims.cols != m2_dims.rows) {
        printf("Error: Inner matrix dimensions does not match:\n"
               "(%d, %d) and (%d, %d)",
               m1_dims.rows, m1_dims.cols, m2_dims.rows, m2_dims.cols);
        return 0;
    }

    // result matrix
    struct dim2 m3_dims = {m1_dims.rows, m2_dims.cols};
    int* host_m3 = create_mat(m3_dims, 0);
    print_mat("Third matrix", host_m3, m3_dims);

    // multiplication
    mult_matrices_gpu(host_m1, m1_dims,
                      host_m2, m2_dims,
                      host_m3, m3_dims);

    // showing result
    print_mat("Result matrix", host_m3, m3_dims);

    // multiplication
    mult_matrices_cpu(host_m1, m1_dims,
                      host_m2, m2_dims,
                      host_m3, m3_dims);

    // showing result
    print_mat("Result matrix", host_m3, m3_dims);

    return 0;
}