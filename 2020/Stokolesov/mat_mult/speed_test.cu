#include <stdio.h>
#include <time.h>

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
    // row and col that correspond to current thread
    // process all elements that correspond to current thread
    for (int i = blockIdx.y * blockDim.y + threadIdx.y;
            i < rows; i += blockDim.y * gridDim.y)
        for (int j = blockIdx.x * blockDim.x + threadIdx.x;
                j < cols; j += blockDim.x * gridDim.x) {
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
float mult_matrices_gpu(int* host_m1, dim2 m1_dims,
                       int* host_m2, dim2 m2_dims,
                       int* host_m3, dim2 m3_dims) {
    // Step 1: moving data on device
    int* dev_m1 = cuda_copy_mat(host_m1, m1_dims);
    int* dev_m2 = cuda_copy_mat(host_m2, m3_dims);
    int* dev_m3 = cuda_copy_mat(host_m3, m3_dims);
    
    // Step 2
    // grid (of blocks) dimensions
    dim3 grid_dim(128, 128, 1);
    // block (of threads) dimensions
    dim3 block_dim(32, 32, 1);
    // running kernel multiplication code
    clock_t start = clock();
    mult_matrices_kernel<<<grid_dim, block_dim>>>(dev_m1, dev_m2, dev_m3, m3_dims, m1_dims.cols);
    cudaDeviceSynchronize();
    clock_t end = clock();
    float time = (float)(end - start) / CLOCKS_PER_SEC;

    // Step 3
    // copying result from device to host matrix
    cudaMemcpy(host_m3, dev_m3, m3_dims.rows * m3_dims.cols * sizeof(int), cudaMemcpyDeviceToHost);
    // freeing device memory
    cudaFree(dev_m1);
    cudaFree(dev_m2);
    cudaFree(dev_m3);

    return time;
}

float mult_matrices_cpu(int* m1, dim2 m1_dims,
                       int* m2, dim2 m2_dims,
                       int* m3, dim2 m3_dims) {
    clock_t start = clock();
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
    clock_t end = clock();
    float time = (float)(end - start) / CLOCKS_PER_SEC;
    return time;
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

int main() {
    for (double dim = 1e2; dim <= 1e4; dim *= 2) {
        // first matrix
        int m1_rows = int(dim);
        int inner_dim = int(dim);
        struct dim2 m1_dims = {m1_rows, inner_dim};
        int* host_m1 = create_mat(m1_dims, 1);
        
        // second matrix
        int m2_cols = int(dim);
        struct dim2 m2_dims = {inner_dim, m2_cols};
        int* host_m2 = create_mat(m2_dims, 2);

        // result matrix
        struct dim2 m3_dims = {m1_dims.rows, m2_dims.cols};
        int* host_m3 = create_mat(m3_dims, 0);

        // multiplication
        float gpu_time = mult_matrices_gpu(host_m1, m1_dims,
                        host_m2, m2_dims,
                        host_m3, m3_dims);
        float cpu_time = mult_matrices_cpu(host_m1, m1_dims,
                        host_m2, m2_dims,
                        host_m3, m3_dims);
        printf("Matrix shapes: (%.1e, %.1e) and (%.1e, %.1e)\n",
            double(m1_dims.rows), double(m1_dims.cols),
            double(m2_dims.rows), double(m2_dims.cols));
        printf("CPU time: %.10f\n", cpu_time);
        printf("GPU time: %.10f\n", gpu_time);
        printf("CPU / GPU time: %.10f\n", cpu_time / gpu_time);
        printf("--------------------------\n");
    }

    return 0;
}