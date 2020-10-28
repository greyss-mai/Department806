#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <cuda_runtime.h>

int cpu_matrix_multiplication(int *matrix_a, int *matrix_b, int *matrix_c, int a_n, int b_n, int a_m, int b_m)
{
    clock_t start = clock();
    for(int i = 0; i < a_n; i++)
    {
        for(int j = 0; j < b_m; j++)
        {
            *(matrix_c + i * b_m + j) = 0;
            for(int k = 0; k < a_m; k++){
                * (matrix_c + i * b_m + j) += (* (matrix_a + i * a_m + k)) * (* (matrix_b + k * b_m + j));
            } 
        }
    }
    return (int)(clock() - start)/ (CLOCKS_PER_SEC / 1000);
}

__global__ void matrixMul(int *matrix_a, int *matrix_b, int *matrix_c, int a_n, int b_n, int a_m, int b_m)
{
	  int id = blockIdx.x*blockDim.x+threadIdx.x;
    int temp = id;
    while(temp < a_n * b_m)
    {
        int ROW = temp / b_m;
        int COL = temp % b_m;
      
        *(matrix_c + ROW * b_m + COL) = 0;

		    for(int k = 0; k < a_m; k++){
			    * (matrix_c + ROW * b_m + COL) += (* (matrix_a + ROW * a_m + k)) * (* (matrix_b + k * b_m + COL));
		    } 
     
        temp +=blockDim.x*gridDim.x;
    }
}


int gpu_matrix_multiplication(int *matrix_a, int *matrix_b, int *matrix_c, int a_n, int b_n, int a_m, int b_m)
{
	int * dev_matrix_a, * dev_matrix_b, * dev_matrix_c;
	
	cudaEvent_t start, stop;
	
	float elapsed_time;
 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
 
	cudaMalloc(&dev_matrix_a, sizeof(int) * a_n * a_m);
	cudaMalloc(&dev_matrix_b, sizeof(int) * b_n * b_m);
	cudaMalloc(&dev_matrix_c, sizeof(int) * a_n * b_m);
	
	cudaMemcpy(dev_matrix_a, matrix_a, sizeof(int) * a_n * a_m, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matrix_b, matrix_b, sizeof(int) * b_n * b_m, cudaMemcpyHostToDevice);
	
	cudaEventRecord(start, 0);
	matrixMul<<<32,128>>>(dev_matrix_a, dev_matrix_b, dev_matrix_c, a_n, b_n, a_m, b_m);
 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
 
	cudaMemcpy(matrix_c, dev_matrix_c, sizeof(int) * a_n * b_m, cudaMemcpyDeviceToHost);
	
	cudaFree(dev_matrix_a);
	cudaFree(dev_matrix_b);
	cudaFree(dev_matrix_c);

	return floor(elapsed_time);
    
}

void allocate_matrix(int **matrix, int n, int m)
{
    *matrix = (int*) malloc(sizeof(int) * n * m); 
}

void generate_matrix(int *matrix, int n, int m)
{
    for(int i = 0; i < n; i++){
       for(int j = 0; j < m; j++){
           *(matrix + i*m +j) = rand() % m;
       } 
    }
}

void output_matrix(int *matrix, int n, int m) 
{
    for(int i = 0; i < n; i++){
       printf("\n");
       for(int j = 0; j < m; j++){
           printf("%d ",*(matrix + i*m +j));
       } 
    }
}

void compute(int a_row, int b_row, int a_col, int b_col, bool show_matrix_flag = false)
{
    int * matrix_a, * matrix_b, * matrix_c_cpu, * matrix_c_gpu, time_cpu, time_gpu;

    allocate_matrix(&matrix_a, a_row, a_col);
    allocate_matrix(&matrix_b, b_row, b_col);
    
    generate_matrix(matrix_a, a_row, a_col);
    generate_matrix(matrix_b, b_row, b_col);
    
    if(show_matrix_flag){
        printf("\n\nМатрица А:");
        output_matrix(matrix_a, a_row, a_col);
        printf("\n\nМатрица B:");
        output_matrix(matrix_b, b_row, b_col);
    }
    
    
    allocate_matrix(&matrix_c_cpu, a_row, b_col);
    //time_cpu = 0;
    time_cpu = cpu_matrix_multiplication(matrix_a, matrix_b, matrix_c_cpu, a_row, b_row, a_col, b_col);
 
    if(show_matrix_flag){
        printf("\n\nМатрица C(CPU):");
        output_matrix(matrix_c_cpu, a_row, b_col);
    }
    
    free(matrix_c_cpu);
    allocate_matrix(&matrix_c_gpu, a_row, b_col);
    
    time_gpu = gpu_matrix_multiplication(matrix_a, matrix_b, matrix_c_gpu, a_row, b_row, a_col, b_col);
    
    if(show_matrix_flag){
        printf("\n\nМатрица C(GPU):");
        output_matrix(matrix_c_gpu, a_row, b_col);
    }
    
    free(matrix_c_gpu);
    
    free(matrix_a);
    free(matrix_b);
    
    if(!show_matrix_flag){
		    printf("\n\nВремя выполнения (ms) A[%d,%d] * B[%d, %d]:", a_row, a_col, b_row, b_col);
        printf("CPU - %d, GPU - %d\n",  time_cpu, time_gpu);
    }
}

int main() {
    
    srand(time(NULL));
    compute(5,6,6,2,true);
    compute(32,32,32,32);
    compute(64,64,64,64);
    compute(128,128,128,128);
    compute(256,256,256,256);
    compute(512,512,512,512);
    compute(1024,1024,1024,1024);
    //compute(2048,2048,2048,2048);
    //compute(10000,10000,10000,10000);
    
    return 0;
}