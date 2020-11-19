#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <cuda_runtime.h>

//polymorphism
void allocate_matrix(int **matrix, int N){
    *matrix = (int*) malloc(sizeof(int) * N * N); 
}
void allocate_matrix(float **matrix, int N){
    *matrix = (float*) malloc(sizeof(float) * N * N); 
}
void generate_matrix(int *matrix, int N){
    for(int i = 0; i < N; i++){
       for(int j = 0; j < N; j++){
           *(matrix + i*N +j) = rand() % N + 10;
       } 
    }
}
void output_matrix(int *matrix, int N) {
    for(int i = 0; i < N; i++){
       printf("\n");
       for(int j = 0; j < N; j++){
           printf("%6d ",*(matrix + i*N +j));
       } 
    }
    printf("\n");
}
void output_matrix(float *matrix, int N) {
    for(int i = 0; i < N; i++){
       printf("\n");
       for(int j = 0; j < N; j++){
           printf("%6.2f ",*(matrix + i*N +j));
       } 
    }
    printf("\n");
}


texture<int, 1, cudaReadModeElementType> matrix_t;

__global__ void LU(float *lu_matrix, const int N, bool * flag){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    if(id >=N) return; 
 
    for(int j = id; j < N; j+=blockDim.x*gridDim.x){
        if(tex1Dfetch(matrix_t, j) == 0) *flag = false;
        *(lu_matrix+j) = (float) tex1Dfetch(matrix_t, j);
    }
 
    __syncthreads();
    if(!*flag) return;
 
    for(int r = 1; r < N; r++){
        int temp_id = id;
        while(temp_id < r){
          temp_id+=blockDim.x*gridDim.x;
        }
        if(temp_id >= N) return ;
        __syncthreads();
        

        //L matrix calculation
        for(int i = temp_id; i < N; i+=blockDim.x*gridDim.x){
            *(lu_matrix+i*N + r-1) = (float) tex1Dfetch(matrix_t, i*N+r-1);
            for(int k = 0; k < r-1; k++)
            {
                *(lu_matrix+(i)*N+r-1) -= (*(lu_matrix+i*N+k)) * (*(lu_matrix+k*N + r-1));
            }
            
            *(lu_matrix+(i)*N+r-1) /= *(lu_matrix+(r-1)*N+r-1);
        }

        __syncthreads();

        //U matrix calculation
        for(int j = temp_id; j < N; j+=blockDim.x*gridDim.x){
            *(lu_matrix+r*N+j) = (float) tex1Dfetch(matrix_t, r*N+j);
            
            for(int k = 0; k <r; k++){
                *(lu_matrix+r*N+j) -= (*(lu_matrix+r*N+k)) * (*(lu_matrix+k*N + j));
            }
            if(tex1Dfetch(matrix_t, r*N+j) == 0)  *flag = false;
        }
        __syncthreads();
        if(!*flag) return;
    }
}

int gpu_solution(int *matrix, int N, float *lu_matrix){
    int * dev_matrix, threads;
    float * dev_lu_matrix, elapsed_time;
	  cudaEvent_t start, stop;
    bool flag = true, * dev_flag;

    cudaEventCreate(&start);
	  cudaEventCreate(&stop);
 
	  cudaMalloc(&dev_flag, sizeof(bool));
  	cudaMalloc(&dev_matrix, sizeof(int) * N * N);
	  cudaMalloc(&dev_lu_matrix, sizeof(float) * N * N);
 
    cudaMemcpy(dev_matrix, matrix, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_flag, &flag, sizeof(bool), cudaMemcpyHostToDevice);
    
 
    cudaBindTexture(NULL, matrix_t, dev_matrix, cudaCreateChannelDesc<int>(), sizeof(int) * N * N);
    
    if(N < 250){
        threads = N;
    }
    else{
        threads = 250;
    }

    cudaEventRecord(start, 0);
 
  	LU<<<2,threads>>>(dev_lu_matrix, N, dev_flag);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
	  
    cudaEventElapsedTime(&elapsed_time, start, stop);
	  
    cudaEventDestroy(start);
  	cudaEventDestroy(stop);
    
    cudaMemcpy(&flag, dev_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    if(flag){
      cudaMemcpy(lu_matrix, dev_lu_matrix, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
    }
	  cudaFree(dev_matrix);
	  cudaFree(dev_lu_matrix);
 
    if(!flag) return -1;
 
	  return round(elapsed_time);
}

int cpu_solution(int *matrix, int N, float *lu_matrix){
    clock_t start = clock();
    
    for(int j = 0; j < N; j++){
        if((*(matrix+j)) == 0) return -1;
        *(lu_matrix+j) = (float) *(matrix+j);
    }
    
    for(int r = 1; r < N; r++){
        for(int i = r; i < N; i++)
        {
  
            *(lu_matrix+i*N + r-1) = (float) *(matrix+i*N+r-1);
            for(int k = 0; k < r-1; k++){
                *(lu_matrix+(i)*N+r-1) -= (*(lu_matrix+i*N+k)) * (*(lu_matrix+k*N + r-1));
            }
            
            *(lu_matrix+(i)*N+r-1) /= *(lu_matrix+(r-1)*N+r-1);         
        }
        

        for(int j = r; j < N; j++){
            *(lu_matrix+r*N+j) = (float) *(matrix+r*N+j);
            
            for(int k = 0; k <r; k++){
                *(lu_matrix+r*N+j) -= (*(lu_matrix+r*N+k)) * (*(lu_matrix+k*N + j));
            }
            
            if((*(matrix+(r)*N+j)) == 0) 
                return -1;
        }
    }
    
    return (int)(clock() - start)/ (CLOCKS_PER_SEC / 1000);
}



int main() {
    int N=2048;
 /*   int test_matrix[9] = {
        1, 2, 1,
        2, 1, 1,
        1, -1, 2
    };
*/
    int * matrix;
    float * cpu_lu_matrix, * gpu_lu_matrix;
    int time_cpu, time_gpu;
/*    
    matrix = test_matrix;
    
    printf("\nИсходная матрица:\n");
    output_matrix(matrix, N);
    
    allocate_matrix(&cpu_lu_matrix, N);
    time_cpu = cpu_solution(matrix, N, cpu_lu_matrix);
    allocate_matrix(&gpu_lu_matrix, N);
    time_gpu = gpu_solution(matrix, N, gpu_lu_matrix);

    printf("\nLU GPU:\n");
    output_matrix(gpu_lu_matrix, N);

    free(gpu_lu_matrix);
 
    if(time_cpu >= 0 && time_gpu >= 0){
        printf("\nTime on CPU - %d ms\nTime on GPU - %d ms\n",time_cpu, time_gpu);
    }
*/
//N-matrix
    allocate_matrix(&matrix, N);
    generate_matrix(matrix, N);

    allocate_matrix(&cpu_lu_matrix, N);
    time_cpu = cpu_solution(matrix, N, cpu_lu_matrix);
    
    allocate_matrix(&gpu_lu_matrix, N);
    time_gpu = gpu_solution(matrix, N, gpu_lu_matrix);

 
    if(time_cpu >= 0 && time_gpu >= 0){
        printf("\nTime on CPU - %d ms\nTime on GPU - %d ms\n",time_cpu, time_gpu);
    }

    srand(time(NULL));


    return 0;
}
