// 4.	Вывести на экран числа от 1 до 65535
#include "stdio.h"

__global__ void kernel(long long n, long long* array){
    long long i = blockDim.x * blockIdx.x + threadIdx.x;
    long long offset = blockDim.x * gridDim.x;

    for (;i < n; i += offset)
    {
        array[i] = i * i * i;
    }
}



int main() {
    float time;
    printf("Print something: f.e. 65535:\n");
    long long n; // = 65365;
    scanf("%I64d", &n);
    
    long long* c_array = (long long*)malloc(n * sizeof(long long));
    long long* cu_array;
    cudaMalloc(&cu_array, sizeof(long long) * n);
    cudaMemcpy(cu_array, c_array, sizeof(long long) * n, cudaMemcpyHostToDevice); // from CPU to GPU

    cudaEvent_t start, end;
    cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

    kernel<<<256,256>>>(n, cu_array);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
 
    printf("kernel = <<<%d, %d>>>, time = %f\n", 256, 256, time);

    cudaMemcpy(c_array, cu_array, sizeof(long long) * n, cudaMemcpyDeviceToHost); // From GPU to CPU
    cudaFree(cu_array);

     for (long i = 0; i < n; i++) {
	 	// printf("%d \n", c_array[i]);
	 	printf("%d ", c_array[i]);
	 }
    free(c_array);
    system ( "PAUSE" );
    return 0;
}