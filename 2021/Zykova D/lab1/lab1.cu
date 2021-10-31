#include <stdio.h>
#include <stdlib.h>

#define CSC(call)                      \
do {                                \
    cudaError_t res = call;            \
    if (res != cudaSuccess) {        \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));        \
        exit(0);                    \
    }                                \
} while(0)

// ядро  kernel инвертирует массив
__global__ void kernel(int *arr1, int n) { 
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // вычисляется индекс элемента который будет обрабатываться
    
    if (idx < n/2) { // обработку необходимо производить только на первой половине массива, иначе массив вернется к исходному состоянию
		printf("block №%d, thread №%d: обмен местами эл-та №%d и №%d\n", blockIdx.x, threadIdx.x, idx, n - idx - 1);

        int tmp = arr1[idx];
        arr1[idx] = arr1[n - idx - 1];
        arr1[n - idx - 1] = tmp;
    }

}

int main(){
	int *arrHost = (int *)malloc(512*sizeof(int));
	int *arrDev;
	int i;
	
	for (i = 0; i < 512; i++)
		arrHost[i] = i + 1;
	
    cudaEvent_t before, after;
    CSC(cudaEventCreate(&before)); // инициализируем 2 события cuda
    CSC(cudaEventCreate(&after));

	CSC(cudaMalloc(&arrDev, 512*sizeof(int)));
	CSC(cudaMemcpy(arrDev, arrHost, 512*sizeof(int), cudaMemcpyHostToDevice));
	
    CSC(cudaEventRecord(before)); // сохраняем текущее время начала работы ядра

	kernel<<<16, 32>>>(arrDev, 512);
	
    CSC(cudaGetLastError());
    CSC(cudaEventRecord(after)); // сохраняем время конца работы ядра

    CSC(cudaEventSynchronize(after));
    float t;
    CSC(cudaEventElapsedTime(&t, before, after)); // считаем время работы ядра
    CSC(cudaEventDestroy(before));
    CSC(cudaEventDestroy(after));

    printf("time = %f\n", t); //выводим посчитанную разницу во времени
    
    CSC(cudaMemcpy(arrHost, arrDev, 512*sizeof(int), cudaMemcpyDeviceToHost));
    
    for (i = 0; i < 512; i++){
		printf("%d ", arrHost[i]);
	}	
	printf("\n");
	CSC(cudaFree(arrDev));
	free(arrHost);
	return 0;
}
