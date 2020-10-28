
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define n 10 //длина вектора

//€дро выполн€етс€ паралельно на большом числе нитей
__global__ void kernel(int* a, int* b, int* c)
{
	//глобальный индекс нити
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//выполнить обработку соотвествующих данной нити данных
	c[idx] = a[idx] * b[idx];
}

int main(void)
{
	int numBytes = n * sizeof(int);
	int a[n], b[n], c[n];
	int* adev, * bdev, * cdev;

	//выделить пам€ть на GPU
	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);

	//«адаем массивы
	for (int i = 0; i < n; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}

	//скопировать входные данные из пам€ти CPU в пам€ть GPU
	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

	//вызов €дра с заданной конфигурацией запуска
	kernel <<<n, 1 >>> (adev, bdev, cdev);

	//скопировать результаты в пам€ть CPU
	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

	//¬ывод результата
	for (int idx = 0; idx < n; idx++)
	{
		printf("%d * %d = %d \n", a[idx], b[idx], c[idx]);
	}

	//освободить выделенную пам€ть GPU
	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);

	return 0;
}
