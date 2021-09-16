#include <stdio.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>

__global__ void vectorSqrt(const float* A, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = sqrtf(A[i]);
    }
}

int
main(void)
{
    // цикл по тестируемым длинам массивов
    int pow = 6;
    int numElements = 100;
    while (numElements <= numElements * pow)
    {
        // создание таймера для контроля время выполнения функции
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // выделение памяти под массивы заданной длины
        size_t size = numElements * sizeof(float);

        float* h_A = (float*)malloc(size);
        float* h_C = (float*)malloc(size);

        if (h_A == NULL || h_C == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }

        // инициализация случайными значениями тестируемого массива
        for (int i = 0; i < numElements; ++i)
        {
            h_A[i] = rand() / (float)RAND_MAX;
        }

        // старт таймера на копирование данных на устройство, выполнение функции и копирование результата обратно
        cudaEventRecord(start);

        // выделение массивов заданной длины на устройстве
        float* d_A = NULL;
        cudaMalloc((void**)&d_A, size);
        float* d_C = NULL;
        cudaMalloc((void**)&d_C, size);

        // копирование массива с хоста на устройство
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

        // выделение тредов и блоков
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        // запуск выполнения функции
        vectorSqrt << <blocksPerGrid, threadsPerBlock >> > (d_A, d_C, numElements);

        // копирование результатов обратно на хост
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);

        // остановка таймера
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("For % d elements time elapsed in ms : % f\n", numElements, milliseconds);

        // освобождение памяти на хосте и на устройстве
        cudaFree(d_A);
        cudaFree(d_C);

        free(h_A);
        free(h_C);

        numElements *= 10;
    }

    return 0;
}

