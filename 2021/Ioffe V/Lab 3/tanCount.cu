// System includes
#include <assert.h>
#include <stdio.h>
#include<math.h>
#include <time.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include "device_launch_parameters.h"

#include <math.h> 

//количество потоков
#define N 1024
//количество блоков
#define BL 200

//интервал
#define A -1000
#define B 1000

//точность
#define EPS 1e-5

//код выполняется на GPU

__global__ void staticReverse(float* result_d, double step)
{
    //считаем концы отрезка
    float a = A + blockIdx.x*threadIdx.x*step;
    float b = A + (blockIdx.x*threadIdx.x+1)*step;
    
    //ФУНКЦИЯ f(x) = log(8*x) - 9*x + 3

    float ak = a;
    float bk = b;

    do
    {
        if ((log(8*ak) - 9*ak + 3)*(log(8*bk) - 9*bk + 3)* 1.0 > 0)
            return;

        float xk = (ak + bk)* 1.0 / 2;
        float fxk = log(8*xk) - 9*xk + 3;

        //если мы прям в 0 попали
        if (fxk < 1e-05)
            break;

        if ((log(8*ak) - 9*ak + 3)* fxk * 1.0 < 0)
        {
            bk = xk;
        }
        else
        {
            ak = xk;
        }

    }
    while(bk-ak > EPS);

    float x = (ak + bk) * 1.0 / 2;

    if ((log(8*x) - 9*x + 3) * 1.0 < 1e-5)
    {
      //  printf("d[%d, %d] = %f \n", blockIdx.x, threadIdx.x, x);

        result_d[blockIdx.x*threadIdx.x] = x;
    }

}

int main(int argc, char* argv)
{
    size_t free, total;
    printf("\n");
    cudaMemGetInfo(&free, &total);
    printf("%d KB free of total %d KB\n", free / 1024, total / 1024);

    //создаем массив для результата
    const unsigned int n = N*BL;
    float* result = new float[n];

    for (unsigned int i = 0; i < n; i++) {
        result[i] = 0;
    }

    //вычисляем шаг
    float step = fabs(A-B) * 1.0 / n;

    ////---ВЫЧИСЛЕНИЕ НА ВИДЕОКАРТЕ---
    printf("[Reverse computing Using CUDA] - Starting...\n");

    cudaStream_t stream;

    //// Выделение памяти на устройстве

    float* result_d;
    checkCudaErrors(cudaMalloc(&result_d, n * sizeof(float)));


    //// Выделение памяти на видеокарте и синхронизация всех потоков
    checkCudaErrors(cudaMemcpy(result_d, result, n*sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSynchronize());

    //// Создание событий и стрима для таймера
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));


    printf("Computing result using CUDA Kernel...\n");

    //// Запись начала события
    checkCudaErrors(cudaEventRecord(start, stream));

    //// Выполнение кода на видеокарте и ожидание завершения всех потоков

    staticReverse << <BL, N >> > (result_d, step);


    checkCudaErrors(cudaStreamSynchronize(stream));

    //// Запись окончания события
    checkCudaErrors(cudaEventRecord(stop, stream));

    //// Синхронизация и ожидание завершающего события
    checkCudaErrors(cudaEventSynchronize(stop));

    //// Расчет и вывод производительности

    float m_sec_total = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&m_sec_total, start, stop));

    float mc_sec_total = m_sec_total * 1000;
    printf(
        "Time GPU = %.10f microsec\n",
        mc_sec_total);

    //// Копирование результатов с GPU на CPU
    checkCudaErrors(
        cudaMemcpy(result, result_d, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaStreamSynchronize(stream));


    ////---ВЫЧИСЛЕНИЕ НА ПРОЦЕССОРЕ---

    
    float cpu_result[n];

    for (unsigned int i = 0; i < n; i++)
    {
        cpu_result[i] = 0;
    }

    // Начинаем замер времени
    auto begin = std::chrono::high_resolution_clock::now();

    //ФУНКЦИЯ f(x) = log(8*x) - 9*x + 3

    for (unsigned int i = 0; i < n; i++)
    {
        //считаем концы отрезка
        float a = A + i*step;
        float b = A + (i+1)*step;

        float ak = a;
        float bk = b;

        do
        {
            if ((log(8*ak) - 9*ak + 3)*(log(8*bk) - 9*bk + 3)* 1.0 > 0)
                break;

            float xk = (ak + bk)* 1.0 / 2;
            float fxk = log(8*xk) - 9*xk + 3;

            //если мы прям в 0 попали
            if (fxk < 1e-05)
                break;

            if ((log(8*ak) - 9*ak + 3)*fxk* 1.0 < 0)
            {
                bk = xk;
            }
            else
            {
                ak = xk;
            }

        }
        while(bk-ak > EPS);

        if ((log(8*ak) - 9*ak + 3)*(log(8*bk) - 9*bk + 3)* 1.0 <= 0)
           { 
                float x = (ak + bk) * 1.0 / 2;

                if ((log(8*x) - 9*x + 3) * 1.0 < 1e-5)
                {
                    //printf("d[%d] = %f \n", i, x);

                     cpu_result[i] = x;
                }
                    
           }
    }
    
    
    //// Останавливаем таймер и считаем время выполнения
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    printf(
        "Time CPU = %.10f microsec\n",
        elapsed.count() * 1e-3);

    ////проверка значений

    for (unsigned int i = 0; i < n; i++)
    {
        if (fabs(result[i] - cpu_result[i]) > 1e-05)
        {
           printf("d[%d] != r[%d] (%f, %f) \n", i, i, result[i], cpu_result[i]);
        }

        if (result[i] > 1e-05)
        {
           printf("GPU root (%d) == %f \n", i, result[i]);
        }

        if (cpu_result[i] > 1e-05)
        {
           printf("CPU root (%d) == %f \n", i, cpu_result[i]);
        }
    }

    //// Освобождение памяти
    checkCudaErrors(cudaFree(result_d));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    cudaDeviceReset();

    return 0;
}
