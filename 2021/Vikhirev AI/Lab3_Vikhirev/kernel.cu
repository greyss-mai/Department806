// System includes
#include <assert.h>
#include <stdio.h>
#include<math.h>
#include <time.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include "device_launch_parameters.h"

#include <math.h> 

//Кол-во потоков
#define N 512
//Кол-во блоков
#define BL 32

//Интервал в границах поиска корней
#define A 0
#define B 1000

//Точность
#define EPS 0.000001f

//=======================GPU=====================================

__device__ __host__ double f(double x)  //функция
{
    return  pow((1.2 * x),3) - pow((3.53 * x),2) - (1.36 * x) + 7.11;
}

__device__ __host__ double f1(double x)    // первая производная функции, f'
{
    return  pow((3.6 * x), 2) - (7.06 * x) - 1.36;
}

__device__ __host__ double f2(double x)    //вторая производная функции, f''
{
    return (7.2 * x) - 7.06;
}

__global__ void NewtonsMethod(float* result_d, double step)
{
    //Считаем границы для gpu
    float a = A + blockIdx.x * threadIdx.x * step;
    float b = A + (blockIdx.x * threadIdx.x + 1) * step;

    if (f(a) * f(b) * 1.0 > 0)
        return;

    double c;

    if (f(a) * f2(a) > 0)
        c = a;
    else
        c = b;
    do
    {
        c = c - f(c) / f1(c);

    } while (fabs(f(c)) >= EPS);  // цикл ищет корень пока его значение больше заданой точности


    result_d[blockIdx.x * threadIdx.x] = c;


}

int main(int argc, char* argv)
{
    size_t free, total;
    printf("\n");
    cudaMemGetInfo(&free, &total);
    printf("%d KB free of total %d KB\n", free / 1024, total / 1024);

    //Создаем массив для вывода корней(результата)
    const unsigned int n = N * BL;


    //thrust вектор для вывода корней
    thrust::host_vector<float> h(n, 0);
    thrust::device_vector<float> d(n, 0);


    float* result = thrust::raw_pointer_cast(d);

    //Вычисляем шаг для границ поиска корней
    float step = fabs(A - B) * 1.0 / n;

    ////---ВЫЧИСЛЕНИЕ НА ВИДЕОКАРТЕ---
    printf("[Reverse computing Using CUDA] - Starting...\n");

    cudaStream_t stream;

    //// Выделение памяти на устройстве
    float* result_d;
    checkCudaErrors(cudaMalloc(&result_d, n * sizeof(float)));


    //// Выделение памяти на видеокарте и синхронизация всех потоков
    checkCudaErrors(cudaMemcpy(result_d, result, n * sizeof(float), cudaMemcpyHostToDevice));

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
    NewtonsMethod << <BL, N >> > (result_d, step);

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

    //==========================CPU=================================

    // Начинаем замер времени
    auto begin = std::chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < n; i++)
    {
        //Вычисляем границы для CPU
        float a = A + i * step;
        float b = A + (i + 1) * step;

        if (f(a) * f(b) * 1.0 > 0)
            continue;

        double c;

        if (f(a) * f2(a) > 0)
            c = a;
        else
            c = b;
        do
        {
            c = c - f(c) / f1(c);

        } while (fabs(f(c)) >= EPS);  // цикл ищет корень пока его значение больше заданой точности

        h[i] = c;
    }

    //// Останавливаем таймер и считаем время выполнения
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    printf(
        "Time CPU = %.10f microsec\n",
        elapsed.count() * 1e-3);

    ////Вывод значений


    for (unsigned int i = 0; i < n; i++)
    {
        if (result[i] > 1e-05)
        {
            printf("GPU root (%d) == %f \n", i, result[i]);
        }

        if (h[i] > 1e-05)
        {
            printf("CPU root (%d) == %f \n", i, h[i]);
        }
    }
    //// Освобождение памяти
    checkCudaErrors(cudaFree(result_d));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    cudaDeviceReset();

    return 0;
}