#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void diffVec(double *matr1, double *matr2, double *matr3, int x1, int y1, int x2, int y2) {	
	for (int i = 0; i < x1; i++) {
		for (int j = 0; j < y2; j++) {
			matr3[i * y2 + j] = 0;
			for (int k = 0; k < x2; k++) {
				matr3[i * y2 + j] += matr1[i * y1 + k] * matr2[j + k * y2];
			}
		}
	}
}

#define cuda_err(err)                             \
	do { if (err != cudaSuccess) { printf("ERROR: %s\n", cudaGetErrorString(err)); exit(0);} } while (0)

int main() {
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(nullptr);
	int a1, b1, a2, b2;
	printf("Введите размерность матрицы A:\n");
	std::cin >> a1 >> b1;
	printf("Введите значения матрицы A:\n");
	double *m1 = new double[a1 * b1];
	for (int i = 0; i < a1; ++i) {
		for (int j = 0; j < b1; ++j) {
			std::cin >> m1[j + i * b1];
		}
	}
	printf("Введите размерность матрицы B:\n");
	std::cin >> a2 >> b2;
        printf("Введите значения матрицы B:\n");
	double *m2 = new double[a2 * b2];
	for (int i = 0; i < a2; ++i) {
		for (int j = 0; j < b2; ++j) {
			std::cin >> m2[j + i * b2];
		}
	}

	double *m3 = new double[a1 * b2];
	for (int i = 0; i < a1; ++i) {
		for (int j = 0; j < b2; ++j) {
			m3[i + j*b2] = 0.0;
		}
	}

	double *matr1;
	double *matr2;
	double *matr3;
	cuda_err(cudaMalloc((void **) &matr1, sizeof(double) * a1 * b1));
	cuda_err(cudaMalloc((void **) &matr2, sizeof(double) * a2 * b2));
	cuda_err(cudaMalloc((void **) &matr3, sizeof(double) * a1 * b2));
	cuda_err(cudaMemcpyAsync(matr1, m1, sizeof(double) * a1 * b1, cudaMemcpyHostToDevice));
	cuda_err(cudaMemcpyAsync(matr2, m2, sizeof(double) * a2 * b2, cudaMemcpyHostToDevice));
	cuda_err(cudaMemcpyAsync(matr3, m3, sizeof(double) * a1 * b2, cudaMemcpyHostToDevice));

	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	diffVec<<<4, 4>>>(matr1, matr2, matr3, a1, b1, a2, b2);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	
	
	std::cout << "Время работы при конфигурации ядра (4, 4): " << time << " ms\n";

	cuda_err(cudaGetLastError());    
	cuda_err(cudaMemcpy(m3, matr3, sizeof(double) * a1 * b2, cudaMemcpyDeviceToHost));

	cudaFree(matr1);
	cudaFree(matr2);
	cudaFree(matr3);
	
	printf("Результат умножения матриц A x B:\n");
	for (int i = 0; i < a1; i++) {
		for (int j = 0; j < b2; j++) {
			printf(" %.10e", m3[i * b2 + j]);
		}
		printf("\n");
	}
   
	delete[] m1;
	delete[] m2;
	delete[] m3;
    
	return 0;	
    
}
