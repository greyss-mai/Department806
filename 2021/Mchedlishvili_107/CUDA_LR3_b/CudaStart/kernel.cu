#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#define N 1048576*16
#define EPS 1e-7

#define A -6.0f
#define B 6.0f

__device__ float Function(float x) {
	return sinf(2.2f * x) - x;
}

template<class T> struct Dychotomy {
	T step;
	Dychotomy(T _step) {
		step = _step;
	}
	__device__ T operator()(T& x) const {
		float a = x;
		float b = x + step;

		float f_a = Function(a);
		float f_b = Function(b);

		float half = (b - a) / 2;
		
		while ((b - a) > EPS) {
			if (f_a * f_b > 0)
				return NULL;

			float f_xm = Function(a + half);

			if (f_xm * f_a <= 0) {
				b -= half;
				f_b = f_xm;
			}
			else {
				a += half;
				f_a = f_xm;
			}
			half /= 2;
		}
		
		return (a + b) / 2;
	}
};

int main()
{
	std::cout << "<-------------------------------GPU-------------------------------->\n";

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	float step = fabs(A - B) / N;

	thrust::host_vector<float> data(N);
	thrust::sequence(data.begin(), data.end(), A, step);

	thrust::device_vector<float> input = data;
	thrust::device_vector<float> output(input.size());

	thrust::host_vector<float> result(input.size());

	Dychotomy<float> f(step);

	thrust::transform(input.begin(), input.end(), output.begin(), f);
	thrust::copy(output.begin(), output.end(), result.begin());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	for (size_t i = 0; i < result.size(); i++) {
		if(result[i] != NULL)
			std::cout << result[i] << std::endl;
	}
		

	printf("DEVICE GPU compute time: %.10f milliseconds\n\n", gpuTime);

	return 0;
}

