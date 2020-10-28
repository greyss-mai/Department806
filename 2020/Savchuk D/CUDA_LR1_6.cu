#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

#define CSC(call) {                                 \
    cudaError_t err = call;                         \
    if(err != cudaSuccess) {                        \
        fprintf(stderr, "CUDA Error in file %s in line %d: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(0);                                    \
    }                                               \
} while(0)

#define N 65535


__global__ void add_kernel(int *a)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += offset)
        a[i] = i + 1;
}


int main(void)
{
	int a[N];
	int *dev_a;

	int threadsPerBlock = 1024;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));
	CSC(cudaEventRecord(start, 0));

	CSC(cudaMalloc((void**)&dev_a, N * sizeof(int)));

	add_kernel<<<blocksPerGrid, threadsPerBlock >>>(dev_a);

	CSC(cudaMemcpy(a, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost));

	CSC(cudaThreadSynchronize());
	CSC(cudaEventRecord(stop, 0));
	CSC(cudaEventSynchronize(stop));
	CSC(cudaEventElapsedTime(&gpuTime, start, stop));

	//for (int i = 0; i < N; ++i)
	//	printf("%d\t", a[i]);

	printf("\n===================   GPU    ===================\n");
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	printf("DEVICE GPU compute time: %f milliseconds\n", gpuTime);

	CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));
    CSC(cudaFree(dev_a));
    free(a);

	// CPU
	double time_CPU;
    int a2[N];
    auto t1 = Clock::now();

	for (int i = 0; i < N; ++i)
		a2[i] = i + 1;

	//for (int i = 0; i < N; ++i)
		//printf("%d\t", a2[i]);

	auto t2 = Clock::now();
    time_CPU = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

	printf("\n===================   CPU    ===================\n");
	printf("HOST CPU compute time: %f milliseconds\n", time_CPU);

    time_CPU = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    printf("HOST CPU compute time: %f microseconds\n", time_CPU);

    free(a2);

	return 0;
}