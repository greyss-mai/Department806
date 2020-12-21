#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>

//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
      
#include <cstdlib>
#include <ctime>

#include <curand.h>

#include <stdio.h>
#include <string>
#include <winsock.h>
// includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

using namespace std;

typedef unsigned long long uint64;

// Нахождение НОД
__host__ __device__ uint64 gcd(uint64 u, uint64 v) {
  uint64 shift;
  if (u == 0) return v;
  if (v == 0) return u;
  for (shift = 0; ((u | v) & 1) == 0; ++shift) {
    u >>= 1;
    v >>= 1;
  }
    
  while ((u & 1) == 0)
    u >>= 1;
    
  do {
    while ((v & 1) == 0)
      v >>= 1;
    
    if (u > v) {
      uint64 t = v; v = u; u = t;}
    v = v - u; 
  } while (v != 0);
  
  return u << shift;
}

__host__ __device__ bool prime(uint64 n){ 
	for(uint64 i=2;i<=sqrt(n);i++)
		if(n%i==0)
			return false;
	return true;
}

__global__ void clearPara(uint64 * da, uint64 * dc, uint64 m) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  da[idx] = da[idx] % (m - 1) + 1;
  dc[idx] = dc[idx] % (m - 1) + 1;
}

__global__ void pollardKernel(uint64 num, uint64 * resultd, uint64 * dx, uint64 * dy, uint64 * da, uint64 * dc) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  uint64 n = num;
  uint64 x, y, a, c;
  x = dx[idx];
  y = dy[idx];
  a = da[idx];
  c = dc[idx];

  x = (a * x * x + c) % n;
  y =  a * y * y + c;
  y = (a * y * y + c) % n;

  uint64 z = x > y ? (x - y) : (y - x);
  uint64 d = gcd(z, n);
  
  dx[idx] = x;
  dy[idx] = y;

  if (d != 1 && d != n) *resultd = d;
}

uint64 pollard(uint64 num)
{
  uint64 upper = sqrt(num), result = 0;
  int nT = 256, nB = 256;
  if (num % 2 == 0) return 2;
  if (num % 3 == 0) return 3;
  if (num % 5 == 0) return 5;
  if (num % 7 == 0) return 7;
  if (upper * upper == num) return upper;
  if (prime(num)) return num;
	
  uint64 *resultd = NULL, *dx = NULL, *dy = NULL, *da = NULL, *dc = NULL;
  cudaMalloc((void**)&resultd, sizeof(uint64));
  cudaMemset(resultd, 0, sizeof(uint64));
  cudaMalloc((void**)&dx, nB * nT * sizeof(uint64));
  cudaMalloc((void**)&dy, nB * nT * sizeof(uint64));
  cudaMalloc((void**)&da, nB * nT * sizeof(uint64));
  cudaMalloc((void**)&dc, nB * nT * sizeof(uint64));  
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64);
  curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
  curandGenerateLongLong(gen, da, nB * nT);
  curandGenerateLongLong(gen, dc, nB * nT);
  cudaMemset(dx, 0, nB * nT * sizeof(uint64));
  cudaMemset(dy, 0, nB * nT * sizeof(uint64));
	// nB - gridSize nT - blockSize
  clearPara<<<nB, nT>>>(da, dc, upper);

 while(result == 0) {
    pollardKernel<<<nB, nT>>>(num, resultd, dx, dy, da, dc);
    cudaMemcpy(&result, resultd, sizeof(uint64), cudaMemcpyDeviceToHost);
  }
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(da);
  cudaFree(dc);
  cudaFree(resultd);
  curandDestroyGenerator(gen);
  return result;
}

uint64 pollardhost(uint64 num)
{
  uint64 upper = sqrt(num), result = 0;

  if (num % 2 == 0) return 2;
  if (num % 3 == 0) return 3;
  if (num % 5 == 0) return 5;
  if (num % 7 == 0) return 7;  

  if (upper * upper == num) return upper;
  if (prime(num)) return num;

  bool quit = false;

  uint64 x = 0;
  uint64 a = rand() % (upper-1) + 1;
  uint64 c = rand() % (upper-1) + 1;
  uint64 y, d;

  y = x;
  d = 1;

  do {
    x = (a * x * x + c) % num;
    y =  a * y * y + c;
    y = (a * y * y + c) % num;
    uint64 z = x > y ? (x - y) : (y - x);
    d = gcd(z, num);
  } while (d == 1 && !quit);


  if (d != 1 && d != num ) {
    quit = true;
    result = d;
  }
    
  return result;
}

uint64 pollardhost1(uint64 num)
{
int result = 0;
	while(result == 0) {
		 result =  pollardhost(num);
		}
  return result;
}


int main()
{
	tryAgain: // это лейбл
  //getTime();
  srand(time(NULL));
	
  auto elapsedTimeInMsGPU = 0.0f;
  float elapsedTimeInMsCPU = 0.0f;
  StopWatchInterface *timerCPU = NULL;
  StopWatchInterface *timerGPU = NULL;

  uint64 n = 0;
  printf("Input num: ");
  scanf("%d", &n);             //задаем размер
  uint64 num = n;
  //
  uint64 result;
  uint64 prevNum;
  string res1;
  string res2;
  string res3;
  string res4;
  string res5;
  string res6;
  string res7;
  uint64 rslt;
  string resultString;
  const char * resultStr;
  //
		//SDK timer
   sdkCreateTimer(&timerGPU);
   sdkStartTimer(&timerGPU);
	//
   result = pollard(num);	
   prevNum = num/result;
   res1 = "Result(GPU): ";
   res2 = to_string(num);
   res3 = " = ";
   res4 = to_string(result);
   res5 = " * ";
   resultString = res1+res2+res3+res4;  
   while(!prime(prevNum)) 
   {
	 rslt = pollard(prevNum);	
	 prevNum = prevNum/rslt; 	
     res6 = to_string(rslt);
     resultString += res5 + res6;   
  }
  res7 = to_string(prevNum);
  resultString += res5 + res7;
  resultString += "\n";	
  resultStr = resultString.c_str();
	//	
  sdkStopTimer(&timerGPU);
  elapsedTimeInMsGPU = sdkGetTimerValue(&timerGPU);
	
  //printf("Result(GPU): %lld = %lld * %lld\n", num, result, num / result);  
  printf(resultStr);
  printf("Time  : %.6fs\n", elapsedTimeInMsGPU);

  //SDK timer
  sdkCreateTimer(&timerCPU);
  sdkStartTimer(&timerCPU);

  result = pollardhost1(num);	
  prevNum = num/result;
  res1 = "Result(CPU): ";
  res2 = to_string(num);
  res3 = " = ";
  res4 = to_string(result);
  res5 = " * ";
  resultString = res1 + res2 + res3 + res4;  
  while(!prime(prevNum)) 
  {
   rslt = pollardhost1(prevNum);	
   prevNum = prevNum/rslt; 	
   res6 = to_string(rslt);
   resultString += res5 + res6;   
  }
  res7 = to_string(prevNum);
  resultString += res5 + res7;
  resultString += "\n";	
  resultStr = resultString.c_str();

  sdkStopTimer(&timerCPU);
  elapsedTimeInMsCPU = sdkGetTimerValue(&timerCPU);
	
  printf(resultStr);
  printf("Time  : %.6fs\n", elapsedTimeInMsCPU);

  goto tryAgain; // а это оператор goto
	
  return 0;
}




