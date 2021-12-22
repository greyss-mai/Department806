#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include "newHeader.h"

#define MASKA_W  3
#define RADIUS MASKA_W/2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + MASKA_W - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))


__global__ void svertka(float* I, const float* __restrict__ M, float* P, int width, int height)
{
    __shared__ float N_ds[w][w];


    // Загрузка первой партии
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x, destY = dest / w, destX = dest % w,
        srcY = blockIdx.y * TILE_WIDTH + destY - RADIUS, srcX = blockIdx.x * TILE_WIDTH + destX - RADIUS,
        src = srcY * width + srcX;

    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
    {
        N_ds[destY][destX] = I[src];
    }
    else
    {
        N_ds[destY][destX] = 0;
    }

    for (int i = 1; i <= (w * w) / (TILE_WIDTH * TILE_WIDTH); i++)
    {
        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = dest / w, destX = dest % w;
        srcY = blockIdx.y * TILE_WIDTH + destY - RADIUS;
        srcX = blockIdx.x * TILE_WIDTH + destX - RADIUS;
        src = srcY * width + srcX;
        if (destY < w)
        {
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
                N_ds[destY][destX] = I[src];
            else
                N_ds[destY][destX] = 0;
        }
    }
    __syncthreads();

    float iac_cum = 0;
    int y, x;
    for (y = 0; y < MASKA_W; y++)
    {
        for (x = 0; x < MASKA_W; x++)
        {
            iac_cum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * MASKA_W + x];
        }

    }


    y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if (y < height && x < width)
        P[y * width + x] = iac_cum;

    __syncthreads();

}

void bilateral_filtering()
{

    float weight[3][3] = { { -1,  0,  1 },
                           { -2,  0,  2 },
                           { -1,  0,  1 } };
    float pixel_value;

    int x, y, i, j;
    float* Image_input_from_device_point;
    float* Image_output_from_device_poin;
    float* deviceMask;

    cudaMalloc((void**)&Image_input_from_device_point, x_size1 * y_size1 * sizeof(float));
    cudaMalloc((void**)&Image_output_from_device_poin, x_size1 * y_size1 * sizeof(float));
    cudaMalloc((void**)&deviceMask, 3 * 3 * sizeof(float));

    cudaMemcpy(Image_input_from_device_point, image1, x_size1 * y_size1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMask, weight, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);



    x_size2 = x_size1;
    y_size2 = y_size1;
    for (y = 0; y < y_size2; y++)
    {
        for (x = 0; x < x_size2; x++)
        {
            image2[y][x] = 0;
        }
    }

    dim3 dimGrid(ceil((float)x_size1 / TILE_WIDTH), ceil((float)y_size1 / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    svertka << <dimGrid, dimBlock >> > (Image_input_from_device_point, deviceMask, Image_output_from_device_poin, x_size1, y_size1);


    cudaMemcpy(image2, Image_output_from_device_poin, x_size2 * y_size2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(Image_input_from_device_point);
    cudaFree(Image_output_from_device_poin);
    cudaFree(deviceMask);

}


int main()
{
    load_image_data();

    clock_t begin = clock();
    bilateral_filtering();   //Использование билатерального фильтра 
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("\n\nTime: %f\n", time_spent);
    save_image_data();
    return 0;
}