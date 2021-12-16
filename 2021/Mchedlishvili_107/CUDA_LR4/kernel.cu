#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>

#include <opencv2/opencv.hpp>

#define PI 3.1415926535

__constant__ float cGaussian[64];

void updateGaussian(int r, double sd)
{
    float fGaussian[64];
    for (int i = 0; i < 2 * r + 1; i++)
    {
        float x = i - r;
        fGaussian[i] = expf(-(x * x) / (2 * sd * sd));
    }
    cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float) * (2 * r + 1));
}

__device__ float euclideanLen(float a, float b, float d)
{
    return expf(-1 * (b - a) / (2.f * d * d));
}

__global__ void bilateral_filter(const unsigned char* input, unsigned char* output, 
    int height, int width, 
    const float* kernel, int kernel_width) 
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    float centralPixel = kernel[col * kernel_width + row];

    if (row < height && col < width) {
        int half = kernel_width / 2;

        float factor;
        float temp = 0.f;
        float sum = 0.f;

        for (int i = -half; i <= half; ++i) 
        {
            for (int j = -half; j <= half; ++j) 
            {
                float currentPixel = kernel[(col + i) * kernel_width + (row + j)];
                factor = (cGaussian[i + half] * cGaussian[j + half]) * euclideanLen(currentPixel, centralPixel, half);
                temp += factor * currentPixel;
                sum += factor;
            }
        }
        output[row * width + col] = (unsigned char)(temp / sum);
    }
}

unsigned char* mat_to_bytes(const cv::Mat& mat) 
{
    auto elem_count = mat.total();
    auto elem_bytes_count = elem_count * sizeof(unsigned char);
    auto bytes = new unsigned char[elem_count];
    std::memcpy(bytes, mat.data, elem_bytes_count);
    return bytes;
}

cv::Mat bytes_to_mat(unsigned char* bytes, int height, int width) 
{
    return cv::Mat(height, width, CV_8UC1, bytes);
}

void init_kernel(float*& kernel, int width, int sigma) 
{
    if (width % 2 == 0) {
        width--;
    }

    int size = width * width;
    kernel = new float[size];
    int half = width / 2;

    for (int i = -half; i <= half; ++i) 
    {
        for (int j = -half; j <= half; ++j) 
        {
            kernel[(i + half) * width + j + half] =
                std::exp((float)-(i * i + j * j) / (float)(2 * sigma * sigma)) /
                (2 * PI * sigma * sigma);
        }
    }

    auto sum = std::accumulate(kernel, kernel + size, 0.0f);
    std::transform(kernel, kernel + size, kernel, [sum](float el) { return el / sum; });
}

cv::Mat bilateral(const cv::Mat& mat) 
{
    auto mat_height = mat.rows;
    auto mat_width = mat.cols;
    auto mat_elem_count = mat.total();
    auto mat_bytes_count = mat_elem_count * sizeof(unsigned char);
    auto input = mat_to_bytes(mat);

    float* kernel = nullptr;
    auto kernel_width = 15;
    auto kernel_size = kernel_width * kernel_width;
    auto kernel_bytes_count = kernel_size * sizeof(float);

    init_kernel(kernel, kernel_width, 1);
    updateGaussian(16, 8);

    unsigned char* dinput;
    cudaMalloc((void**)&dinput, mat_bytes_count);
    cudaMemcpy(dinput, input, mat_bytes_count, cudaMemcpyHostToDevice);

    float* dkernel;

    cudaMalloc((void**)&dkernel, kernel_bytes_count);
    cudaMemcpy(dkernel, kernel, kernel_bytes_count, cudaMemcpyHostToDevice);

    dim3 grid_dim(mat_width / 32, mat_height / 32);
    dim3 block_dim(32, 32, 1);

    bilateral_filter <<<grid_dim, block_dim>>> (dinput, dinput, mat_height, mat_width, dkernel, kernel_width);

    cudaMemcpy(input, dinput, mat_bytes_count, cudaMemcpyDeviceToHost);

    cudaFree(dkernel);
    cudaFree(dinput);

    return bytes_to_mat(input, mat_height, mat_width);
}


//--------------------------------------------------

int main(int argc, char** argv) {

    auto input_file = argv[1];
    auto output_file = argv[2];

    auto img = cv::imread(input_file, cv::IMREAD_COLOR);

    cv::Mat input_img_channels[3];
    cv::split(img, input_img_channels);

    cv::Mat output_img_channels[3];
    for (int c = 0; c < 3; ++c) 
    {
        output_img_channels[c] = bilateral(input_img_channels[c]);
    }

    cv::Mat output_img;
    cv::merge(output_img_channels, 3, output_img);
    cv::imwrite(output_file, output_img);

    return 0;
}