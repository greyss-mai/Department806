#include "../book.h"
#include "../cpu_bitmap.h"

#define DIM 1000

namespace cpu_julia
{
    /* структура для хранения комплексного числа */
    struct cuComplex
    {
        float   r;
        float   i;
        cuComplex(float a, float b) : r(a), i(b)
        {
        }

        float magnitude2(void)
        {
            return r * r + i * i;
        }

        cuComplex operator*(const cuComplex& a)
        {
            return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
        }
        cuComplex operator+(const cuComplex& a)
        {
            return cuComplex(r + a.r, i + a.i);
        }
    };

    /*
      Определяет принадлежит ли точка фракталу Джулия.
      Если точка принадлежит фракталу - возвращает 1, иначе возвращает ноль
    */
    int julia(int x, int y)
    {
        /*
          преобразуем координаты пикселя в координаты на комплексной плоскости.
        */
        const float scale = 1.5; // коэф масштабирования
        /* устанавливаем центр плоскости */
        float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
        float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

        /* определяем принадлежность точки к фракталу */
        cuComplex c(-0.8, 0.156);
        cuComplex a(jx, jy);

        int i = 0;
        for (i = 0; i < 200; i++)
        {
            a = a * a + c;
            if (a.magnitude2() > 1000)
                return 0;
        }
        return 1;
    }

    /*
      Принимает указатель на изображение. Обходит все точки которые мы собирнаемся визуализировать
      и для каждой вызывает функцию julia, далее устанавливает цвет точки
    */
    void kernel(unsigned char* ptr)
    {
        for (int y = 0; y < DIM; y++)
        {
            for (int x = 0; x < DIM; x++)
            {
                int offset = x + y * DIM;

                int juliaValue = julia(x, y);
                ptr[offset * 4 + 0] = 255 * juliaValue;
                ptr[offset * 4 + 1] = 0;
                ptr[offset * 4 + 2] = 0;
                ptr[offset * 4 + 3] = 255;
            }
        }
    }

    /*
      Создает растровое изображение нужного размера
    */
    void cpu_main()
    {
        CPUBitmap bitmap(DIM, DIM);
        unsigned char* ptr = bitmap.get_ptr();

        kernel(ptr);

        bitmap.display_and_exit();
    }
}


namespace gpu_julia
{
    struct cuComplex
    {
        float   r;
        float   i;
        __device__ cuComplex(float a, float b) : r(a), i(b)
        {

        }
        __device__ float magnitude2(void)
        {
            return r * r + i * i;
        }
        __device__ cuComplex operator*(const cuComplex& a)
        {
            return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
        }
        __device__ cuComplex operator+(const cuComplex& a)
        {
            return cuComplex(r + a.r, i + a.i);
        }
    };

    __device__ int julia(int x, int y)
    {
        const float scale = 1.5;
        float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
        float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

        cuComplex c(-0.8, 0.156);
        cuComplex a(jx, jy);

        int i = 0;
        for (i = 0; i < 200; i++)
        {
            a = a * a + c;
            if (a.magnitude2() > 1000)
                return 0;
        }
        return 1;
    }

    __global__ void kernel(unsigned char* ptr)
    {
        int x = blockIdx.x;
        int y = blockIdx.y;
        int offset = x + y * gridDim.x;

        int juliaValue = julia(x, y);
        ptr[offset * 4 + 0] = 255 * juliaValue;
        ptr[offset * 4 + 1] = 0;
        ptr[offset * 4 + 2] = 0;
        ptr[offset * 4 + 3] = 255;
    }

    struct DataBlock
    {
        unsigned char* dev_bitmap;
    };

    void gpu_main(void)
    {
        DataBlock   data;
        CPUBitmap bitmap(DIM, DIM, &data);
        unsigned char* dev_bitmap;

        HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
        data.dev_bitmap = dev_bitmap;

        dim3 grid(DIM, DIM);
        kernel << <grid, 1 >> > (dev_bitmap);

        cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

        cudaFree(dev_bitmap);

        bitmap.display_and_exit();
    }
}

int main(void)
{
    cpu_julia::cpu_main();
    gpu_julia::gpu_main();
}