
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <helper_string.h>

#include "SobelFilter_kernels.h"

// Объект текстуры
cudaTextureObject_t texObject;
//массив на куде
static cudaArray *array = NULL;


#define MAX(a,b) ((a > b) ? a : b)

#define MIN(a,b) ((a > b) ? b : a)

// Функция обработки ошибок
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


__device__ unsigned char
ComputeSobel(unsigned char ul, // upper left
             unsigned char um, // upper middle
             unsigned char ur, // upper right
             unsigned char ml, // middle left
             unsigned char mm, // middle (unused)
             unsigned char mr, // middle right
             unsigned char ll, // lower left
             unsigned char lm, // lower middle
             unsigned char lr // lower right
    )
{
    short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
    short Vert = ul + 2*um + ur - ll - 2*lm - lr;
    short Sum = (short)round(sqrt((float)((int)Horz*(int)Horz +(int)Vert*(int)Vert)));

    if (Sum < 0)
    {
        return 0;
    }
    else if (Sum > 0xff)
    {
        return 0xff;
    }

    return (unsigned char) Sum;
}


//каждый блок - горизонатльная линия изображения, каждый поток - очередной пиксель в линии

__global__ void
SobelCopyImageNew(Pixel* pSobelOriginal, unsigned int Pitch,
    int w, int h, cudaTextureObject_t tex)
{
    //после выполенения кода указатель массива встанет ровно на начало обрабатываемой части. По сути это сдвиг на текущий height (уровень высоты)*ширину картинки 
    unsigned char* pSobel =
        (unsigned char*)(((char*)pSobelOriginal) + blockIdx.x * Pitch);

    
    int i = threadIdx.x;
        //нормализация яркости от 0 до 255
        pSobel[i] = MIN(MAX((tex2D<unsigned char>(tex, (float)i, (float)blockIdx.x)), 0.f), 255.f);
}

__global__ void
SobelTexNew(Pixel* pSobelOriginal, unsigned int Pitch,
    int w, int h, cudaTextureObject_t tex)
{
    //после выполенения кода указатель массива встанет ровно на начало обрабатываемой части. По сути это сдвиг на текущий height (уровень высоты)*ширину картинки 
    unsigned char* pSobel =
        (unsigned char*)(((char*)pSobelOriginal) + blockIdx.x * Pitch);

    //выбор матрицы 3*3 вокруг данного пикселя
    int i = threadIdx.x;
    unsigned char pix00 = tex2D<unsigned char>(tex, (float)i - 1, (float)blockIdx.x - 1);
    unsigned char pix01 = tex2D<unsigned char>(tex, (float)i + 0, (float)blockIdx.x - 1);
    unsigned char pix02 = tex2D<unsigned char>(tex, (float)i + 1, (float)blockIdx.x - 1);
    unsigned char pix10 = tex2D<unsigned char>(tex, (float)i - 1, (float)blockIdx.x + 0);
    unsigned char pix11 = tex2D<unsigned char>(tex, (float)i + 0, (float)blockIdx.x + 0);
    unsigned char pix12 = tex2D<unsigned char>(tex, (float)i + 1, (float)blockIdx.x + 0);
    unsigned char pix20 = tex2D<unsigned char>(tex, (float)i - 1, (float)blockIdx.x + 1);
    unsigned char pix21 = tex2D<unsigned char>(tex, (float)i + 0, (float)blockIdx.x + 1);
    unsigned char pix22 = tex2D<unsigned char>(tex, (float)i + 1, (float)blockIdx.x + 1);
    //расчет значения нового пикселя с помощью оператора Собеля
    pSobel[i] = ComputeSobel(pix00, pix01, pix02,
        pix10, pix11, pix12,
        pix20, pix21, pix22);

    
}

extern "C" void setupTexture(int iw, int ih, Pixel *data)
{
    cudaChannelFormatDesc desc;

    desc = cudaCreateChannelDesc<unsigned char>();

    checkCudaErrors(cudaMallocArray(&array, &desc, iw, ih));
    checkCudaErrors(cudaMemcpy2DToArray(array, 0, 0, data, iw * sizeof(Pixel), 
                                        iw * sizeof(Pixel), ih, cudaMemcpyHostToDevice));

    cudaResourceDesc            texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = array;

    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModePoint;
    texDescr.addressMode[0]   = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    //использование текстурной памяти

    checkCudaErrors(cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));

}

extern "C" void deleteTexture(void)
{
    checkCudaErrors(cudaFreeArray(array));
    checkCudaErrors(cudaDestroyTextureObject(texObject));
}

//В зависимости от режима подготавливает изображение для рендеринга
extern "C" void sobelFilter(Pixel *odata, int iw, int ih, enum SobelDisplayMode mode)
{
    switch (mode)
    {
        case SOBELDISPLAY_IMAGE:
           // SobelCopyImage<<<ih, 1>>>(odata, iw, iw, ih, texObject);
            SobelCopyImageNew << <ih, iw >> > (odata, iw, iw, ih, texObject);
            break;

        case SOBELDISPLAY_SOBELTEX:
           // SobelTex<<<ih, 1>>>(odata, iw, iw, ih, texObject);
            SobelTexNew << <ih, iw >> > (odata, iw, iw, ih, texObject);
            break;
    }
}
