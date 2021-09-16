#include <iostream>
#include <time.h>

double host_sqrt(int numElements)
{
    size_t size = numElements * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
    }

    clock_t start = clock();
    for (int i = 0; i < numElements; ++i)
    {
        h_C[i] = sqrtf(h_A[i]);
    }
    clock_t stop = clock();
    double elapsed = (double)(stop - start)*1000.0 / CLOCKS_PER_SEC;

    return elapsed;
}

int main()
{
    int pow = 6;
    int numElements = 100;
    while (numElements <= numElements*pow) {
        printf("For %d elements time elapsed in ms: %f\n", 10, host_sqrt(10));
        numElements *= 10;
    }
}
