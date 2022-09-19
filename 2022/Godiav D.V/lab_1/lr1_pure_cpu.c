// 4.	Вывести на экран числа от 1 до 65535
#include <stdlib.h>
#include <time.h>
#include "stdio.h"

// Сишная часть программы, CPU часть
void fillArray(long long n, long long* array)
{
    // int* array = (double*)malloc(n * sizeof(double));
    for (long long i = 0; i < n; i++)
    {
        array[i] = i * i * i;
    }

}

int main() {
    //timer part
    struct timeval start, stop;
    double secs = 0;
    // pure C part
    // long long n = 65365;
    long long n;
    scanf("%d", &n);
    long long* c_array = (long long*) malloc(n * sizeof(long long));
    gettimeofday(&start, NULL);
    fillArray(n, c_array);
    gettimeofday(&stop, NULL);
    secs = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
    printf("----------------------------------");
    printf("time taken %f\n",secs);
    printf("----------------------------------");
    for (long long i = 0; i < n; i++) {
		// printf("%d \n", c_array[i]);
		printf("%d ", c_array[i]);
	}
    free(c_array);
    system ( "PAUSE" );
    return 0;
}