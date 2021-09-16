

//CPU//CPU//CPU//CPU//CPU//CPU//CPU//CPU//CPU//CPU//CPU//CPU//CPU//CPU//CPU//CPU//CPU//CPU

#include <stdio.h>
#include <time.h>

#define SIZE	16
void Square(int* a, int* c, int n)
{
	int i;

	for (i = 0; i < n; ++i)
		c[i] = a[i] * a[i];
}

int main()
{
	float elapsedTime;
	clock_t start = clock();

	int *a, *c;
	a = (int*)malloc(SIZE * sizeof(int));
	c = (int*)malloc(SIZE * sizeof(int));

	for (int i = 0; i < SIZE; ++i)
	{
		a[i] = i;
		c[i] = 0;
	}

	Square(a, c, SIZE);

	for (int i = 0; i < SIZE; ++i)
		printf("c[%d] = %d\n", i, c[i]);

	free(a);
	free(c);

	elapsedTime = ((double)clock() - start) / CLOCKS_PER_SEC;
	printf("CPU time elapsed: %f seconds \n", elapsedTime);

	return 0;
}