#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void kernel(int *arr1, int n) { 
    for (int i = 0; i < n/2; i++) {
		int tmp = arr1[i];
        arr1[i] = arr1[n - i - 1];
        arr1[n - i - 1] = tmp;
    }
}

double get_ms() {
    struct timespec _t;
    clock_gettime(CLOCK_REALTIME, &_t);
    return _t.tv_sec*1000 + (_t.tv_nsec/1.0e6);
}

int main(){
	int *arrHost = (int *)malloc(512*sizeof(int));
	int i;
	
	for (i = 0; i < 512; i++)
		arrHost[i] = i + 1;

    double tstart = get_ms();
	
    kernel(arrHost, 512);
    
    double tend   = get_ms();
    fprintf(stderr, "time = %g\n", tend - tstart);
    
    for (i = 0; i < 512; i++){
		printf("%d ", arrHost[i]);
	}	
	printf("\n");
	free(arrHost);
	return 0;
}
