//A simple hello world program
#include <stdio.h>

void __global__ mykernel() {

}

int main(){
	mykernel <<<4, 16>>> ();
    printf("Hello Wolrd!");
    return 0;
}
