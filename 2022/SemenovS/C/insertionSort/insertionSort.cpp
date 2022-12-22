// insertionSort.cpp : программа выполняет сортировку вставками. 
// Функция generate создаёт массив случайных чисел
// Функция insertionSort сортирует массив

#include <stdio.h>
#include <stdlib.h>

void generate(int *array, int n) {
    for (int i = 0; i < n; i++) array[i] = rand();
}

void insertionSort(int* array, int n) {
    for (int i = 1; i < n; i++) {
        int j = i;
        int temp = array[j];
        while (array[j-1]>temp && j>0) {
            array[j] = array[j-1];
            --j;
        }
        array[j] = temp;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    int n = 10;
    int *array = new int[n];

    generate(array, n);

    //output to the screen
    for (int i = 0; i < n; i++) printf("%d ", array[i]);

    insertionSort(array, n);

    //output to the screen
    printf("\n");
    for (int i = 0; i < n; i++) printf("%d ", array[i]);
    
    delete[] array;
    system("pause");
    return 0;
}