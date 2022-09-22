#include <iostream>
#include <string>

int CPU_sqrt(float* array, int N);
int GPU_sqrt(float* array, int N);


float* create_array(int N);
void   destroy_array(float* ptr);

int main(int argc, char** argv) {

    int N = 1024;
    int NUM_TESTS = 100;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    std::cout << "N: " << N << '\n';

    float* array = create_array(N);
    {
        int min_duration = 100000000;
        for(int i=0; i < NUM_TESTS; i++) {
            int duration = CPU_sqrt(array, N);
            if (duration < min_duration) {
                min_duration = duration;
            }
        }
        std::cout << "CPU_time: " << min_duration << " microseconds.\n";
    }
    {
        int min_duration = 100000000;
        for(int i=0; i < NUM_TESTS; i++) {
            int duration = GPU_sqrt(array, N);
            if (duration < min_duration) {
                min_duration = duration;
            }
        }
        std::cout << "GPU_time: " << min_duration << " microseconds.\n";
    }
    destroy_array(array);

    return 0;
}

float* create_array(int N) {
    float* ptr = new float[N];
    for (int i = 0; i < N; i++) ptr[i] = i;
    return ptr;
}

void destroy_array(float* ptr) { delete[] ptr; }