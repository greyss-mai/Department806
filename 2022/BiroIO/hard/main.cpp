#include <iostream>
#include <string>

int CPU_mat_add(float* dst, float* mat1, float* mat2, int N, int M);
int GPU_mat_add(float* dst, float* mat1, float* mat2, int N, int M);


float* create_array(int N, int M);
float* create_mat(int N, int M);

void   destroy_array(float* ptr);

int main(int argc, char** argv) {

    int N = 1024;
    int M = 1024;

    int NUM_TESTS = 100;
    if (argc > 2) {
        N = atoi(argv[1]);
        M = atoi(argv[2]);
    }

    std::cout << "N: " << N << '\n';
    std::cout << "M: " << M << '\n';

    float* dst  = create_array(N, M);
    float* mat1 = create_mat(N, M);
    float* mat2 = create_mat(N, M);
    {
        int min_duration = 100000000;
        for(int i=0; i < NUM_TESTS; i++) {
            int duration = CPU_mat_add(dst, mat1, mat2, N, M);
            if (duration < min_duration) {
                min_duration = duration;
            }
        }
        std::cout << "CPU_time: " << min_duration << " microseconds.\n";
    }
    {
        int min_duration = 100000000;
        for(int i=0; i < NUM_TESTS; i++) {
            int duration = GPU_mat_add(dst, mat1, mat2, N, M);
            if (duration < min_duration) {
                min_duration = duration;
            }
        }
        std::cout << "GPU_time: " << min_duration << " microseconds.\n";
    }
    destroy_array(dst);
    destroy_array(mat1);
    destroy_array(mat2);

    return 0;
}

float* create_array(int N, int M) {
    float* ptr = new float[N * M];
    for (int i = 0; i < N * M; i++) ptr[i] = 0.f;
    return ptr;
}

float* create_mat(int N, int M) {
    float* ptr = new float[N * M];
    for (int i = 0; i < N * M; i++) ptr[i] = static_cast<float>(i);
    return ptr;
}

void destroy_array(float* ptr) { delete[] ptr; }