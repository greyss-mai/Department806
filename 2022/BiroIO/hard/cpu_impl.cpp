#include <omp.h>
#include <math.h>
#include <chrono>

namespace timer {

    using point = std::chrono::system_clock::time_point;

    enum class unit {
        second,
        millisecond,
        microsecond
    };

    point get_time() { return std::chrono::system_clock::now(); };
    int   get_duration(point& end, point& start, unit time_unit) {
        switch (time_unit)
        {
        case unit::second:
            return std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
        case unit::millisecond:
            return std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        case unit::microsecond:
            return std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
        default:
            return -1;
        }
    }

}

int CPU_mat_add(float* dst, float* mat1, float* mat2, int N, int M) {

    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);

    timer::point start = timer::get_time();

    for(int i=0; i < N * M; i++) {
        dst[i] = mat1[i] + mat2[i];
    }

    timer::point end = timer::get_time();

    return timer::get_duration(end, start, timer::unit::microsecond);
}