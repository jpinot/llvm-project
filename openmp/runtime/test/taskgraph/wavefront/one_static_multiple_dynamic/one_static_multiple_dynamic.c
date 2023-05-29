// RUN: %libomp-tdg-compile-and-run
#include "../square.h"

int test_one_static_multiple_dynamic()
{
    init_matrix();
    #pragma omp parallel
    #pragma omp single
    {
        for (int i = 0; i < NUM_ITER / 4; i++) {
        #pragma omp taskgraph tdg_type(static)
        wavefront(square);
        }

        for (int i = NUM_ITER / 4; i < NUM_ITER / 2; i++){
        #pragma omp taskgraph tdg_type(dynamic)
        wavefront(square);
        }

        for (int i = NUM_ITER / 2; i < NUM_ITER; i++) {
        #pragma omp taskgraph tdg_type(dynamic)
        wavefront(square);
        }
    }

    if (vanilla_result_8 ==  square[N - 1][N - 1][BS - 1][BS - 1])
        return 1;

    return 0;
}

int main()
{
    int i;
    int num_failed=0;

    vanilla_result_computation();

    if (omp_get_max_threads() < 2)
        omp_set_num_threads(8);
    
    for(i = 0; i < REPETITIONS; i++) {
        if(!test_one_static_multiple_dynamic()) {
            num_failed++;
        }
    }
    return num_failed;
}
