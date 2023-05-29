// RUN: %libomp-tdg-compile-and-run
#include "../square.h"

int test_static_tdg()
{
    init_matrix();
    #pragma omp parallel num_threads(1)
    #pragma omp single
    {
        for (int iter = 0; iter < NUM_ITER; iter++) {
        #pragma omp taskgraph tdg_type(static)
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
        if(!test_static_tdg()) {
            num_failed++;
        }
    }
    return num_failed;
}
