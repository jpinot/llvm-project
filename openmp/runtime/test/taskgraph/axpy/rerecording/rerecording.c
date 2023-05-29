// RUN: %libomp-tdg-compile-and-run
#include "../axpy.h"

#include <stdio.h>

int test_rerecording()
{
    double *x, *y;
    x = (double*)malloc(sizeof(double) * N);
    y = (double*)malloc(sizeof(double) * N);

    for(int i = 0; i < N; ++i){
        x[i] = y[i] = 1;
    }

    int tasks_added = 0;
    #pragma omp parallel shared(x,y,tasks_added)
    #pragma omp single
    {
      for (int i=0; i<NUM_ITER; i++)
      #ifdef TDG
      #pragma omp taskgraph tdg_type(dynamic) if(i==5)
      #endif
      {
        if(i>2)
        {
            #pragma omp task shared(tasks_added)
            {
                tasks_added++;
            }
        }
        saxpy(x, y);
      }
    }

    if (tasks_added != 7) return 0;

    for (int i=0; i<NUM_ITER; i++)
        if (x[i]*(NUM_ITER)+1 != y[i]) return 0;

    free(x);
    free(y);
    
    return 1;
}

int main()
{
    int i;
    int num_failed=0;

    if (omp_get_max_threads() < 2)
        omp_set_num_threads(8);
    
    for(i = 0; i < REPETITIONS; i++) {
        if(!test_rerecording()) {
            num_failed++;
        }
    }

    return num_failed;
}
