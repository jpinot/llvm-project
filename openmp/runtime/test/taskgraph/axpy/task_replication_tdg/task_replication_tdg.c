// RUN: %libomp-tdg-compile-and-run
#include "../axpy.h"

#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>

int test_task_replication_tdg()
{
    double *x, *y;
    x = (double*)malloc(sizeof(double) * N);
    y = (double*)malloc(sizeof(double) * N);
    bool ret = 1;
    
    for(int i = 0; i < N; ++i){
        x[i] = y[i] = 1;
    }

    int nreplicas_detected = 0;
    int dummy = 2;

    #pragma omp parallel shared(x,y)
    #pragma omp single
    {
      for (int i=0; i<NUM_ITER; i++) {
        #pragma omp taskgraph tdg_type(dynamic)
        {
          #pragma omp task shared(dummy, nreplicas_detected) replicated(3, dummy, isEqual)
          {
              #pragma omp atomic
              dummy++;
              #pragma omp atomic
              nreplicas_detected++;
          }
          saxpy(x, y);
        }
      }
      if (2+NUM_ITER != dummy) ret = 0;
      if (NUM_ITER*4 != nreplicas_detected) ret = 0;
    }

    for (int i=0; i<NUM_ITER; i++)
        if (x[i]*NUM_ITER+1 != y[i]) ret = 0;

    free(x);
    free(y);
    
    return ret;
}

int main()
{
    int i;
    int num_failed=0;

    if (omp_get_max_threads() < 2)
        omp_set_num_threads(8);
    
    for(i = 0; i < REPETITIONS; i++) {
        if(!test_task_replication_tdg()) {
            num_failed++;
        }
    }
    
    return num_failed;
}
