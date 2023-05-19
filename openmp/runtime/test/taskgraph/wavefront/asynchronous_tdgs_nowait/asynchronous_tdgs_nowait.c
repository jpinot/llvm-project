// RUN: %libomp-tdg-compile-and-run
#include "../square.h"
#include <stdbool.h>

int test_asynchronous_tdgs_nowait()
{
  bool volatile firstFinished = false;
  bool parallelCreation = true;

  init_matrix();

  #pragma omp parallel shared(parallelCreation)
  #pragma omp single
  {
    #pragma omp taskgraph tdg_type(static) nowait
    {
      for (int i = 0; i < NUM_ITER / 2; i++)
        wavefront(square);

      #pragma omp task
      {
        sleep(2);
        firstFinished = true;
      }
    }

    #pragma omp taskgraph tdg_type(dynamic) nowait
    {
      if(firstFinished)
          parallelCreation = false;

      for (int i = NUM_ITER / 2; i < NUM_ITER; i++)
        wavefront(square);
    }
  }

  if (parallelCreation)
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
        if(!test_asynchronous_tdgs_nowait()) {
            num_failed++;
        }
    }
    return num_failed;
}
