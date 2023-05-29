// RUN: %libomp-tdg-compile-and-run
#include "../square.h"
#include <stdbool.h>

int test_parallel_execution_one_static_one_dynamic()
{
    bool volatile firstFinished = false;
    bool parallelExecution = true;

    init_matrix();

    #pragma omp parallel shared(parallelExecution)
    #pragma omp single
    {
      for (int j= 0; j < 2 ; j++) {
        firstFinished = false;
        #pragma omp taskgraph tdg_type(dynamic) nowait
        {
          for (int i=0; i<NUM_ITER / 2; i++)
            wavefront(square);

          #pragma omp task
          {
            sleep(1);
            firstFinished = true;
          }
        }

        #pragma omp taskgraph tdg_type(static)
        {
          #pragma omp task
          {
            if(j == 1 && firstFinished)
              parallelExecution = false;
          }
        }
      }
    }

    //The parallel execution works because main thread starts executing tasks from the tail instead the head once it finishes the creation, and the task from the second taskgraph is the tail of the queue
    if (vanilla_result_8 == square[N - 1][N - 1][BS - 1][BS - 1] &&
        parallelExecution)
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
        if(!test_parallel_execution_one_static_one_dynamic()) {
            num_failed++;
        }
    }
    return num_failed;
}
