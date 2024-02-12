//RUN:%libomp-tdg-compile-and-run
#include <stdio.h>
#include <omp.h>
#include "../firstprivateCarry.h"

int taskgraph_1_fp = 0;
int taskgraph_2_fp = 0;

int foo(int i) {
  return i;
}

int test_two_taskgraphs() {
  #pragma omp parallel
  #pragma omp single
  {
    int b = 100;
    for (int i = 0; i < NUM_ITER; ++i) {
      #ifdef TDG
      #pragma omp taskgraph recapture(b) // Note: when executing this taskgraph
                                           // we will be updating task2.private.b
                                           // but it's not a problem(?), because we need
                                           // to update it anyway, though we paid EXTRA
                                           // cost
      #endif
      {
        #pragma omp task firstprivate(b) // task 1
        {
         taskgraph_1_fp = foo(b);
        }
      }
      b++;
    }

    int c = b;
    for (int i = 0; i < NUM_ITER; ++i) {
      #ifdef TDG
      #pragma omp taskgraph recapture(c)
      #endif
      {
        #pragma omp task firstprivate(c) // task 2
        taskgraph_2_fp = foo(c);
      }
      c--;
    }
  }

  // difference is 8 because 1_fp should be 109 and 2_fp must be 101
  if (taskgraph_1_fp - taskgraph_2_fp != 8)
    return 0;
  else
    return 1;

  return 0;
}

int main()
{
    int i;
    int num_failed=0;

    if (omp_get_max_threads() < 2)
        omp_set_num_threads(8);
    
    for(i = 0; i < REPETITIONS; i++) {
        if(!test_two_taskgraphs()) {
            num_failed++;
        }
    }
    return num_failed;
}
