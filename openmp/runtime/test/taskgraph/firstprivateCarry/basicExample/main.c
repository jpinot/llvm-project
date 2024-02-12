//RUN:%libomp-tdg-compile-and-run
#include <stdio.h>
#include <omp.h>
#include "../firstprivateCarry.h"
int taskgraph_recap = 0;
int task_fp = 0;

int foo(int i) {
  return i;
}

int test_basic() {
  #pragma omp parallel
  #pragma omp single
  {
    int a = 0;
    for (int i = 0; i < NUM_ITER; ++i) {
      #ifdef TDG
      #pragma omp taskgraph recapture(a)
      #endif
      {
        #pragma omp task firstprivate(a) // task 1
        {
         taskgraph_recap = foo(a);
        }
      }
      #pragma omp task firstprivate(a) // task 2
      {
        task_fp = foo(a);
      }
      a++;
    }
  }

  if (taskgraph_recap != task_fp)
    return 0;

  return 1;
}


int main() {

  int num_failed=0;

  if (omp_get_max_threads() < 2)
    omp_set_num_threads(8);

  for(int i = 0; i < REPETITIONS; ++i) {
    if(!test_basic())
      num_failed++;
  }

  return num_failed;
}
