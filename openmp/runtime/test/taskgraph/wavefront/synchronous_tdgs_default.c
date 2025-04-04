// REQUIRES: ompx_taskgraph
// RUN: %libomp-compile-and-run
#include "square.h"
#include <stdbool.h>

int test_synchronous_tdgs_default() {
  bool volatile firstFinished = false;
  bool parallelCreation = true;

  init_matrix();

#pragma omp parallel shared(parallelCreation)
#pragma omp single
  {
#pragma omp taskgraph
    {
      for (int i = 0; i < NUM_ITER / 2; i++)
        wavefront(square);

#pragma omp task
      {
        sleep(1);
        firstFinished = true;
      }
    }

#pragma omp taskgraph
    {
      if (firstFinished)
        parallelCreation = false;

      for (int i = NUM_ITER / 2; i < NUM_ITER; i++)
        wavefront(square);
    }
  }

  if (!parallelCreation)
    return 1;

  return 0;
}

int main() {
  int i;
  int num_failed = 0;

  vanilla_result_computation();

  if (omp_get_max_threads() < 2)
    omp_set_num_threads(8);

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_synchronous_tdgs_default()) {
      num_failed++;
    }
  }

  return num_failed;
}
