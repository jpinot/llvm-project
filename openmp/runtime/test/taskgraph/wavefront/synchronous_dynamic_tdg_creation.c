// REQUIRES: ompx_taskgraph
// RUN: %libomp-compile-and-run
#include "square.h"
#include <stdbool.h>

#undef REPETITIONS
#define REPETITIONS 1

int test_synchronous_dynamic_tdg_creation() {
  bool volatile firstFinished = false;
  bool parallelCreation = true;
  bool parallelExecution = true;

  init_matrix();

#pragma omp parallel shared(parallelCreation, parallelExecution)
#pragma omp single
  {
    for (int j = 0; j < NUM_ITER; j++) {
      firstFinished = false;
#pragma omp taskgraph
      {
        wavefront(square);

#pragma omp task
        {
          sleep(1);
          firstFinished = true;
        }
      }

#pragma omp taskgraph
      {
        if (j == 0 && firstFinished)
          parallelCreation = false;

#pragma omp task
        {
          if (j >= 1 && firstFinished)
            parallelExecution = false;
        }
      }
    }
  }

  // The parallel execution works because main thread starts executing tasks
  // from the tail instead the head once it finishes the creation, and the task
  // from the second taskgraph is the tail of the queue
  if (!parallelCreation && parallelExecution &&
      vanilla_result_8 == square[N - 1][N - 1][BS - 1][BS - 1])
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
    if (!test_synchronous_dynamic_tdg_creation()) {
      num_failed++;
    }
  }
  return num_failed;
}
