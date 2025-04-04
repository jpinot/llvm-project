// REQUIRES: ompx_taskgraph
// RUN: %libomp-compile-and-run
#include "axpy.h"

#include <stdio.h>
#include <unistd.h>

#define REPETITIONS 1

int test_synchronous_dynamic_tdg_creation() {
  double *x, *y;
  x = (double *)malloc(sizeof(double) * N);
  y = (double *)malloc(sizeof(double) * N);

  for (int i = 0; i < N; ++i) {
    x[i] = y[i] = 1;
  }

  bool volatile firstFinished = false;
  bool parallelCreation = true;
  bool parallelExecution = true;
#pragma omp parallel shared(x, y, parallelCreation, parallelExecution)
#pragma omp single
  {
    for (int j = 0; j < 2; j++) {
      firstFinished = false;
#pragma omp taskgraph
      {
        for (int i = 0; i < NUM_ITER / 2; i++)
          saxpy(x, y);

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
          if (j == 1 && firstFinished)
            parallelExecution = false;
        }
      }
    }
  }

  if (parallelCreation)
    return 0;
  // The parallel execution works because main thread starts executing tasks
  // from the tail instead the head once it finishes the creation, and the task
  // from the second taskgraph is the tail of the queue
  if (!parallelExecution)
    return 0;
  for (int i = 0; i < NUM_ITER; i++)
    if (x[i] * (NUM_ITER) + 1 != y[i])
      return 0;

  free(x);
  free(y);

  return 1;
}

int main() {
  int i;
  int num_failed = 0;

  if (omp_get_max_threads() < 2)
    omp_set_num_threads(8);

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_synchronous_dynamic_tdg_creation()) {
      num_failed++;
    }
  }

  return num_failed;
}
