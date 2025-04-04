// REQUIRES: ompx_taskgraph
// RUN: %libomp-compile-and-run
#include "axpy.h"

#include <stdio.h>
#include <unistd.h>

#undef REPETITIONS
#define REPETITIONS 1

int test_synchronous_tdgs_default() {
  double *x, *y;
  x = (double *)malloc(sizeof(double) * N);
  y = (double *)malloc(sizeof(double) * N);

  for (int i = 0; i < N; ++i) {
    x[i] = y[i] = 1;
  }

  bool volatile firstFinished = false;
  bool parallelCreation = true;
#pragma omp parallel shared(x, y, parallelCreation)
#pragma omp single
  {
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
      if (firstFinished)
        parallelCreation = false;

      for (int i = NUM_ITER / 2; i < NUM_ITER; i++)
        saxpy(x, y);
    }
  }

  if (parallelCreation)
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
    if (!test_synchronous_tdgs_default()) {
      num_failed++;
    }
  }

  return num_failed;
}
