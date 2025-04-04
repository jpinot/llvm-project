// REQUIRES: ompx_taskgraph
// RUN: %libomp-compile-and-run
#include "axpy.h"

#include <stdio.h>

int test_multiple_dynamic() {
  double *x, *y;
  x = (double *)malloc(sizeof(double) * N);
  y = (double *)malloc(sizeof(double) * N);

  for (int i = 0; i < N; ++i) {
    x[i] = y[i] = 1;
  }

#pragma omp parallel shared(x, y)
#pragma omp single
  {
    for (int i = 0; i < NUM_ITER / 2; i++)
#ifdef TDG
#pragma omp taskgraph
#endif
    {
      saxpy(x, y);
    }

    for (int i = NUM_ITER / 2; i < NUM_ITER; i++)
#ifdef TDG
#pragma omp taskgraph
#endif
    {
      saxpy(x, y);
    }
  }

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
    if (!test_multiple_dynamic()) {
      num_failed++;
    }
  }

  return num_failed;
}
