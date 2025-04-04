// REQUIRES: ompx_taskgraph
// RUN: %libomp-cxx-compile-and-run

#include <omp.h>
#include <stdio.h>

#define NUM_ITERS 4
#define N 16384
#define NUM_TASKS 200
#define REPETITIONS 1

int vec_x[N];
int vec_y[N];

void init() {
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    vec_x[i] = 2;
    vec_y[i] = 1;
  }
}

int test_ut_taskloop_reduction() {
  init();
  int sum = 0;
#pragma omp parallel
#pragma omp single
  {
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
#pragma omp taskgraph
#pragma omp taskloop reduction(+ : sum) num_tasks(NUM_TASKS)
      for (int i = 0; i < N; ++i) {
        vec_y[i] += vec_x[i];
        sum += vec_y[i];
      }
    }
  }

  for (int i = 0; i < N; ++i) {
    if (vec_y[i] != 1 + NUM_ITERS * 2)
      return 0;
  }

  int expected_value = 0;
  for (int i = 0; i < NUM_ITERS; ++i)
    expected_value += (vec_y[0] - vec_x[0] * i) * N;

  if (sum != expected_value)
    return 0;

  return 1;
}

int main(int argc, char **argv) {
  int i;
  int num_failed = 0;

  if (omp_get_max_threads() < 2)
    omp_set_num_threads(8);

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_ut_taskloop_reduction())
      num_failed++;
  }

  return num_failed;
}
