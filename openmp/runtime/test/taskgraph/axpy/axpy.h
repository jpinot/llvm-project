// Part of the CG (Conjugate gradient) method
// Complete source in BAR/Apps/CG from https://pm.bsc.es/svn/BAR

// Part of Taskgraph directive tests. Verifying the correct
// behaviors of Taskgraph in different scenarios, as defined in RESULT.txt
//
// Using *axpy* benchmark, without task dependencies

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>
#include <stdbool.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifndef N
#define N 16777216
// #define N 268435456
#endif
#ifndef BS1
#define BS1 (N / 1024)
#endif
#ifndef BS2
#define BS2 (BS1 / 32)
#endif
#define a 1.0

#define NUM_ITER 10
#define REPLICAS 3
#define REPETITIONS 10

struct timeval t1, t2;

void saxpy(double *x, double *y) {
  for (int i = 0; i < N; i += BS1) {
#ifdef PREALLOC
#pragma omp task dep_check(dynamic)
#else
#pragma omp task // shared(x,y)
#endif
    {
      for (int j = 0; j < BS1; j++) {
        y[i + j] += a * x[i + j];
      }
    }
  }
#pragma omp taskwait
}

int isEqual(int *original, int *replica) {
  // printf("Ey %d %d \n", *original, *replica);
  bool equal = false;
  if (*original == *replica)
    equal = true;
  else
    printf("Numbers %d %d \n", *original, *replica);

  if (!equal)
    return 0;

  return 1;
}