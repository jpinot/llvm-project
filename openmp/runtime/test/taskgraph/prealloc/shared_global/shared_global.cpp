// RUN: %libomp-prealloc-compile-and-run
#include "../prealloc.h"
#include <stdio.h>
#include <stdbool.h>
#include <omp.h>

#define REPETITIONS 1

int test_shared_global()
{
  bool ret = 1;

  for (int j = 0; j < 4; j++)
    for (int z = 0; z < 4; z++)
      e_global[j][z] = 4.1;

  for (int j = 0; j < 2; j++)
    for (int z = 0; z < 2; z++)
        for (int y = 0; y < 2; y++)
          f_global[j][z][y] = 2;

  #pragma omp parallel
  #pragma omp single
  {
    for (int i=0 ; i < NUM_ITER; i++)
    #pragma omp taskgraph tdg_type(static)
    {
    #pragma omp task shared(a_global, b_global, c_global, d_global,          \
                                  e_global, f_global, g_global)
    {
      int local;
      a_global++;
      b_global++;
      c_global = &local;

      for (int j = 0; j < 2; j++)
        d_global[j]++;

      for (int j = 0; j < 4; j++)
        for (int z = 0; z < 4; z++)
          e_global[j][z]++;

      for (int j = 0; j < 2; j++)
        for (int z = 0; z < 2; z++)
          for (int y = 0; y < 2; y++)
            f_global[j][z][y]++;

      g_global.g_1++;
      g_global.g_2++;
      g_global.g_3 = &local;
      g_global.g_4[0]++;
      g_global.g_4[1]++;

      if (a_global != 4 + i) ret = 0;
      if (b_global != 4.2 + i) ret = 0;
      if (c_global != &local) ret = 0;
      for (int j = 0; j < 2; j++)
        if (d_global[j] != 3+i) ret = 0;

      for (int j = 0; j < 4; j++)
        for (int z = 0; z < 4; z++)
          if (e_global[j][z] != 5.1+i) ret = 0;

      for (int j = 0; j < 2; j++)
        for (int z = 0; z < 2; z++)
          for (int y = 0; y < 2; y++)
            if (f_global[j][z][y] != 3+i) ret = 0;

      if (g_global.g_1 != 4+i) ret = 0;
      if (g_global.g_2 != 4.2+i) ret = 0;
      if (g_global.g_3 != &local) ret = 0;
      for (int j = 0; j < 2; j++)
          if (g_global.g_4[j] != 3+i) ret = 0;
    }
    }
  }
  return ret;
}

int main()
{
    int i;
    int num_failed=0;

    if (omp_get_max_threads() < 2)
        omp_set_num_threads(8);
  
    for(i = 0; i < REPETITIONS; i++) {
        if (!test_shared_global()) 
            num_failed++;
    }

    return num_failed;
}
