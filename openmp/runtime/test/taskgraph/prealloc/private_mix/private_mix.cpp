// RUN: %libomp-prealloc-compile-and-run
#include "../prealloc.h"
#include <stdio.h>
#include <stdbool.h>
#include <omp.h>

int test_private_mix()
{
    bool ret = 1;

    int a_local = 3;
    double b_local = 3.2;
    void *c_local = &a_local;
    int d_local[2] = {2,2};
    double e_local[4][4];
    int f_local[2][2][2];
    struct g_local_type {
        int g_1;
        double g_2;
        void *g_3;
        int g_4[2];
    };

    struct g_local_type g_local = {.g_1 = a_local, .g_2 = b_local, .g_3 = c_local, .g_4 = {2,2}};

    for (int j = 0; j < 4; j++)
      for (int z = 0; z < 4; z++)
        e_local[j][z] = 4.1;

    for (int j = 0; j < 2; j++)
      for (int z = 0; z < 2; z++)
         for (int y = 0; y < 2; y++)
            f_local[j][z][y] = 2;

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
    #pragma omp task private(a_global, b_local, c_global, d_local,    \
                                  e_local, f_global, g_local)
         {
            if (a_global != -2) ret = 0;
            if (b_local != -2) ret = 0;
            if (c_global != (void *) -2) ret = 0;
            for (int j = 0; j < 2; j++)
              if (d_local[j] != -2) ret = 0;

            for (int j = 0; j < 4; j++)
              for (int z = 0; z < 4; z++)
                if (e_local[j][z] != -2) ret = 0;

            for (int j = 0; j < 2; j++)
              for (int z = 0; z < 2; z++)
                for (int y = 0; y < 2; y++)
                  if (f_global[j][z][y] != -2) ret = 0;

            if (g_local.g_1 != -2) ret = 0;
            if (g_local.g_2 != -2) ret = 0;
            if (g_local.g_3 != (void *)-2) ret = 0;
            for (int j = 0; j < 2; j++)
                if (g_local.g_4[j] != -2) ret = 0;

            int local;
            a_global++;
            b_local = 5.2;
            c_global = &local;

            for (int j = 0; j < 2; j++)
              d_local[j] = 3;

            for (int j = 0; j < 4; j++)
              for (int z = 0; z < 4; z++)
                e_local[j][z] = 6.2;

            for (int j = 0; j < 2; j++)
              for (int z = 0; z < 2; z++)
                for (int y = 0; y < 2; y++)
                  f_global[j][z][y] = 7;

            g_local = {.g_1 = 9, .g_2 = 3.6, .g_3 = &local, .g_4 = {11,11}};

            if (a_global != -1) ret  = 0;
            if (b_local != 5.2) ret = 0;
            if (c_global != &local) ret = 0;
            for (int j = 0; j < 2; j++)
              if (d_local[j] != 3) ret = 0;

            for (int j = 0; j < 4; j++)
              for (int z = 0; z < 4; z++)
                if (e_local[j][z] != 6.2) ret = 0;

            for (int j = 0; j < 2; j++)
              for (int z = 0; z < 2; z++)
                for (int y = 0; y < 2; y++)
                  if (f_global[j][z][y] != 7) ret = 0;

            if (g_local.g_1 != 9) ret = 0;
            if (g_local.g_2 != 3.6) ret = 0;
            if (g_local.g_3 != &local) ret = 0;
            for (int j = 0; j < 2; j++)
                if (g_local.g_4[j] != 11) ret = 0;
         }
        }

        if (a_global != 3) ret = 0;
        if (b_local != 3.2) ret = 0;
        if (c_global != &a_global) ret = 0;
        for (int j = 0; j < 2; j++)
         if (d_local[j] != 2) ret = 0;

        for (int j = 0; j < 4; j++)
         for (int z = 0; z < 4; z++)
            if (e_local[j][z] != 4.1) ret = 0;

        for (int j = 0; j < 2; j++)
         for (int z = 0; z < 2; z++)
            for (int y = 0; y < 2; y++)
              if (f_global[j][z][y] != 2) ret = 0;

        if (g_local.g_1 != a_local) ret = 0;
        if (g_local.g_2 != b_local) ret = 0;
        if (g_local.g_3 != c_local) ret = 0;
        for (int j = 0; j < 2; j++)
         if (g_local.g_4[j] != 2) ret = 0;
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
        if (!test_private_mix())
            num_failed++;
    }
    return num_failed;
}
