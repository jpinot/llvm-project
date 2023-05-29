// RUN: %libomp-tdg-compile-and-run
#include "../square.h"

int test_task_replication_vanilla()
{
  init_matrix();
  int error = 0;

  int dummy = 2;
  int nreplicas_detected = 0;

  #pragma omp parallel
  #pragma omp single
  {
    for (int i=0; i<NUM_ITER; i++) {
        #pragma omp task shared(dummy, nreplicas_detected) replicated(3, dummy, isEqual)
        {
            #pragma omp atomic
            dummy++;
            #pragma omp atomic
            nreplicas_detected++;
        }
        wavefront(square);
    }
    #pragma omp taskwait
    if (2+NUM_ITER != dummy) error = 1;
    if (NUM_ITER*4 != nreplicas_detected) error = 1;
  }
 
  if (vanilla_result_8 == square[N - 1][N - 1][BS - 1][BS - 1] && !error)
    return 1;

  return 0;
}

int main()
{
    int i;
    int num_failed=0;

    vanilla_result_computation();

    if (omp_get_max_threads() < 2)
        omp_set_num_threads(8);
    
    for(i = 0; i < REPETITIONS; i++) {
        if(!test_task_replication_vanilla())
            num_failed++;
    }

    return num_failed;
}
