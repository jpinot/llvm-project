// Part of Taskgraph directive tests. Verifying the correct
// behaviors of Taskgraph in different scenarios, as defined in RESULT.txt
//
// Using *wavefront* benchmark, with task dependencies

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdbool.h>

#ifndef DIM
#define DIM 2048
#endif

#ifndef N
#define N 8
#endif

#define NUM_ITER 8
#define BS (DIM / N)
#define REPETITIONS 10

unsigned long int square[N][N][BS][BS];
unsigned long int square1[N][N][BS][BS];
unsigned long int square2[N][N][BS][BS];
unsigned long int square3[N][N][BS][BS];

#define SIZE (sizeof(square) / sizeof(long long))

unsigned long vanilla_result_1 = 0; // vanilla_result after 2 iterations
unsigned long vanilla_result_2 = 0; // vanilla_result after 2 iterations
unsigned long vanilla_result_4 = 0; // vanilla_result after 4 iterations
unsigned long vanilla_result_8 = 0; // vanilla_result after 8 iterations

int isEqual(int *original, int *replica) {
  // printf("Ey %d %d \n", *original, *replica);
  bool equal = false;
  if (*original == *replica)
    equal = true;
  // else
  //   printf("Numbers %d %d \n", *original, *replica);

  return equal;
}

void init_matrix() {
  int i, j, x, y;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      for (x = 0; x < BS; x++) {
        for (y = 0; y < BS; y++) {
          square[i][j][x][y] = 2;
          square1[i][j][x][y] = 2;
          square2[i][j][x][y] = 2;
          square3[i][j][x][y] = 2;
        }
      }
    }
  }
}

void sequential(int i, int j, unsigned long square[N][N][BS][BS]) {
  int x, y;

  /*
   * Leftmost upper item. All the neighbors are in different blocks
   */
  if (i > 0 && j > 0) { // _________
    square[i][j][0][0] += // | | # | |
        square[i][j - 1][0][BS - 1] + // ----#----
        square[i - 1][j][BS - 1][0] + // | |3#2| |
        square[i - 1][j - 1][BS - 1][BS - 1]; // ____#____
  } // | |1#o| |
    // ----#----
    // | | # | |
    // ____#____
  /*
   * First column of items. Left neighbors are in the left block
   */
  if (j > 0) { // _________
    for (y = 1; y < BS; y++) { // | | # | |
      square[i][j][y][0] += // ----#-----
          square[i][j - 1][y][BS - 1] + // | | # | |
          square[i][j][y - 1][0] + // ____#____
          square[i][j - 1][y - 1][BS - 1]; // | |3#2| |
    } // ----#----
  } // | |1#o| |
    // ____#____
  /*
   * First row of items. Upper neighbors are in the upper block
   */
  if (i > 0) {
    for (x = 1; x < BS; x++) { // _________
      square[i][j][0][x] += // | | # | |
          square[i][j][0][x - 1] + // ----#----
          square[i - 1][j][BS - 1][x] + // | | #3|2|
          square[i - 1][j][BS - 1][x - 1]; // ____#____
    } // | | #1|o|
  } // ----#----
    // | | # | |
    // ____#_____
  /*
   * Remaining items. All the neighbors are in the same block
   */
  for (x = 1; x < BS; x++) { // _________
    for (y = 1; y < BS; y++) { // |3|1# | |
      square[i][j][y][x] += // ----#----
          square[i][j][y][x - 1] + // |2|o# | |
          square[i][j][y - 1][x] + // ____#____
          square[i][j][y - 1][x - 1]; // | | # | |
    } // ----#----
  } // | | # | |
}

void wavefront(unsigned long square[N][N][BS][BS]) {
  int i, j;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (j == 0 && i == 0) {
#pragma omp task depend(inout : square[i][j])
        { sequential(i, j, square); }
      } else if (i == 0) {
#pragma omp task depend(in : square[i][j - 1]) depend(inout : square[i][j])
        { sequential(i, j, square); }
      } else if (j == 0) {
#pragma omp task depend(in : square[i - 1][j]) depend(inout : square[i][j])
        { sequential(i, j, square); }
      } else {
#pragma omp task depend(in : square[i - 1][j]) depend(in : square[i][j - 1])   \
    depend(in : square[i - 1][j - 1]) depend(inout : square[i][j])
        { sequential(i, j, square); }
      } // final else
    } // j
  } // i
}

void vanilla_result_computation() {
  init_matrix();
#pragma omp parallel
#pragma omp single
  {

    for (int iter = 0; iter < NUM_ITER; iter++) {
      wavefront(square);
#pragma omp taskwait
      if (iter == 0 && vanilla_result_1 == 0)
        vanilla_result_1 = square[N - 1][N - 1][BS - 1][BS - 1];
      if (iter == 1 && vanilla_result_2 == 0)
        vanilla_result_2 = square[N - 1][N - 1][BS - 1][BS - 1];
      if (iter == 3 && vanilla_result_4 == 0)
        vanilla_result_4 = square[N - 1][N - 1][BS - 1][BS - 1];
    }
  }
  if (vanilla_result_8 == 0)
    vanilla_result_8 = square[N - 1][N - 1][BS - 1][BS - 1];
}
