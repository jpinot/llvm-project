// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -verify -fopenmp -emit-llvm \
// RUN:            -emit-llvm -o - %s

extern void foo();

void empty_taskgraph(void) {
#pragma omp taskgraph // // expected-error {{taskgraph directive must be enabled using -fopenmp-taskgraph}}
  {
    foo();
  }
}

