// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

int main() {
// CHECK: #pragma omp taskgraph
#pragma omp taskgraph
{}
// CHECK: int foo = 0;
// CHECK-NEXT: #pragma omp taskgraph
  int foo = 0;
#pragma omp taskgraph
{
  foo++;
}
// CHECK: #pragma omp taskgraph
  for(int i = 0; i < 10; ++i)
#pragma omp taskgraph
{
  #pragma omp task
    foo++;
}
  return 0;
}

#endif
