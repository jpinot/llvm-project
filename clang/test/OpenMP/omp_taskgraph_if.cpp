// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm -fexceptions -fcxx-exceptions -o - %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm -fexceptions -fcxx-exceptions -o - %s | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

int main() {
  int foo = 0;

  // CHECK: call void @__kmpc_taskgraph
  #pragma omp taskgraph if(true)
  {
    #pragma omp task
    foo++;
  }
  // CHECK-NOT: call void @__kmpc_taskgraph
  #pragma omp taskgraph if(false)
  {
    #pragma omp task
    foo++;
  }

  return 0;
}
