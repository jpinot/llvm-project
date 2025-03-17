// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm -fexceptions -fcxx-exceptions -o - %s | FileCheck %s

// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm -fexceptions -fcxx-exceptions -o - %s | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

int exe(int graph_id) {
  int foo = 0;
  // CHECK: [[GRAPH_ID:%.*]] = load i32, ptr {{.*}}, align 4
  // CHECK: call void @__kmpc_taskgraph(ptr @{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 [[GRAPH_ID]], ptr @{{.*}}.omp_outlined{{.*}}, ptr {{.*}})
  #pragma omp taskgraph graph_id(graph_id)
  {
    #pragma omp task
    foo++;
  }
  return foo;
}

int main() {
  int foo = 0;
  exe(0);
  exe(1);
  exe(0);

  // CHECK: call void @__kmpc_taskgraph(ptr @{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 0, ptr @{{.*}}.omp_outlined{{.*}}, ptr {{.*}})
  #pragma omp taskgraph
  {
    #pragma omp task
    foo++;
  }

  return 0;
}
