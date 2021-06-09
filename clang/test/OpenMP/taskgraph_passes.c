// RUN: %clang_cc1 -O0 -fopenmp -fopenmp-taskgraph -emit-llvm-bc -o /dev/null \
// RUN:   -fexperimental-new-pass-manager -fdebug-pass-manager %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-NOPASS %s
// RUN: %clang_cc1 -O2 -fopenmp -fopenmp-taskgraph -emit-llvm-bc -o /dev/null \
// RUN:   -fexperimental-new-pass-manager -fdebug-pass-manager %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-NEW-PM %s
// RUN: %clang_cc1 -O0 -fopenmp -fopenmp-taskgraph -emit-llvm-bc -o /dev/null \
// RUN:   -flegacy-pass-manager -mllvm -debug-pass=Structure %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-NOPASS %s
// RUN: %clang_cc1 -O2 -fopenmp -fopenmp-taskgraph -emit-llvm-bc -o /dev/null \
// RUN:   -flegacy-pass-manager -mllvm -debug-pass=Structure %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-LEGACY-PM %s

// CHECK-NEW-PM: Running pass: StaticTDGIdentPass on {{.*}}taskgraph_passes.c
// CHECK-NEW-PM: Running analysis: StaticTDGAnalysis on foo
// CHECK-NEW-PM: Running analysis: StaticTDGAnalysis on __captured_stmt
// CHECK-NEW-PM: Running analysis: StaticTDGAnalysis on .omp_outlined.
// CHECK-NEW-PM: Running analysis: StaticTDGAnalysis on .omp_task_privates_map.
// CHECK-NEW-PM: Running analysis: StaticTDGAnalysis on .omp_task_entry.

// CHECK-LEGACY-PM: Static TDG task identifier calculation
// CHECK-LEGACY-PM:   FunctionPass Manager
// CHECK-LEGACY-PM:     Dominator Tree Construction
// CHECK-LEGACY-PM:     Natural Loop Information
// CHECK-LEGACY-PM:     Scalar Evolution Analysis
// CHECK-LEGACY-PM:     Static TDG Analysis

// CHECK-NOPASS-NOT: TDG

void bar(int i);

void foo(void)
{
#pragma omp taskgraph
  {
    for (int i = 0; i < 1000; i++)
    {
#pragma omp task
      {
        bar(i);
      }
    }
  }
}
