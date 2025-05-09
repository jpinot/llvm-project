! REQUIRES: flang-supports-f128-math
! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK: fir.call @_FortranAErfF128({{.*}}){{.*}}: (f128) -> f128
! CHECK: fir.call @_FortranAErfF128({{.*}}){{.*}}: (f128) -> f128
  real(16) :: a, b, c
  b = erf(a)
  c = qerf(a)
end
