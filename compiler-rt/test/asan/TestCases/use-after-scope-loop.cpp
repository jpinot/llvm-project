// RUN: %clangxx_asan %if MSVC %{ /Od %} %else %{ -O1 %} \
// RUN:     %s -o %t && not %run %t 2>&1 | FileCheck %s

int *p[3];

int main() {
  for (int i = 0; i < 3; i++) {
    int x;
    p[i] = &x;
  }
  return **p;  // BOOM
  // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
  // CHECK: #0 0x{{.*}} in main {{.*}}.cpp:[[@LINE-2]]
}
