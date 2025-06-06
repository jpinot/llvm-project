; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=+flat-for-global | FileCheck -check-prefix=HSA -check-prefix=HSA-DEFAULT -check-prefix=ALL %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-flat-for-global | FileCheck -check-prefix=HSA -check-prefix=HSA-NODEFAULT -check-prefix=ALL %s
; RUN: llc < %s -mtriple=amdgcn-- -mcpu=tonga | FileCheck -check-prefix=HSA-NOADDR64 -check-prefix=ALL %s
; RUN: llc < %s -mtriple=amdgcn-- -mcpu=kaveri -mattr=-flat-for-global | FileCheck -check-prefix=NOHSA-DEFAULT -check-prefix=ALL %s
; RUN: llc < %s -mtriple=amdgcn-- -mcpu=kaveri -mattr=+flat-for-global | FileCheck -check-prefix=NOHSA-NODEFAULT -check-prefix=ALL %s
; RUN: llc < %s -mtriple=amdgcn-- -mcpu=tonga | FileCheck -check-prefix=NOHSA-NOADDR64 -check-prefix=ALL %s


; There are no stack objects even though flat is used by default, so
; flat_scratch_init should be disabled.

; ALL-LABEL: {{^}}test:

; ALL-NOT: flat_scr

; HSA-DEFAULT: flat_store_dword
; HSA-NODEFAULT: buffer_store_dword
; HSA-NOADDR64: flat_store_dword

; HSA: .amdhsa_user_sgpr_flat_scratch_init 0

; NOHSA-DEFAULT: buffer_store_dword
; NOHSA-NODEFAULT: flat_store_dword
; NOHSA-NOADDR64: flat_store_dword
define amdgpu_kernel void @test(ptr addrspace(1) %out) #0 {
entry:
  store i32 0, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}test_addr64:

; HSA-DEFAULT: flat_store_dword
; HSA-NODEFAULT: buffer_store_dword
; HSA-NOADDR64: flat_store_dword

; NOHSA-DEFAULT: buffer_store_dword
; NOHSA-NODEFAULT: flat_store_dword
; NOHSA-NOADDR64: flat_store_dword
define amdgpu_kernel void @test_addr64(ptr addrspace(1) %out) #0 {
entry:
  %out.addr = alloca ptr addrspace(1), align 4, addrspace(5)

  store ptr addrspace(1) %out, ptr addrspace(5) %out.addr, align 4
  %ld0 = load ptr addrspace(1), ptr addrspace(5) %out.addr, align 4

  store i32 1, ptr addrspace(1) %ld0, align 4

  %ld1 = load ptr addrspace(1), ptr addrspace(5) %out.addr, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(1) %ld1, i32 1
  store i32 2, ptr addrspace(1) %arrayidx1, align 4

  ret void
}

attributes #0 = { "amdgpu-no-flat-scratch-init" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
