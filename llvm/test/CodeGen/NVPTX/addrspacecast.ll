; RUN: llc -O0 < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s -check-prefixes=ALL,CLS32
; RUN: llc -O0 < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s -check-prefixes=ALL,NOPTRCONV,CLS64
; RUN: llc -O0 < %s -mtriple=nvptx64 -mcpu=sm_20 --nvptx-short-ptr | FileCheck %s -check-prefixes=ALL,PTRCONV,CLS64
; RUN: %if ptxas && !ptxas-12.0 %{ llc -O0 < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc -O0 < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc -O0 < %s -mtriple=nvptx64 -mcpu=sm_20 --nvptx-short-ptr | %ptxas-verify %}

; ALL-LABEL: conv1
define i32 @conv1(ptr addrspace(1) %ptr) {
; CLS32: cvta.global.u32
; ALL-NOT: cvt.u64.u32
; CLS64: cvta.global.u64
; ALL: ld.b32
  %genptr = addrspacecast ptr addrspace(1) %ptr to ptr
  %val = load i32, ptr %genptr
  ret i32 %val
}

; ALL-LABEL: conv2
define i32 @conv2(ptr addrspace(3) %ptr) {
; CLS32: cvta.shared.u32
; PTRCONV: cvt.u64.u32
; NOPTRCONV-NOT: cvt.u64.u32
; CLS64: cvta.shared.u64
; ALL: ld.b32
  %genptr = addrspacecast ptr addrspace(3) %ptr to ptr
  %val = load i32, ptr %genptr
  ret i32 %val
}

; ALL-LABEL: conv3
define i32 @conv3(ptr addrspace(4) %ptr) {
; CLS32: cvta.const.u32
; PTRCONV: cvt.u64.u32
; NOPTRCONV-NOT: cvt.u64.u32
; CLS64: cvta.const.u64
; ALL: ld.b32
  %genptr = addrspacecast ptr addrspace(4) %ptr to ptr
  %val = load i32, ptr %genptr
  ret i32 %val
}

; ALL-LABEL: conv4
define i32 @conv4(ptr addrspace(5) %ptr) {
; CLS32: cvta.local.u32
; PTRCONV: cvt.u64.u32
; NOPTRCONV-NOT: cvt.u64.u32
; CLS64: cvta.local.u64
; ALL: ld.b32
  %genptr = addrspacecast ptr addrspace(5) %ptr to ptr
  %val = load i32, ptr %genptr
  ret i32 %val
}

; ALL-LABEL: conv5
define i32 @conv5(ptr %ptr) {
; CLS32: cvta.to.global.u32
; ALL-NOT: cvt.u64.u32
; CLS64: cvta.to.global.u64
; ALL: ld.global.b32
  %specptr = addrspacecast ptr %ptr to ptr addrspace(1)
  %val = load i32, ptr addrspace(1) %specptr
  ret i32 %val
}

; ALL-LABEL: conv6
define i32 @conv6(ptr %ptr) {
; CLS32: cvta.to.shared.u32
; CLS64: cvta.to.shared.u64
; PTRCONV: cvt.u32.u64
; NOPTRCONV-NOT: cvt.u32.u64
; ALL: ld.shared.b32
  %specptr = addrspacecast ptr %ptr to ptr addrspace(3)
  %val = load i32, ptr addrspace(3) %specptr
  ret i32 %val
}

; ALL-LABEL: conv7
define i32 @conv7(ptr %ptr) {
; CLS32: cvta.to.const.u32
; CLS64: cvta.to.const.u64
; PTRCONV: cvt.u32.u64
; NOPTRCONV-NOT: cvt.u32.u64
; ALL: ld.const.b32
  %specptr = addrspacecast ptr %ptr to ptr addrspace(4)
  %val = load i32, ptr addrspace(4) %specptr
  ret i32 %val
}

; ALL-LABEL: conv8
define i32 @conv8(ptr %ptr) {
; CLS32: cvta.to.local.u32
; CLS64: cvta.to.local.u64
; PTRCONV: cvt.u32.u64
; NOPTRCONV-NOT: cvt.u32.u64
; ALL: ld.local.b32
  %specptr = addrspacecast ptr %ptr to ptr addrspace(5)
  %val = load i32, ptr addrspace(5) %specptr
  ret i32 %val
}

; ALL-LABEL: conv9
define i32 @conv9(ptr addrspace(1) %ptr) {
; CLS32:     // implicit-def: %[[ADDR:r[0-9]+]]
; PTRCONV:   // implicit-def: %[[ADDR:r[0-9]+]]
; NOPTRCONV: // implicit-def: %[[ADDR:rd[0-9]+]]
; ALL: ld.shared.b32 %r{{[0-9]+}}, [%[[ADDR]]]
  %specptr = addrspacecast ptr addrspace(1) %ptr to ptr addrspace(3)
  %val = load i32, ptr addrspace(3) %specptr
  ret i32 %val
}

; Check that we support addrspacecast when splitting the vector
; result (<2 x ptr> => 2 x <1 x ptr>).
; This also checks that scalarization works for addrspacecast
; (when going from <1 x ptr> to ptr.)
; ALL-LABEL: split1To0
define void @split1To0(ptr nocapture noundef readonly %xs) {
; CLS32: cvta.global.u32
; CLS32: cvta.global.u32
; CLS64: cvta.global.u64
; CLS64: cvta.global.u64
; ALL: st.b32
; ALL: st.b32
  %vec_addr = load <2 x ptr addrspace(1)>, ptr %xs, align 16
  %addrspacecast = addrspacecast <2 x ptr addrspace(1)> %vec_addr to <2 x ptr>
  %extractelement0 = extractelement <2 x ptr> %addrspacecast, i64 0
  store float 0.5, ptr %extractelement0, align 4
  %extractelement1 = extractelement <2 x ptr> %addrspacecast, i64 1
  store float 1.0, ptr %extractelement1, align 4
  ret void
}

; Same as split1To0 but from 0 to 1, to make sure the addrspacecast preserve
; the source and destination addrspaces properly.
; ALL-LABEL: split0To1
define void @split0To1(ptr nocapture noundef readonly %xs) {
; CLS32: cvta.to.global.u32
; CLS32: cvta.to.global.u32
; CLS64: cvta.to.global.u64
; CLS64: cvta.to.global.u64
; ALL: st.global.b32
; ALL: st.global.b32
  %vec_addr = load <2 x ptr>, ptr %xs, align 16
  %addrspacecast = addrspacecast <2 x ptr> %vec_addr to <2 x ptr addrspace(1)>
  %extractelement0 = extractelement <2 x ptr addrspace(1)> %addrspacecast, i64 0
  store float 0.5, ptr addrspace(1) %extractelement0, align 4
  %extractelement1 = extractelement <2 x ptr addrspace(1)> %addrspacecast, i64 1
  store float 1.0, ptr addrspace(1) %extractelement1, align 4
  ret void
}

; Check that we support addrspacecast when a widening is required
; (3 x ptr => 4 x ptr).
; ALL-LABEL: widen1To0
define void @widen1To0(ptr nocapture noundef readonly %xs) {
; CLS32: cvta.global.u32
; CLS32: cvta.global.u32
; CLS32: cvta.global.u32

; CLS64: cvta.global.u64
; CLS64: cvta.global.u64
; CLS64: cvta.global.u64

; ALL: st.b32
; ALL: st.b32
; ALL: st.b32
  %vec_addr = load <3 x ptr addrspace(1)>, ptr %xs, align 16
  %addrspacecast = addrspacecast <3 x ptr addrspace(1)> %vec_addr to <3 x ptr>
  %extractelement0 = extractelement <3 x ptr> %addrspacecast, i64 0
  store float 0.5, ptr %extractelement0, align 4
  %extractelement1 = extractelement <3 x ptr> %addrspacecast, i64 1
  store float 1.0, ptr %extractelement1, align 4
  %extractelement2 = extractelement <3 x ptr> %addrspacecast, i64 2
  store float 1.5, ptr %extractelement2, align 4
  ret void
}

; Same as widen1To0 but from 0 to 1, to make sure the addrspacecast preserve
; the source and destination addrspaces properly.
; ALL-LABEL: widen0To1
define void @widen0To1(ptr nocapture noundef readonly %xs) {
; CLS32: cvta.to.global.u32
; CLS32: cvta.to.global.u32
; CLS32: cvta.to.global.u32

; CLS64: cvta.to.global.u64
; CLS64: cvta.to.global.u64
; CLS64: cvta.to.global.u64

; ALL: st.global.b32
; ALL: st.global.b32
; ALL: st.global.b32
  %vec_addr = load <3 x ptr>, ptr %xs, align 16
  %addrspacecast = addrspacecast <3 x ptr> %vec_addr to <3 x ptr addrspace(1)>
  %extractelement0 = extractelement <3 x ptr addrspace(1)> %addrspacecast, i64 0
  store float 0.5, ptr addrspace(1) %extractelement0, align 4
  %extractelement1 = extractelement <3 x ptr addrspace(1)> %addrspacecast, i64 1
  store float 1.0, ptr addrspace(1) %extractelement1, align 4
  %extractelement2 = extractelement <3 x ptr addrspace(1)> %addrspacecast, i64 2
  store float 1.5, ptr addrspace(1) %extractelement2, align 4
  ret void
}
