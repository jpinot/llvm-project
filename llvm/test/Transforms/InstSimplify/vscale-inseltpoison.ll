; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=instsimplify,verify -S | FileCheck %s

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Vector Operations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; insertelement

define <vscale x 4 x i32> @insertelement_idx_undef(<vscale x 4 x i32> %a) {
; CHECK-LABEL: @insertelement_idx_undef(
; CHECK-NEXT:    ret <vscale x 4 x i32> poison
;
  %r = insertelement <vscale x 4 x i32> %a, i32 5, i64 undef
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x i32> @insertelement_value_undef(<vscale x 4 x i32> %a) {
; CHECK-LABEL: @insertelement_value_undef(
; CHECK-NEXT:    [[R:%.*]] = insertelement <vscale x 4 x i32> [[A:%.*]], i32 undef, i64 0
; CHECK-NEXT:    ret <vscale x 4 x i32> [[R]]
;
  %r = insertelement <vscale x 4 x i32> %a, i32 undef, i64 0
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x i32> @insertelement_idx_maybe_out_of_bound(<vscale x 4 x i32> %a) {
; CHECK-LABEL: @insertelement_idx_maybe_out_of_bound(
; CHECK-NEXT:    [[R:%.*]] = insertelement <vscale x 4 x i32> [[A:%.*]], i32 5, i64 4
; CHECK-NEXT:    ret <vscale x 4 x i32> [[R]]
;
  %r = insertelement <vscale x 4 x i32> %a, i32 5, i64 4
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x i32> @insertelement_idx_large_bound(<vscale x 4 x i32> %a) {
; CHECK-LABEL: @insertelement_idx_large_bound(
; CHECK-NEXT:    [[R:%.*]] = insertelement <vscale x 4 x i32> [[A:%.*]], i32 5, i64 12345
; CHECK-NEXT:    ret <vscale x 4 x i32> [[R]]
;
  %r = insertelement <vscale x 4 x i32> %a, i32 5, i64 12345
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x i32> @insert_extract_element_same_vec_idx_1(<vscale x 4 x i32> %a) {
; CHECK-LABEL: @insert_extract_element_same_vec_idx_1(
; CHECK-NEXT:    ret <vscale x 4 x i32> [[A:%.*]]
;
  %v = extractelement <vscale x 4 x i32> %a, i64 1
  %r = insertelement <vscale x 4 x i32> %a, i32 %v, i64 1
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x i32> @insertelement_inline_to_ret() {
; CHECK-LABEL: @insertelement_inline_to_ret(
; CHECK-NEXT:    ret <vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 1, i32 0)
;
  %i = insertelement <vscale x 4 x i32> poison, i32 1, i32 0
  ret <vscale x 4 x i32> %i
}

define <vscale x 4 x i32> @insertelement_shufflevector_inline_to_ret() {
; CHECK-LABEL: @insertelement_shufflevector_inline_to_ret(
; CHECK-NEXT:    ret <vscale x 4 x i32> splat (i32 1)
;
  %i = insertelement <vscale x 4 x i32> poison, i32 1, i32 0
  %i2 = shufflevector <vscale x 4 x i32> %i, <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
  ret <vscale x 4 x i32> %i2
}

; extractelement

define i32 @extractelement_idx_undef(<vscale x 4 x i32> %a) {
; CHECK-LABEL: @extractelement_idx_undef(
; CHECK-NEXT:    ret i32 poison
;
  %r = extractelement <vscale x 4 x i32> %a, i64 undef
  ret i32 %r
}

define i32 @extractelement_vec_undef(<vscale x 4 x i32> %a) {
; CHECK-LABEL: @extractelement_vec_undef(
; CHECK-NEXT:    ret i32 undef
;
  %r = extractelement <vscale x 4 x i32> undef, i64 1
  ret i32 %r
}

define i32 @extractelement_idx_maybe_out_of_bound(<vscale x 4 x i32> %a) {
; CHECK-LABEL: @extractelement_idx_maybe_out_of_bound(
; CHECK-NEXT:    [[R:%.*]] = extractelement <vscale x 4 x i32> [[A:%.*]], i64 4
; CHECK-NEXT:    ret i32 [[R]]
;
  %r = extractelement <vscale x 4 x i32> %a, i64 4
  ret i32 %r
}
define i32 @extractelement_idx_large_bound(<vscale x 4 x i32> %a) {
; CHECK-LABEL: @extractelement_idx_large_bound(
; CHECK-NEXT:    [[R:%.*]] = extractelement <vscale x 4 x i32> [[A:%.*]], i64 12345
; CHECK-NEXT:    ret i32 [[R]]
;
  %r = extractelement <vscale x 4 x i32> %a, i64 12345
  ret i32 %r
}

define i32 @insert_extract_element_same_vec_idx_2() {
; CHECK-LABEL: @insert_extract_element_same_vec_idx_2(
; CHECK-NEXT:    ret i32 1
;
  %v = insertelement <vscale x 4 x i32> poison, i32 1, i64 4
  %r = extractelement <vscale x 4 x i32> %v, i64 4
  ret i32 %r
}

define i32 @insert_extract_element_same_vec_idx_3() {
; CHECK-LABEL: @insert_extract_element_same_vec_idx_3(
; CHECK-NEXT:    ret i32 1
;
  %r = extractelement <vscale x 4 x i32> insertelement (<vscale x 4 x i32> undef, i32 1, i64 4), i64 4
  ret i32 %r
}

define i32 @insert_extract_element_same_vec_idx_4() {
; CHECK-LABEL: @insert_extract_element_same_vec_idx_4(
; CHECK-NEXT:    ret i32 1
;
  %r = extractelement <vscale x 4 x i32> insertelement (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> undef, i32 1, i32 4), i32 2, i64 3), i64 4
  ret i32 %r
}

; more complicated expressions

define <vscale x 2 x i1> @cmp_le_smax_always_true(<vscale x 2 x i64> %x) {
; CHECK-LABEL: @cmp_le_smax_always_true(
; CHECK-NEXT:    ret <vscale x 2 x i1> splat (i1 true)
;
  %cmp = icmp sle <vscale x 2 x i64> %x, splat (i64 9223372036854775807)
  ret <vscale x 2 x i1> %cmp
}

define <vscale x 4 x float> @bitcast() {
; CHECK-LABEL: @bitcast(
; CHECK-NEXT:    ret <vscale x 4 x float> splat (float 0x36A0000000000000)
;
  %i1 = insertelement <vscale x 4 x i32> poison, i32 1, i32 0
  %i2 = shufflevector <vscale x 4 x i32> %i1, <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
  %i3 = bitcast <vscale x 4 x i32> %i2 to <vscale x 4 x float>
  ret <vscale x 4 x float> %i3
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Memory Access and Addressing Operations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define i32 @extractelement_splat_constant_index(i32 %v) {
; CHECK-LABEL: @extractelement_splat_constant_index(
; CHECK-NEXT:    ret i32 [[V:%.*]]
;
  %in = insertelement <vscale x 4 x i32> poison, i32 %v, i32 0
  %splat = shufflevector <vscale x 4 x i32> %in, <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
  %r = extractelement <vscale x 4 x i32> %splat, i32 1
  ret i32 %r
}

define i32 @extractelement_splat_variable_index(i32 %v, i32 %idx) {
; CHECK-LABEL: @extractelement_splat_variable_index(
; CHECK-NEXT:    ret i32 [[V:%.*]]
;
  %in = insertelement <vscale x 4 x i32> poison, i32 %v, i32 0
  %splat = shufflevector <vscale x 4 x i32> %in, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %r = extractelement <vscale x 4 x i32> %splat, i32 %idx
  ret i32 %r
}
