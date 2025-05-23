; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 5
; RUN: opt -p loop-unroll -S %s | FileCheck %s

define i64 @peel_single_block_loop_iv_step_1() {
; CHECK-LABEL: define i64 @peel_single_block_loop_iv_step_1() {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ 0, %[[ENTRY]] ], [ [[IV_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[CMP18_NOT:%.*]] = icmp eq i64 [[IV]], 63
; CHECK-NEXT:    [[COND:%.*]] = select i1 [[CMP18_NOT]], i32 10, i32 20
; CHECK-NEXT:    call void @foo(i32 [[COND]])
; CHECK-NEXT:    [[IV_NEXT]] = add i64 [[IV]], 1
; CHECK-NEXT:    [[EC:%.*]] = icmp ne i64 [[IV_NEXT]], 64
; CHECK-NEXT:    br i1 [[EC]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[IV_LCSSA:%.*]] = phi i64 [ [[IV]], %[[LOOP]] ]
; CHECK-NEXT:    ret i64 [[IV_LCSSA]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %cmp = icmp eq i64 %iv, 63
  %cond = select i1 %cmp, i32 10, i32 20
  call void @foo(i32 %cond)
  %iv.next = add i64 %iv, 1
  %ec = icmp ne i64 %iv.next, 64
  br i1 %ec, label %loop, label %exit

exit:
  ret i64 %iv
}

; The predicate %cmp doesn't become known in all iterations after peeling.
define i64 @single_block_loop_iv_step_1_predicate_not_known_true_false_after_peeling() {
; CHECK-LABEL: define i64 @single_block_loop_iv_step_1_predicate_not_known_true_false_after_peeling() {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ 0, %[[ENTRY]] ], [ [[IV_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[UREM:%.*]] = urem i64 [[IV]], 2
; CHECK-NEXT:    [[CMP18_NOT:%.*]] = icmp eq i64 [[UREM]], 1
; CHECK-NEXT:    [[COND:%.*]] = select i1 [[CMP18_NOT]], i32 10, i32 20
; CHECK-NEXT:    call void @foo(i32 [[COND]])
; CHECK-NEXT:    [[IV_NEXT]] = add i64 [[IV]], 1
; CHECK-NEXT:    [[EC:%.*]] = icmp ne i64 [[IV_NEXT]], 64
; CHECK-NEXT:    br i1 [[EC]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[IV_LCSSA:%.*]] = phi i64 [ [[IV]], %[[LOOP]] ]
; CHECK-NEXT:    ret i64 [[IV_LCSSA]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %urem = urem i64 %iv, 2
  %cmp = icmp eq i64 %urem, 1
  %cond = select i1 %cmp, i32 10, i32 20
  call void @foo(i32 %cond)
  %iv.next = add i64 %iv, 1
  %ec = icmp ne i64 %iv.next, 64
  br i1 %ec, label %loop, label %exit

exit:
  ret i64 %iv
}



define i64 @peel_single_block_loop_iv_step_1_eq_pred() {
; CHECK-LABEL: define i64 @peel_single_block_loop_iv_step_1_eq_pred() {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ 0, %[[ENTRY]] ], [ [[IV_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[CMP18_NOT:%.*]] = icmp eq i64 [[IV]], 63
; CHECK-NEXT:    [[COND:%.*]] = select i1 [[CMP18_NOT]], i32 10, i32 20
; CHECK-NEXT:    call void @foo(i32 [[COND]])
; CHECK-NEXT:    [[IV_NEXT]] = add i64 [[IV]], 1
; CHECK-NEXT:    [[EC:%.*]] = icmp eq i64 [[IV_NEXT]], 64
; CHECK-NEXT:    br i1 [[EC]], label %[[EXIT:.*]], label %[[LOOP]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[IV_LCSSA:%.*]] = phi i64 [ [[IV]], %[[LOOP]] ]
; CHECK-NEXT:    ret i64 [[IV_LCSSA]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %cmp = icmp eq i64 %iv, 63
  %cond = select i1 %cmp, i32 10, i32 20
  call void @foo(i32 %cond)
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 64
  br i1 %ec, label %exit, label %loop

exit:
  ret i64 %iv
}

define i64 @peel_single_block_loop_iv_step_1_slt_pred() {
; CHECK-LABEL: define i64 @peel_single_block_loop_iv_step_1_slt_pred() {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ 0, %[[ENTRY]] ], [ [[IV_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[CMP18_NOT:%.*]] = icmp eq i64 [[IV]], 63
; CHECK-NEXT:    [[COND:%.*]] = select i1 [[CMP18_NOT]], i32 10, i32 20
; CHECK-NEXT:    call void @foo(i32 [[COND]])
; CHECK-NEXT:    [[IV_NEXT]] = add i64 [[IV]], 1
; CHECK-NEXT:    [[EC:%.*]] = icmp slt i64 [[IV_NEXT]], 64
; CHECK-NEXT:    br i1 [[EC]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[IV_LCSSA:%.*]] = phi i64 [ [[IV]], %[[LOOP]] ]
; CHECK-NEXT:    ret i64 [[IV_LCSSA]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %cmp = icmp eq i64 %iv, 63
  %cond = select i1 %cmp, i32 10, i32 20
  call void @foo(i32 %cond)
  %iv.next = add i64 %iv, 1
  %ec = icmp slt i64 %iv.next, 64
  br i1 %ec, label %loop, label %exit

exit:
  ret i64 %iv
}

define i64 @peel_single_block_loop_iv_step_1_nested_loop() {
; CHECK-LABEL: define i64 @peel_single_block_loop_iv_step_1_nested_loop() {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    br label %[[OUTER_HEADER:.*]]
; CHECK:       [[OUTER_HEADER]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV_NEXT_LCSSA:%.*]] = phi i64 [ 0, %[[OUTER_HEADER]] ], [ [[IV_NEXT_PEEL:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[CMP18_NOT_PEEL:%.*]] = icmp eq i64 [[IV_NEXT_LCSSA]], 63
; CHECK-NEXT:    [[COND_PEEL:%.*]] = select i1 [[CMP18_NOT_PEEL]], i32 10, i32 20
; CHECK-NEXT:    call void @foo(i32 [[COND_PEEL]])
; CHECK-NEXT:    [[IV_NEXT_PEEL]] = add i64 [[IV_NEXT_LCSSA]], 1
; CHECK-NEXT:    [[EC_PEEL:%.*]] = icmp ne i64 [[IV_NEXT_PEEL]], 64
; CHECK-NEXT:    br i1 [[EC_PEEL]], label %[[LOOP]], label %[[OUTER_LATCH:.*]]
; CHECK:       [[OUTER_LATCH]]:
; CHECK-NEXT:    [[IV_LCSSA:%.*]] = phi i64 [ [[IV_NEXT_LCSSA]], %[[LOOP]] ]
; CHECK-NEXT:    call void @foo(i32 1)
; CHECK-NEXT:    ret i64 [[IV_LCSSA]]
;
entry:
  br label %outer.header

outer.header:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %outer.latch ]
  br label %loop

loop:
  %iv = phi i64 [ 0, %outer.header ], [ %iv.next, %loop ]
  %cmp = icmp eq i64 %iv, 63
  %cond = select i1 %cmp, i32 10, i32 20
  call void @foo(i32 %cond)
  %iv.next = add i64 %iv, 1
  %ec = icmp ne i64 %iv.next, 64
  br i1 %ec, label %loop, label %outer.latch

outer.latch:
  call void @foo(i32 1)
  %outer.iv.next = add i64 %outer.iv, 1
  %outer.ec = icmp ne i64 %outer.iv.next, 100
  br i1 %outer.ec, label %exit, label %outer.header

exit:
  ret i64 %iv
}

define i64 @peel_multi_block_loop_iv_step_1() {
; CHECK-LABEL: define i64 @peel_multi_block_loop_iv_step_1() {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV_NEXT_LCSSA:%.*]] = phi i64 [ 0, %[[ENTRY]] ], [ [[IV_NEXT_PEEL:%.*]], %[[LATCH:.*]] ]
; CHECK-NEXT:    [[CMP18_NOT_PEEL:%.*]] = icmp eq i64 [[IV_NEXT_LCSSA]], 63
; CHECK-NEXT:    [[COND_PEEL:%.*]] = select i1 [[CMP18_NOT_PEEL]], i32 10, i32 20
; CHECK-NEXT:    call void @foo(i32 [[COND_PEEL]])
; CHECK-NEXT:    [[C_PEEL:%.*]] = call i1 @cond()
; CHECK-NEXT:    br i1 [[C_PEEL]], label %[[THEN:.*]], label %[[LATCH]]
; CHECK:       [[THEN]]:
; CHECK-NEXT:    call void @foo(i32 [[COND_PEEL]])
; CHECK-NEXT:    br label %[[LATCH]]
; CHECK:       [[LATCH]]:
; CHECK-NEXT:    [[IV_NEXT_PEEL]] = add i64 [[IV_NEXT_LCSSA]], 1
; CHECK-NEXT:    [[EC_PEEL:%.*]] = icmp ne i64 [[IV_NEXT_PEEL]], 64
; CHECK-NEXT:    br i1 [[EC_PEEL]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[IV_LCSSA:%.*]] = phi i64 [ [[IV_NEXT_LCSSA]], %[[LATCH]] ]
; CHECK-NEXT:    ret i64 [[IV_LCSSA]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %cmp = icmp eq i64 %iv, 63
  %cond = select i1 %cmp, i32 10, i32 20
  call void @foo(i32 %cond)
  %c = call i1 @cond()
  br i1 %c, label %then, label %latch

then:
  call void @foo(i32 %cond)
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp ne i64 %iv.next, 64
  br i1 %ec, label %loop, label %exit

exit:
  ret i64 %iv
}

define i64 @peel_multi_exit_loop_iv_step_1() {
; CHECK-LABEL: define i64 @peel_multi_exit_loop_iv_step_1() {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ 0, %[[ENTRY]] ], [ [[IV_NEXT:%.*]], %[[LATCH:.*]] ]
; CHECK-NEXT:    [[CMP18_NOT:%.*]] = icmp eq i64 [[IV]], 63
; CHECK-NEXT:    [[COND:%.*]] = select i1 [[CMP18_NOT]], i32 10, i32 20
; CHECK-NEXT:    call void @foo(i32 [[COND]])
; CHECK-NEXT:    [[C:%.*]] = call i1 @cond()
; CHECK-NEXT:    br i1 [[C]], label %[[EXIT:.*]], label %[[LATCH]]
; CHECK:       [[LATCH]]:
; CHECK-NEXT:    [[IV_NEXT]] = add i64 [[IV]], 1
; CHECK-NEXT:    [[EC:%.*]] = icmp ne i64 [[IV_NEXT]], 64
; CHECK-NEXT:    br i1 [[EC]], label %[[LOOP]], label %[[EXIT]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[IV_LCSSA:%.*]] = phi i64 [ [[IV]], %[[LATCH]] ], [ [[IV]], %[[LOOP]] ]
; CHECK-NEXT:    ret i64 [[IV_LCSSA]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %cmp = icmp eq i64 %iv, 63
  %cond = select i1 %cmp, i32 10, i32 20
  call void @foo(i32 %cond)
  %c = call i1 @cond()
  br i1 %c, label %exit, label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp ne i64 %iv.next, 64
  br i1 %ec, label %loop, label %exit

exit:
  ret i64 %iv
}


define i64 @peel_single_block_loop_iv_step_1_may_execute_only_once(i64 %n) {
; CHECK-LABEL: define i64 @peel_single_block_loop_iv_step_1_may_execute_only_once(
; CHECK-SAME: i64 [[N:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    [[N_NOT_0:%.*]] = icmp ne i64 [[N]], 0
; CHECK-NEXT:    call void @llvm.assume(i1 [[N_NOT_0]])
; CHECK-NEXT:    [[SUB:%.*]] = add nsw i64 [[N]], 1
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ 0, %[[ENTRY]] ], [ [[IV_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[CMP18_NOT:%.*]] = icmp eq i64 [[IV]], [[N]]
; CHECK-NEXT:    [[COND:%.*]] = select i1 [[CMP18_NOT]], i32 10, i32 20
; CHECK-NEXT:    call void @foo(i32 [[COND]])
; CHECK-NEXT:    [[IV_NEXT]] = add i64 [[IV]], 1
; CHECK-NEXT:    [[EC:%.*]] = icmp ne i64 [[IV_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[EC]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[IV_LCSSA:%.*]] = phi i64 [ [[IV]], %[[LOOP]] ]
; CHECK-NEXT:    ret i64 [[IV_LCSSA]]
;
entry:
  %n.not.0 = icmp ne i64 %n, 0
  call void @llvm.assume(i1 %n.not.0)
  %sub = add nsw i64 %n, 1
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %cmp = icmp eq i64 %iv, %n
  %cond = select i1 %cmp, i32 10, i32 20
  call void @foo(i32 %cond)
  %iv.next = add i64 %iv, 1
  %ec = icmp ne i64 %iv.next, %n
  br i1 %ec, label %loop, label %exit

exit:
  ret i64 %iv
}

define i64 @peel_single_block_loop_iv_step_neg_1() {
; CHECK-LABEL: define i64 @peel_single_block_loop_iv_step_neg_1() {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ 64, %[[ENTRY]] ], [ [[IV_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[CMP18_NOT:%.*]] = icmp eq i64 [[IV]], 1
; CHECK-NEXT:    [[COND:%.*]] = select i1 [[CMP18_NOT]], i32 10, i32 20
; CHECK-NEXT:    call void @foo(i32 [[COND]])
; CHECK-NEXT:    [[IV_NEXT]] = add i64 [[IV]], -1
; CHECK-NEXT:    [[EC:%.*]] = icmp ne i64 [[IV_NEXT]], 0
; CHECK-NEXT:    br i1 [[EC]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[IV_LCSSA:%.*]] = phi i64 [ [[IV]], %[[LOOP]] ]
; CHECK-NEXT:    ret i64 [[IV_LCSSA]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 64, %entry ], [ %iv.next, %loop ]
  %cmp = icmp eq i64 %iv, 1
  %cond = select i1 %cmp, i32 10, i32 20
  call void @foo(i32 %cond)
  %iv.next = add i64 %iv, -1
  %ec = icmp ne i64 %iv.next, 0
  br i1 %ec, label %loop, label %exit

exit:
  ret i64 %iv
}

define i64 @peel_single_block_loop_iv_step_2() {
; CHECK-LABEL: define i64 @peel_single_block_loop_iv_step_2() {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ 0, %[[ENTRY]] ], [ [[IV_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[CMP18_NOT:%.*]] = icmp eq i64 [[IV]], 62
; CHECK-NEXT:    [[COND:%.*]] = select i1 [[CMP18_NOT]], i32 10, i32 20
; CHECK-NEXT:    call void @foo(i32 [[COND]])
; CHECK-NEXT:    [[IV_NEXT]] = add i64 [[IV]], 2
; CHECK-NEXT:    [[EC:%.*]] = icmp ne i64 [[IV_NEXT]], 64
; CHECK-NEXT:    br i1 [[EC]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[IV_LCSSA:%.*]] = phi i64 [ [[IV]], %[[LOOP]] ]
; CHECK-NEXT:    ret i64 [[IV_LCSSA]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %cmp = icmp eq i64 %iv, 62
  %cond = select i1 %cmp, i32 10, i32 20
  call void @foo(i32 %cond)
  %iv.next = add i64 %iv, 2
  %ec = icmp ne i64 %iv.next, 64
  br i1 %ec, label %loop, label %exit

exit:
  ret i64 %iv
}

define i32 @peel_loop_with_branch_and_phi_uses(ptr %x, i1 %c) {
; CHECK-LABEL: define i32 @peel_loop_with_branch_and_phi_uses(
; CHECK-SAME: ptr [[X:%.*]], i1 [[C:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br i1 [[C]], label %[[LOOP_HEADER_PREHEADER:.*]], label %[[EXIT:.*]]
; CHECK:       [[LOOP_HEADER_PREHEADER]]:
; CHECK-NEXT:    br label %[[LOOP_HEADER:.*]]
; CHECK:       [[LOOP_HEADER]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ [[IV_NEXT:%.*]], %[[LOOP_LATCH:.*]] ], [ 0, %[[LOOP_HEADER_PREHEADER]] ]
; CHECK-NEXT:    [[RED:%.*]] = phi i32 [ [[ADD:%.*]], %[[LOOP_LATCH]] ], [ 0, %[[LOOP_HEADER_PREHEADER]] ]
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i32 [[IV]], 99
; CHECK-NEXT:    br i1 [[CMP1]], label %[[IF_THEN:.*]], label %[[LOOP_LATCH]]
; CHECK:       [[IF_THEN]]:
; CHECK-NEXT:    tail call void @foo(i32 10)
; CHECK-NEXT:    br label %[[LOOP_LATCH]]
; CHECK:       [[LOOP_LATCH]]:
; CHECK-NEXT:    [[GEP_X:%.*]] = getelementptr inbounds nuw i32, ptr [[X]], i32 [[IV]]
; CHECK-NEXT:    [[L:%.*]] = load i32, ptr [[GEP_X]], align 4
; CHECK-NEXT:    [[ADD]] = add nsw i32 [[L]], [[RED]]
; CHECK-NEXT:    [[IV_NEXT]] = add nuw nsw i32 [[IV]], 1
; CHECK-NEXT:    [[EC:%.*]] = icmp ne i32 [[IV_NEXT]], 100
; CHECK-NEXT:    br i1 [[EC]], label %[[LOOP_HEADER]], label %[[LOOPEXIT:.*]]
; CHECK:       [[LOOPEXIT]]:
; CHECK-NEXT:    [[ADD_LCSSA:%.*]] = phi i32 [ [[ADD]], %[[LOOP_LATCH]] ]
; CHECK-NEXT:    br label %[[EXIT]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[SUM_0_LCSSA:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ [[ADD_LCSSA]], %[[LOOPEXIT]] ]
; CHECK-NEXT:    ret i32 [[SUM_0_LCSSA]]
;
entry:
  br i1 %c, label %loop.header, label %exit

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %red = phi i32 [ 0, %entry ], [ %add, %loop.latch ]
  %cmp1 = icmp eq i32 %iv, 99
  br i1 %cmp1, label %if.then, label %loop.latch

if.then:
  tail call void @foo(i32 10)
  br label %loop.latch

loop.latch:
  %gep.x = getelementptr inbounds nuw i32, ptr %x, i32 %iv
  %l = load i32, ptr %gep.x, align 4
  %add = add nsw i32 %l, %red
  %iv.next = add nuw nsw i32 %iv, 1
  %ec = icmp ne i32 %iv.next, 100
  br i1 %ec, label %loop.header, label %loopexit

loopexit:
  %add.lcssa = phi i32 [ %add, %loop.latch ]
  br label %exit

exit:
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa, %loopexit ]
  ret i32 %sum.0.lcssa
}

declare void @foo(i32)
declare i1 @cond()
