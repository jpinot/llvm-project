//===- llvm/Analysis/StaticTDG.h - StaticTDG Analysis -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ANALYSIS_STATICTDGANALYSIS_H
#define LLVM_ANALYSIS_STATICTDGANALYSIS_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

// End Analysis data structures

struct StaticTaskInfo {
  int id;
  SmallVector<Loop *, 4> loops;
  int maxIteration;
  Instruction *TaskAllocInstruction;
};

struct StaticTDGInfo {
  int NumberOfTasks;
  SmallVector<StaticTaskInfo, 4> Tasks;
  SmallVector<Loop *, 4> FinalTaskLoops;
};

// This is used in both legacy and new passes.
class StaticTDGData {
  // Info used by the transform pass
  StaticTDGInfo Info;
  int NumberOfTasks = 0;

  public:
  void calculateTasks(Function &F, DominatorTree &DT, LoopInfo &LI,
                      ScalarEvolution &SE);
  StaticTDGInfo getTaskData() const { return Info; }
};

class StaticTDGLegacyPass : public FunctionPass {
private:
  StaticTDGData TDGInfo;

public:
  static char ID;

  StaticTDGLegacyPass();

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "Static TDG Analysis"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void releaseMemory() override;

  StaticTDGInfo getTaskData();
};

// New PassManager Analysis
class StaticTDGAnalysis : public AnalysisInfoMixin<StaticTDGAnalysis> {
  friend AnalysisInfoMixin<StaticTDGAnalysis>;

  static AnalysisKey Key;

public:
  /// Provide the result typedef for this analysis pass.
  using Result = StaticTDGInfo;
  StaticTDGInfo run(Function &F, FunctionAnalysisManager &FAM);
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_OMPSSREGIONANALYSIS_H

