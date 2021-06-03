//===- llvm/Analysis/StaticTDG.h - StaticTDG Analysis -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

struct StaticData {
  int NumberOfTasks;
  SmallVector<StaticTaskInfo, 4> Tasks;
  SmallVector<Loop *, 4> FinalTaskLoops;
};

class StaticTDGLegacyPass : public FunctionPass {
private:
  // Info used by the transform pass
  StaticData FinalData;

  int NumberOfTasks = 0;

  void calculateTasks(Function &F, DominatorTree &DT, LoopInfo &LI,
                      ScalarEvolution &SE);

public:
  static char ID;

  StaticTDGLegacyPass();

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "Static TDG Analysis"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void releaseMemory() override;

  StaticData getTaskData();
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_OMPSSREGIONANALYSIS_H

