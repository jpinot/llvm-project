//===- StaticTDG.cpp - Static TDG Analysis --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/StaticTDG.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"

using namespace llvm;

char StaticTDGLegacyPass::ID = 0;

StaticTDGLegacyPass::StaticTDGLegacyPass() : FunctionPass(ID) {
  initializeStaticTDGLegacyPassPass(*PassRegistry::getPassRegistry());
}

FunctionPass *createStaticTDGAnalysis() { return new StaticTDGLegacyPass(); }

void StaticTDGData::calculateTasks(Function &F, DominatorTree &DT, LoopInfo &LI,
                                   ScalarEvolution &SE) {

  // dbgs() << "Function: " << F.getName() << "\n";
  SmallVector<BasicBlock *, 8> Worklist;
  SmallPtrSet<BasicBlock *, 8> Visited;

  Worklist.push_back(&F.getEntryBlock());
  Visited.insert(&F.getEntryBlock());
  while (!Worklist.empty()) {
    auto WIt = Worklist.begin();
    BasicBlock *BB = *WIt;
    Worklist.erase(WIt);

    for (Instruction &I : *BB) {
      if (CallInst *TaskCall = dyn_cast<CallInst>(&I)) {
        if (TaskCall->getCalledFunction() &&
            TaskCall->getCalledFunction()->getName() ==
                "__kmpc_set_task_static_id") {

          NumberOfTasks++;

          SmallVector<Loop *, 4> taskloops;

          std::for_each(LI.begin(), LI.end(),
                        [&taskloops, TaskCall](llvm::Loop *loop) {
                          SmallVector<Loop *, 4> auxiliar;
                          if (loop->contains(TaskCall)) {
                            auxiliar = loop->getLoopsInPreorder();

                            for (auto Selectedloop : auxiliar) {
                              if (Selectedloop->contains(TaskCall)) {
                                taskloops.push_back(Selectedloop);
                              }
                            }
                            // dbgs() << "loop contains task creation: " <<
                            // taskloops.size() << " \n";
                          }
                        });

          int max = 0;
          for (auto loop : taskloops) {

            bool exists = false;
            for (auto &storedLoop : Info.FinalTaskLoops) {
              if (storedLoop == loop) {
                exists = true;
                break;
              }
            }
            if (!exists) {
              Info.FinalTaskLoops.push_back(loop);
            }

            int iterations = SE.getSmallConstantTripCount(loop);
            // dbgs() << "Max iter : " << iterations << "\n";
            if (iterations > max)
              max = iterations;
          }

          // dbgs() << "Maximum: " << max << " \n";

          std::reverse(taskloops.begin(), taskloops.end());

          Info.Tasks.push_back({NumberOfTasks, taskloops, max, &I});
          Info.NumberOfTasks = NumberOfTasks;

          // dbgs() << "Encontrado! " <<  NumberOfTasks << " \n";
        }
      }
    }

    for (auto It = succ_begin(BB); It != succ_end(BB); ++It) {
      if (!Visited.count(*It)) {
        Worklist.push_back(*It);
        Visited.insert(*It);
      }
    }
  }
  // dbgs() << "Sale \n";
}

void StaticTDGLegacyPass::releaseMemory() { TDGInfo = StaticTDGData(); }

StaticTDGInfo StaticTDGLegacyPass::getTaskData() { return TDGInfo.getTaskData(); }

bool StaticTDGLegacyPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  TDGInfo.calculateTasks(F, DT, LI, SE);

  return false;
}

void StaticTDGLegacyPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
}

INITIALIZE_PASS_BEGIN(StaticTDGLegacyPass, "static-tdg", "Static tdg", false, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(StaticTDGLegacyPass, "static-tdg", "Static tdg", false, true)

// New PM pass
AnalysisKey StaticTDGAnalysis::Key;

StaticTDGInfo StaticTDGAnalysis::run(
    Function &F, FunctionAnalysisManager &FAM) {
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &LI = FAM.getResult<LoopAnalysis>(F);
  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);

  StaticTDGData TDGInfo;
  TDGInfo.calculateTasks(F, DT, LI, SE);

  return TDGInfo.getTaskData();
}
