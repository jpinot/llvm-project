//===- StaticTDGIdent.cpp -- Strip parts of Debug Info --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/StaticTDG.h"
#include "llvm/Transforms/StaticTDG.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
using namespace llvm;

#define DEBUG_TYPE "static-tdg-id"

namespace {

using StaticTDGInfoForFunctionTy = function_ref<StaticTDGInfo(Function &)>;

struct StaticTDGIdent {
  StaticTDGIdent(StaticTDGInfoForFunctionTy StaticTDGInfoForFunction)
      : StaticTDGInfoForFunction(StaticTDGInfoForFunction) {}
  StaticTDGInfoForFunctionTy StaticTDGInfoForFunction;

  bool runOnModule(Module &M) {
    if (M.empty())
      return false;

    SmallVector<Function *, 4> Functs;
    for (auto &F : M) {
      // Nothing to do for declarations.
      if (F.isDeclaration() || F.empty())
        continue;

      Functs.push_back(&F);
    }

    bool Changed = false;
    for (auto *F : Functs) {
      StaticTDGInfo SD = StaticTDGInfoForFunction(*F);

      auto FinalTaskLoops = SD.FinalTaskLoops;
      auto NumberOfTasks = SD.NumberOfTasks;
      auto Tasks = SD.Tasks;

      LLVM_DEBUG(if (!FinalTaskLoops.empty() || !Tasks.empty()) {
        dbgs() << "Processing function " << F->getName() << "\n";
      });

      SmallVector<std::pair<Loop *, Value *>, 4> LoopValueMap;
      for (auto loop : FinalTaskLoops) {
        // TODO: Work for loops with multiple predecessors
        BasicBlock *predecessor = loop->getLoopPredecessor();
        IRBuilder<> IRB(predecessor->getTerminator());
        Value *LoopVariable = IRB.CreateAlloca(IRB.getInt32Ty());
        IRB.CreateStore(IRB.getInt32(0), LoopVariable);
        LoopValueMap.push_back(std::make_pair(loop, LoopVariable));

        BasicBlock *FirstBlock = *(loop->block_begin());
        IRB.SetInsertPoint(FirstBlock->getFirstNonPHI());
        Value *LoopVariableValue =
            IRB.CreateLoad(IRB.getInt32Ty(), LoopVariable);
        Value *FirstAdd = IRB.CreateAdd(LoopVariableValue, IRB.getInt32(1));
        IRB.CreateStore(FirstAdd, LoopVariable);

        Changed = true;
      }

      for (auto &task : Tasks) {

        Instruction *TaskAlloc = task.TaskAllocInstruction;
        IRBuilder<> IRB(TaskAlloc);

        Value *PreviousLoop = IRB.getInt32(0);
        for (auto loop : task.loops) {

          Value *LoopVariable = nullptr;

          for (auto SelectedPair : LoopValueMap) {
            if (SelectedPair.first == loop)
              LoopVariable = SelectedPair.second;
          }

          assert(LoopVariable != nullptr &&
                 "Error finding created loop variable");

          Value *LoopVariableValue =
              IRB.CreateLoad(IRB.getInt32Ty(), LoopVariable);
          Value *FirstAdd = IRB.CreateAdd(LoopVariableValue, PreviousLoop);
          Value *FirstMultiply =
              IRB.CreateMul(FirstAdd, IRB.getInt32(task.maxIteration));
          PreviousLoop = FirstMultiply;
        }

        Value *SecondMultiply =
            IRB.CreateMul(PreviousLoop, IRB.getInt32(NumberOfTasks));
        Value *SecondAdd = IRB.CreateAdd(SecondMultiply, IRB.getInt32(task.id));

        CallInst *TaskAllocCall = dyn_cast<CallInst>(TaskAlloc);
        TaskAllocCall->setArgOperand(TaskAllocCall->getNumOperands() - 1,
                                     SecondAdd);

        Changed = true;
      }
    }

    return Changed;
  }
};

struct StaticTDGIdentLegacyPass : public ModulePass {
  /// Pass identification, replacement for typeid
  static char ID;
  StaticTDGIdentLegacyPass() : ModulePass(ID) {
    initializeStaticTDGIdentLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    // See below for the reason of this.
    auto StaticTDGInfoForFunction = [this](Function &F) -> StaticTDGInfo {
      return this->getAnalysis<StaticTDGLegacyPass>(F).getTaskData();
    };
    return StaticTDGIdent(StaticTDGInfoForFunction).runOnModule(M);
  }

  StringRef getPassName() const override {
    return "Static TDG task identifier calculation";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<StaticTDGLegacyPass>();
  }
};

} // namespace

char StaticTDGIdentLegacyPass::ID = 0;

ModulePass *llvm::createStaticTDGIdentLegacyPass() {
  return new StaticTDGIdentLegacyPass();
}

void LLVMStaticTDGPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createStaticTDGIdentLegacyPass());
}

INITIALIZE_PASS_BEGIN(StaticTDGIdentLegacyPass, "static-tdg-id",
                      "Static TDG task identifier calculation", false, false)
INITIALIZE_PASS_DEPENDENCY(StaticTDGLegacyPass)
INITIALIZE_PASS_END(StaticTDGIdentLegacyPass, "static-tdg-id",
                    "Static TDG task identifier calculation", false, false)

// New pass manager.
PreservedAnalyses StaticTDGIdentPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  // This is an adapter that will be used to get the analysis data of a given
  // function. This is done this way so we can reuse the code for the new and
  // legacy pass manager (they used different functions to get this
  // information).
  auto StaticTDGInfoForFunction = [&FAM](Function &F) -> StaticTDGInfo {
    return FAM.getResult<StaticTDGAnalysis>(F);
  };
  if (!StaticTDGIdent(StaticTDGInfoForFunction).runOnModule(M))
    return PreservedAnalyses::all();

  // FIXME: we can be more precise here.
  return PreservedAnalyses::none();
}
