//===- OmpSs.cpp -- Strip parts of Debug Info --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

namespace {

struct StaticTDGIdent : public ModulePass {
  /// Pass identification, replacement for typeid
  static char ID;
  StaticTDGIdent() : ModulePass(ID) {
    initializeStaticTDGIdentPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    SmallVector<Function *, 4> Functs;
    for (auto &F : M) {
      // Nothing to do for declarations.
      if (F.isDeclaration() || F.empty())
        continue;

      Functs.push_back(&F);
    }

    for (auto *F : Functs) {
      auto SD = getAnalysis<StaticTDGPass>(*F).getTaskData();

      auto FinalTaskLoops = SD.FinalTaskLoops;
      auto NumberOfTasks = SD.NumberOfTasks;
      auto Tasks = SD.Tasks;

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
        TaskAllocCall->setArgOperand(TaskAllocCall->getNumArgOperands() - 1,
                                     SecondAdd);
      }
    }
    // dbgs() << "Done! \n";
    return true;
  }

  StringRef getPassName() const override {
    return "Static TDG task identifier calculation";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<StaticTDGPass>();
  }
};

} // namespace

char StaticTDGIdent::ID = 0;

ModulePass *llvm::createStaticTDGIdentPass() { return new StaticTDGIdent(); }

void LLVMStaticTDGPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createStaticTDGIdentPass());
}

INITIALIZE_PASS_BEGIN(StaticTDGIdent, "static-tdg-id",
                      "Static TDG task identifier calculation", false, false)
INITIALIZE_PASS_DEPENDENCY(StaticTDGPass)
INITIALIZE_PASS_END(StaticTDGIdent, "static-tdg-id",
                    "Static TDG task identifier calculation", false, false)
