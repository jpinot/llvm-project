//===-- DynamicVariant.h - DynamicVariant Transforms ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_DYNAMIC_VARIANT_H
#define LLVM_TRANSFORMS_DYNAMIC_VARIANT_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include <functional>

namespace llvm {

class ModulePass;

//===----------------------------------------------------------------------===//

// Creates dynamic variants
ModulePass *createDynamicVariantLegacyPass();

struct DynamicVariantPass : public PassInfoMixin<DynamicVariantPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // End llvm namespace

#endif
