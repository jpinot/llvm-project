//===-- StaticTDG.h - StaticTDG Transforms ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_STATIC_TDG_H
#define LLVM_TRANSFORMS_STATIC_TDG_H

#include <functional>

namespace llvm {

class ModulePass;

//===----------------------------------------------------------------------===//

// Creates identifiers for the Static TDGs
ModulePass *createStaticTDGIdentLegacyPass();

struct StaticTDGIdentPass : public PassInfoMixin<StaticTDGIdentPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // End llvm namespace

#endif
