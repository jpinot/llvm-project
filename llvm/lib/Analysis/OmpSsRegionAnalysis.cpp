//===- OmpSsRegionAnalysis.cpp - OmpSs Region Analysis -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/OmpSsRegionAnalysis.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsOmpSs.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
using namespace llvm;
using namespace autoscope;

#define DEBUG_TYPE "ompss-2-regions"

void analyzeFirstTaskUse(BasicBlock *BB, Value *Var, Instruction *Exit,
                         Instruction *Entry,
                         SmallPtrSetImpl<BasicBlock *> *Already,
                         SmallPtrSetImpl<Function *> *AnalyzedFunctions,
                         ValueAccess &AccessToCheck, AAResults &BAA, AliveValue TypeOfAlive, bool ContinueSubblocks);

// Auxiliar arrays for printing enums
std::string SyncTypeToString[]{"FUNCTION_START", "FUNCTION_END", "TASKWAIT",
                               "TASK"};
std::string VarUseToString[]{"NOT_USED", "WRITTEN", "READED", "UNKNOWN"};
std::string VarDependencyToString[]{"NONE", "IN", "OUT", "INOUT"};
std::string DSAToString[]{
    "PRIVATE", "FIRSTPRIVATE",     "SHARED",     "SHARED_OR_FIRSTPRIVATE",
    "UNDEF",   "RACEFIRSTPRIVATE", "RACEPRIVATE"};

static cl::opt<bool>
DisableChecks("disable-checks",
                  cl::desc("Avoid checking OmpSs-2 directive bundle correctness"),
                  cl::Hidden,
                  cl::init(false));

enum PrintVerbosity {
  PV_Directive,
  PV_Uses,
  PV_UnpackAndConst,
  PV_DsaMissing,
  PV_DsaVLADimsMissing,
  PV_VLADimsCaptureMissing,
  PV_NonPODDSAMissing,
  PV_ReductionInitsCombiners,
  PV_AutoScoping,
  PV_AutoDependencies
};

static cl::opt<PrintVerbosity>
PrintVerboseLevel("print-verbosity",
  cl::desc("Choose verbosity level"),
  cl::Hidden,
  cl::values(
  clEnumValN(PV_Directive, "directive", "Print directive layout only"),
  clEnumValN(PV_Uses, "uses", "Print directive layout with uses"),
  clEnumValN(PV_DsaMissing, "dsa_missing", "Print directive layout with uses without DSA"),
  clEnumValN(PV_DsaVLADimsMissing, "dsa_vla_dims_missing", "Print directive layout with DSAs without VLA info or VLA info without DSAs"),
  clEnumValN(PV_VLADimsCaptureMissing, "vla_dims_capture_missing", "Print directive layout with VLA dimensions without capture"),
  clEnumValN(PV_NonPODDSAMissing, "non_pod_dsa_missing", "Print directive layout with non-pod info without according DSA"),
  clEnumValN(PV_ReductionInitsCombiners, "reduction_inits_combiners", "Print directive layout with reduction init and combiner functions"),
  clEnumValN(PV_AutoScoping, "autoscope", "Print autoscoping results"),
  clEnumValN(PV_AutoDependencies, "autodependencies", "Print autodependencies results"))
  );


// returns if V is in DSAInfo
static bool valueInDSABundles(const DirectiveDSAInfo& DSAInfo,
                              const Value *V) {
  auto SharedIt = find(DSAInfo.Shared, V);
  auto PrivateIt = find(DSAInfo.Private, V);
  auto FirstprivateIt = find(DSAInfo.Firstprivate, V);
  if (SharedIt == DSAInfo.Shared.end()
      && PrivateIt == DSAInfo.Private.end()
      && FirstprivateIt == DSAInfo.Firstprivate.end())
    return false;

  return true;
}

// returns if V is in CapturedInfo
static bool valueInCapturedBundle(const DirectiveCapturedInfo& CapturedInfo,
                                  Value *const V) {
  return CapturedInfo.count(V);
}

/// NOTE: from old OrderedInstructions
static bool localDominates(
    const Instruction *InstA, const Instruction *InstB) {
  assert(InstA->getParent() == InstB->getParent() &&
         "Instructions must be in the same basic block");

  return InstA->comesBefore(InstB);
}

/// Given 2 instructions, check for dominance relation if the instructions are
/// in the same basic block. Otherwise, use dominator tree.
/// NOTE: from old OrderedInstructions
static bool orderedInstructions(
    DominatorTree &DT, const Instruction *InstA, const Instruction *InstB) {
  // Use ordered basic block to do dominance check in case the 2 instructions
  // are in the same basic block.
  if (InstA->getParent() == InstB->getParent())
    return localDominates(InstA, InstB);
  return DT.dominates(InstA->getParent(), InstB->getParent());
}

static DependInfo::DependType getDependTypeFromId(uint64_t Id) {
  switch (Id) {
  case LLVMContext::OB_oss_dep_in:
  case LLVMContext::OB_oss_multidep_range_in:
    return DependInfo::DT_in;
  case LLVMContext::OB_oss_dep_out:
  case LLVMContext::OB_oss_multidep_range_out:
    return DependInfo::DT_out;
  case LLVMContext::OB_oss_dep_inout:
  case LLVMContext::OB_oss_multidep_range_inout:
    return DependInfo::DT_inout;
  case LLVMContext::OB_oss_dep_concurrent:
  case LLVMContext::OB_oss_multidep_range_concurrent:
    return DependInfo::DT_concurrent;
  case LLVMContext::OB_oss_dep_commutative:
  case LLVMContext::OB_oss_multidep_range_commutative:
    return DependInfo::DT_commutative;
  case LLVMContext::OB_oss_dep_reduction:
    return DependInfo::DT_reduction;
  case LLVMContext::OB_oss_dep_weakin:
  case LLVMContext::OB_oss_multidep_range_weakin:
    return DependInfo::DT_weakin;
  case LLVMContext::OB_oss_dep_weakout:
  case LLVMContext::OB_oss_multidep_range_weakout:
    return DependInfo::DT_weakout;
  case LLVMContext::OB_oss_dep_weakinout:
  case LLVMContext::OB_oss_multidep_range_weakinout:
    return DependInfo::DT_weakinout;
  case LLVMContext::OB_oss_dep_weakconcurrent:
  case LLVMContext::OB_oss_multidep_range_weakconcurrent:
    return DependInfo::DT_weakconcurrent;
  case LLVMContext::OB_oss_dep_weakcommutative:
  case LLVMContext::OB_oss_multidep_range_weakcommutative:
    return DependInfo::DT_weakcommutative;
  case LLVMContext::OB_oss_dep_weakreduction:
    return DependInfo::DT_weakreduction;
  }
  llvm_unreachable("unknown depend type id");
}

void DirectiveEnvironment::gatherDirInfo(OperandBundleDef &OB) {
  assert(DirectiveKind == OSSD_unknown && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  ConstantDataArray *DirectiveKindDataArray = cast<ConstantDataArray>(OB.inputs()[0]);
  assert(DirectiveKindDataArray->isCString() && "Directive kind must be a C string");
  DirectiveKindStringRef = DirectiveKindDataArray->getAsCString();

  if (DirectiveKindStringRef == "TASK")
    DirectiveKind = OSSD_task;
  else if (DirectiveKindStringRef == "TASK.FOR")
    DirectiveKind = OSSD_task_for;
  else if (DirectiveKindStringRef == "TASKLOOP")
    DirectiveKind = OSSD_taskloop;
  else if (DirectiveKindStringRef == "TASKLOOP.FOR")
    DirectiveKind = OSSD_taskloop_for;
  else if (DirectiveKindStringRef == "TASKWAIT")
    DirectiveKind = OSSD_taskwait;
  else if (DirectiveKindStringRef == "RELEASE")
    DirectiveKind = OSSD_release;
  else
    llvm_unreachable("Unhandled DirectiveKind string");
}

void DirectiveEnvironment::gatherSharedInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  DSAInfo.Shared.insert(OB.inputs()[0]);
}

void DirectiveEnvironment::gatherPrivateInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  DSAInfo.Private.insert(OB.inputs()[0]);
}

void DirectiveEnvironment::gatherFirstprivateInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  DSAInfo.Firstprivate.insert(OB.inputs()[0]);
}

void DirectiveEnvironment::gatherVLADimsInfo(OperandBundleDef &OB) {
  assert(OB.input_size() > 1 &&
    "VLA dims OperandBundle must have at least a value for the VLA and one dimension");
  ArrayRef<Value *> OBArgs = OB.inputs();
  assert(VLADimsInfo[OBArgs[0]].empty() && "There're VLA dims duplicated OperandBundles");
  VLADimsInfo[OBArgs[0]].insert(&OBArgs[1], OBArgs.end());
}

static void gatherDependInfo(
    ArrayRef<Value *> OBArgs, std::map<Value *, int> &DepSymToIdx,
    DirectiveDependsInfo &DependsInfo, DependInfo &DI, uint64_t Id) {
  DI.DepType = getDependTypeFromId(Id);

  if (DI.isReduction()) {
    DI.RedKind = OBArgs[0];
    // Skip the reduction kind
    OBArgs = OBArgs.drop_front(1);
  }

  // First operand has to be the DSA over the dependency is made
  DI.Base = OBArgs[0];

  ConstantDataArray *DirectiveKindDataArray = cast<ConstantDataArray>(OBArgs[1]);
  assert(DirectiveKindDataArray->isCString() && "Region text must be a C string");
  DI.RegionText = DirectiveKindDataArray->getAsCString();

  DI.ComputeDepFun = cast<Function>(OBArgs[2]);

  // Gather compute_dep function params
  for (size_t i = 3; i < OBArgs.size(); ++i) {
    DI.Args.push_back(OBArgs[i]);
  }

  if (!DepSymToIdx.count(DI.Base)) {
    DepSymToIdx[DI.Base] = DepSymToIdx.size();
    DependsInfo.NumSymbols++;
  }
  DI.SymbolIndex = DepSymToIdx[DI.Base];

}

void DirectiveEnvironment::gatherDependInfo(
    OperandBundleDef &OB, uint64_t Id) {

  assert(OB.input_size() > 2 &&
    "Depend OperandBundle must have at least depend base, function and one argument");
  DependInfo *DI = new DependInfo();

  ::gatherDependInfo(OB.inputs(), DepSymToIdx, DependsInfo, *DI, Id);

  DependsInfo.List.emplace_back(DI);
}

void DirectiveEnvironment::gatherReductionInitInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 2 &&
    "Reduction init/combiner must have a Value matching a DSA and a function pointer Value");
  ArrayRef<Value *> OBArgs = OB.inputs();

  // This assert should not trigger since clang allows an unique reduction per DSA
  assert(!ReductionsInitCombInfo.count(OBArgs[0])
         && "Two or more reductions of the same DSA in the same directive are not allowed");
  ReductionsInitCombInfo[OBArgs[0]].Init = OBArgs[1];

  if (SeenInits.count(OBArgs[1])) {
    ReductionsInitCombInfo[OBArgs[0]].ReductionIndex = SeenInits[OBArgs[1]];
  } else {
    SeenInits[OBArgs[1]] = ReductionIndex;
    ReductionsInitCombInfo[OBArgs[0]].ReductionIndex = ReductionIndex;
    ReductionIndex++;
  }
}

void DirectiveEnvironment::gatherReductionCombInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 2 &&
    "Reduction init/combiner must have a Value matching a DSA and a function pointer Value");
  ArrayRef<Value *> OBArgs = OB.inputs();

  ReductionsInitCombInfo[OBArgs[0]].Comb = OBArgs[1];
}

void DirectiveEnvironment::gatherFinalInfo(OperandBundleDef &OB) {
  assert(!Final && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  Final = OB.inputs()[0];
}

void DirectiveEnvironment::gatherIfInfo(OperandBundleDef &OB) {
  assert(!If && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  If = OB.inputs()[0];
}

void DirectiveEnvironment::gatherCostInfo(OperandBundleDef &OB) {
  assert(OB.input_size() > 0 &&
    "Cost OperandBundle must have at least function");
  ArrayRef<Value *> OBArgs = OB.inputs();
  CostInfo.Fun = cast<Function>(OBArgs[0]);
  for (size_t i = 1; i < OBArgs.size(); ++i)
    CostInfo.Args.push_back(OBArgs[i]);
}

void DirectiveEnvironment::gatherPriorityInfo(OperandBundleDef &OB) {
  assert(OB.input_size() > 0 &&
    "Priority OperandBundle must have at least function");
  ArrayRef<Value *> OBArgs = OB.inputs();
  PriorityInfo.Fun = cast<Function>(OBArgs[0]);
  for (size_t i = 1; i < OBArgs.size(); ++i)
    PriorityInfo.Args.push_back(OBArgs[i]);
}

void DirectiveEnvironment::gatherLabelInfo(OperandBundleDef &OB) {
  assert(!Label && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  Label = OB.inputs()[0];
}

void DirectiveEnvironment::gatherOnreadyInfo(OperandBundleDef &OB) {
  assert(OB.input_size() > 0 &&
    "Onready OperandBundle must have at least function");
  ArrayRef<Value *> OBArgs = OB.inputs();
  OnreadyInfo.Fun = cast<Function>(OBArgs[0]);
  for (size_t i = 1; i < OBArgs.size(); ++i)
    OnreadyInfo.Args.push_back(OBArgs[i]);
}

void DirectiveEnvironment::gatherWaitInfo(OperandBundleDef &OB) {
  assert(!Wait && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  Wait = OB.inputs()[0];
}

void DirectiveEnvironment::gatherCapturedInfo(OperandBundleDef &OB) {
  assert(CapturedInfo.empty() && "Only allowed one OperandBundle with this Id");
  CapturedInfo.insert(OB.input_begin(), OB.input_end());
}

void DirectiveEnvironment::gatherNonPODInitInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 2 &&
    "Non-POD info must have a Value matching a DSA and a function pointer Value");
  ArrayRef<Value *> OBArgs = OB.inputs();
  NonPODsInfo.Inits[OBArgs[0]] = OBArgs[1];
}

void DirectiveEnvironment::gatherNonPODDeinitInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 2 &&
    "Non-POD info must have a Value matching a DSA and a function pointer Value");
  ArrayRef<Value *> OBArgs = OB.inputs();
  NonPODsInfo.Deinits[OBArgs[0]] = OBArgs[1];
}

void DirectiveEnvironment::gatherNonPODCopyInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 2 && "Non-POD info must have a Value matching a DSA and a function pointer Value");
  ArrayRef<Value *> OBArgs = OB.inputs();
  NonPODsInfo.Copies[OBArgs[0]] = OBArgs[1];
}

void DirectiveEnvironment::gatherLoopTypeInfo(OperandBundleDef &OB) {
  assert(LoopInfo.LoopType.empty() && "Only allowed one OperandBundle with this Id");
  assert(LoopInfo.IndVarSigned.empty() && "Only allowed one OperandBundle with this Id");
  assert(LoopInfo.LBoundSigned.empty() && "Only allowed one OperandBundle with this Id");
  assert(LoopInfo.UBoundSigned.empty() && "Only allowed one OperandBundle with this Id");
  assert(LoopInfo.StepSigned.empty() && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size()%5 == 0 && "Expected loop type and indvar, lb, ub, step signedness");

  for (size_t i = 0; i < OB.input_size()/5; i++) {
    LoopInfo.LoopType.push_back(cast<ConstantInt>(OB.inputs()[i*5 + 0])->getSExtValue());
    LoopInfo.IndVarSigned.push_back(cast<ConstantInt>(OB.inputs()[i*5 + 1])->getSExtValue());
    LoopInfo.LBoundSigned.push_back(cast<ConstantInt>(OB.inputs()[i*5 + 2])->getSExtValue());
    LoopInfo.UBoundSigned.push_back(cast<ConstantInt>(OB.inputs()[i*5 + 3])->getSExtValue());
    LoopInfo.StepSigned.push_back(cast<ConstantInt>(OB.inputs()[i*5 + 4])->getSExtValue());
  }
}

void DirectiveEnvironment::gatherLoopIndVarInfo(OperandBundleDef &OB) {
  assert(LoopInfo.IndVar.empty() && "Only allowed one OperandBundle with this Id");
  for (size_t i = 0; i < OB.input_size(); i++)
    LoopInfo.IndVar.push_back(OB.inputs()[i]);
}

void DirectiveEnvironment::gatherLoopLowerBoundInfo(OperandBundleDef &OB) {
  assert(LoopInfo.LBound.empty() && "Only allowed one OperandBundle with this Id");
  for (size_t i = 0; i < OB.input_size(); i++) {
    if (auto *F = dyn_cast<Function>(OB.inputs()[i])) {
      LoopInfo.LBound.emplace_back();
      LoopInfo.LBound.back().Fun = F;
    } else
      LoopInfo.LBound.back().Args.push_back(OB.inputs()[i]);
  }
}

void DirectiveEnvironment::gatherLoopUpperBoundInfo(OperandBundleDef &OB) {
  assert(LoopInfo.UBound.empty() && "Only allowed one OperandBundle with this Id");
  for (size_t i = 0; i < OB.input_size(); i++) {
    if (auto *F = dyn_cast<Function>(OB.inputs()[i])) {
      LoopInfo.UBound.emplace_back();
      LoopInfo.UBound.back().Fun = F;
    } else
      LoopInfo.UBound.back().Args.push_back(OB.inputs()[i]);
  }
}

void DirectiveEnvironment::gatherLoopStepInfo(OperandBundleDef &OB) {
  assert(LoopInfo.Step.empty() && "Only allowed one OperandBundle with this Id");
  for (size_t i = 0; i < OB.input_size(); i++) {
    if (auto *F = dyn_cast<Function>(OB.inputs()[i])) {
      LoopInfo.Step.emplace_back();
      LoopInfo.Step.back().Fun = F;
    } else
      LoopInfo.Step.back().Args.push_back(OB.inputs()[i]);
  }
}

void DirectiveEnvironment::gatherLoopChunksizeInfo(OperandBundleDef &OB) {
  assert(!LoopInfo.Chunksize && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  LoopInfo.Chunksize = OB.inputs()[0];
}

void DirectiveEnvironment::gatherLoopGrainsizeInfo(OperandBundleDef &OB) {
  assert(!LoopInfo.Grainsize && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  LoopInfo.Grainsize = OB.inputs()[0];
}

void DirectiveEnvironment::gatherMultiDependInfo(
    OperandBundleDef &OB, uint64_t Id) {
  // TODO: add asserts

  MultiDependInfo *MDI = new MultiDependInfo();
  MDI->DepType = getDependTypeFromId(Id);

  ArrayRef<Value *> OBArgs = OB.inputs();

  size_t i;
  size_t ComputeFnCnt = 0;
  // 1. Gather iterators from begin to compute multidep function
  // 2. Gather compute multidep function args from 1. to compute dep function previous element
  //    which is the dep base
  for (i = 0; i < OBArgs.size(); ++i) {
    if (auto *ComputeFn = dyn_cast<Function>(OBArgs[i])) {
      if (ComputeFnCnt == 0) // ComputeMultiDepFun
        MDI->ComputeMultiDepFun = ComputeFn;
      else // Seen ComputeDepFun
        break;
      ++ComputeFnCnt;
      continue;
    }
    if (ComputeFnCnt == 0)
      MDI->Iters.push_back(OBArgs[i]);
    else if (ComputeFnCnt == 1)
      MDI->Args.push_back(OBArgs[i]);
  }
  // TODO: this is used because we add dep base and region text too
  // which is wrong...
  MDI->Args.pop_back();
  MDI->Args.pop_back();

  ::gatherDependInfo(OBArgs.drop_front(i - 2), DepSymToIdx, DependsInfo, *MDI, Id);

  DependsInfo.List.emplace_back(MDI);
}

void DirectiveEnvironment::gatherDeclSource(OperandBundleDef &OB) {
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  ConstantDataArray *DeclSourceDataArray = cast<ConstantDataArray>(OB.inputs()[0]);
  assert(DeclSourceDataArray->isCString() && "Region text must be a C string");
  DeclSourceStringRef = DeclSourceDataArray->getAsCString();
}

void DirectiveEnvironment::verifyVLADimsInfo() {
  for (const auto &VLAWithDimsMap : VLADimsInfo) {
    if (!valueInDSABundles(DSAInfo, VLAWithDimsMap.first))
      llvm_unreachable("VLA dims OperandBundle must have an associated DSA");
    // VLA Dims that are not Captured is an error
    for (auto *V : VLAWithDimsMap.second) {
      if (!valueInCapturedBundle(CapturedInfo, V))
        llvm_unreachable("VLA dimension has not been captured");
    }
  }
}

void DirectiveEnvironment::verifyDependInfo() {
  for (auto &DI : DependsInfo.List) {
    if (!valueInDSABundles(DSAInfo, DI->Base))
      llvm_unreachable("Dependency has no associated DSA");
    for (auto *V : DI->Args) {
      if (!valueInDSABundles(DSAInfo, V)
          && !valueInCapturedBundle(CapturedInfo, V))
        llvm_unreachable("Dependency has no associated DSA or capture");
    }
  }
}

void DirectiveEnvironment::verifyReductionInitCombInfo() {
  for (const auto &RedInitCombMap : ReductionsInitCombInfo) {
    if (!valueInDSABundles(DSAInfo, RedInitCombMap.first))
      llvm_unreachable(
        "Reduction init/combiner must have a Value matching a DSA and a function pointer Value");
    if (!RedInitCombMap.second.Init)
      llvm_unreachable("Missing reduction initializer");
    if (!RedInitCombMap.second.Comb)
      llvm_unreachable("Missing reduction combiner");
  }
}

void DirectiveEnvironment::verifyCostInfo() {
  for (auto *V : CostInfo.Args) {
    if (!valueInDSABundles(DSAInfo, V)
        && !valueInCapturedBundle(CapturedInfo, V))
      llvm_unreachable("Cost function argument has no associated DSA or capture");
  }
}

void DirectiveEnvironment::verifyPriorityInfo() {
  for (auto *V : PriorityInfo.Args) {
    if (!valueInDSABundles(DSAInfo, V)
        && !valueInCapturedBundle(CapturedInfo, V))
      llvm_unreachable("Priority function argument has no associated DSA or capture");
  }
}

void DirectiveEnvironment::verifyOnreadyInfo() {
  for (auto *V : OnreadyInfo.Args) {
    if (!valueInDSABundles(DSAInfo, V)
        && !valueInCapturedBundle(CapturedInfo, V))
      llvm_unreachable("Onready function argument has no associated DSA or capture");
  }
}

void DirectiveEnvironment::verifyNonPODInfo() {
  for (const auto &InitMap : NonPODsInfo.Inits) {
    // INIT may only be in private clauses
    auto It = find(DSAInfo.Private, InitMap.first);
    if (It == DSAInfo.Private.end())
      llvm_unreachable("Non-POD INIT OperandBundle must have a PRIVATE DSA");
  }
  for (const auto &DeinitMap : NonPODsInfo.Deinits) {
      // DEINIT may only be in firstprivate clauses
      auto PrivateIt = find(DSAInfo.Private, DeinitMap.first);
      auto FirstprivateIt = find(DSAInfo.Firstprivate, DeinitMap.first);
      if (FirstprivateIt == DSAInfo.Firstprivate.end()
          && PrivateIt == DSAInfo.Private.end())
        llvm_unreachable("Non-POD DEINIT OperandBundle must have a PRIVATE or FIRSTPRIVATE DSA");
  }
  for (const auto &CopyMap : NonPODsInfo.Copies) {
    // COPY may only be in firstprivate clauses
    auto It = find(DSAInfo.Firstprivate, CopyMap.first);
    if (It == DSAInfo.Firstprivate.end())
      llvm_unreachable("Non-POD COPY OperandBundle must have a FIRSTPRIVATE DSA");
  }
}

void DirectiveEnvironment::verifyLoopInfo() {
  if (isOmpSsLoopDirective() && LoopInfo.empty())
    llvm_unreachable("LoopInfo is missing some information");
  for (size_t i = 0; i < LoopInfo.IndVar.size(); ++i) {
    if (!valueInDSABundles(DSAInfo, LoopInfo.IndVar[i]))
      llvm_unreachable("Loop induction variable has no associated DSA");
    for (size_t j = 0; j < LoopInfo.LBound[i].Args.size(); ++j) {
      if (!valueInDSABundles(DSAInfo, LoopInfo.LBound[i].Args[j])
          && !valueInCapturedBundle(CapturedInfo, LoopInfo.LBound[i].Args[j]))
        llvm_unreachable("Loop lbound argument value has no associated DSA or capture");
    }
    for (size_t j = 0; j < LoopInfo.UBound[i].Args.size(); ++j) {
      if (!valueInDSABundles(DSAInfo, LoopInfo.UBound[i].Args[j])
          && !valueInCapturedBundle(CapturedInfo, LoopInfo.UBound[i].Args[j]))
        llvm_unreachable("Loop ubound argument value has no associated DSA or capture");
    }
    for (size_t j = 0; j < LoopInfo.Step[i].Args.size(); ++j) {
      if (!valueInDSABundles(DSAInfo, LoopInfo.Step[i].Args[j])
          && !valueInCapturedBundle(CapturedInfo, LoopInfo.Step[i].Args[j]))
        llvm_unreachable("Loop step argument value has no associated DSA or capture");
    }
  }
}

void DirectiveEnvironment::verifyMultiDependInfo() {
  for (auto &DI : DependsInfo.List)
    if (auto *MDI = dyn_cast<MultiDependInfo>(DI.get())) {
      for (auto *V : MDI->Iters)
        if (!valueInDSABundles(DSAInfo, V)
            && !valueInCapturedBundle(CapturedInfo, V))
          llvm_unreachable("Multidependency value has no associated DSA or capture");
      for (auto *V : MDI->Args)
        if (!valueInDSABundles(DSAInfo, V)
            && !valueInCapturedBundle(CapturedInfo, V))
          llvm_unreachable("Multidependency value has no associated DSA or capture");
    }
}

void DirectiveEnvironment::verify() {
  verifyVLADimsInfo();

  // release directive does not need data-sharing checks
  if (DirectiveKind != OSSD_release)
    verifyDependInfo();

  verifyReductionInitCombInfo();
  verifyCostInfo();
  verifyPriorityInfo();
  verifyOnreadyInfo();
  verifyNonPODInfo();
  verifyLoopInfo();
  verifyMultiDependInfo();
}

DirectiveEnvironment::DirectiveEnvironment(const Instruction *I) {
  const IntrinsicInst *II = cast<IntrinsicInst>(I);
  for (unsigned i = 0, e = II->getNumOperandBundles(); i != e; ++i) {
    OperandBundleUse OBUse = II->getOperandBundleAt(i);
    OperandBundleDef OBDef(OBUse);
    uint64_t Id = OBUse.getTagID();
    switch (Id) {
    case LLVMContext::OB_oss_dir:
      gatherDirInfo(OBDef);
      break;
    case LLVMContext::OB_oss_shared:
      gatherSharedInfo(OBDef);
      break;
    case LLVMContext::OB_oss_private:
      gatherPrivateInfo(OBDef);
      break;
    case LLVMContext::OB_oss_firstprivate:
      gatherFirstprivateInfo(OBDef);
      break;
    case LLVMContext::OB_oss_vla_dims:
      gatherVLADimsInfo(OBDef);
      break;
    case LLVMContext::OB_oss_dep_in:
    case LLVMContext::OB_oss_dep_out:
    case LLVMContext::OB_oss_dep_inout:
    case LLVMContext::OB_oss_dep_concurrent:
    case LLVMContext::OB_oss_dep_commutative:
    case LLVMContext::OB_oss_dep_weakin:
    case LLVMContext::OB_oss_dep_weakout:
    case LLVMContext::OB_oss_dep_weakinout:
    case LLVMContext::OB_oss_dep_weakconcurrent:
    case LLVMContext::OB_oss_dep_weakcommutative:
    case LLVMContext::OB_oss_dep_reduction:
    case LLVMContext::OB_oss_dep_weakreduction:
      gatherDependInfo(OBDef, Id);
      break;
    case LLVMContext::OB_oss_reduction_init:
      gatherReductionInitInfo(OBDef);
      break;
    case LLVMContext::OB_oss_reduction_comb:
      gatherReductionCombInfo(OBDef);
      break;
    case LLVMContext::OB_oss_final:
      gatherFinalInfo(OBDef);
      break;
    case LLVMContext::OB_oss_if:
      gatherIfInfo(OBDef);
      break;
    case LLVMContext::OB_oss_cost:
      gatherCostInfo(OBDef);
      break;
    case LLVMContext::OB_oss_priority:
      gatherPriorityInfo(OBDef);
      break;
    case LLVMContext::OB_oss_label:
      gatherLabelInfo(OBDef);
      break;
    case LLVMContext::OB_oss_onready:
      gatherOnreadyInfo(OBDef);
      break;
    case LLVMContext::OB_oss_wait:
      gatherWaitInfo(OBDef);
      break;
    case LLVMContext::OB_oss_captured:
      gatherCapturedInfo(OBDef);
      break;
    case LLVMContext::OB_oss_init:
      gatherNonPODInitInfo(OBDef);
      break;
    case LLVMContext::OB_oss_deinit:
      gatherNonPODDeinitInfo(OBDef);
      break;
    case LLVMContext::OB_oss_copy:
      gatherNonPODCopyInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_type:
      gatherLoopTypeInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_ind_var:
      gatherLoopIndVarInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_lower_bound:
      gatherLoopLowerBoundInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_upper_bound:
      gatherLoopUpperBoundInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_step:
      gatherLoopStepInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_chunksize:
      gatherLoopChunksizeInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_grainsize:
      gatherLoopGrainsizeInfo(OBDef);
      break;
    case LLVMContext::OB_oss_multidep_range_in:
    case LLVMContext::OB_oss_multidep_range_out:
    case LLVMContext::OB_oss_multidep_range_inout:
    case LLVMContext::OB_oss_multidep_range_concurrent:
    case LLVMContext::OB_oss_multidep_range_commutative:
    case LLVMContext::OB_oss_multidep_range_weakin:
    case LLVMContext::OB_oss_multidep_range_weakout:
    case LLVMContext::OB_oss_multidep_range_weakinout:
    case LLVMContext::OB_oss_multidep_range_weakconcurrent:
    case LLVMContext::OB_oss_multidep_range_weakcommutative:
      gatherMultiDependInfo(OBDef, Id);
      break;
    case LLVMContext::OB_oss_decl_source:
      gatherDeclSource(OBDef);
      break;
    default:
      llvm_unreachable("unknown ompss-2 bundle id");
    }
  }
}

void OmpSsRegionAnalysis::print_verbose(
    Instruction *Cur, int Depth, int PrintSpaceMultiplier) const {
  if (Cur) {
    const DirectiveAnalysisInfo &AnalysisInfo = DEntryToDAnalysisInfo.lookup(Cur);
    const DirectiveInfo *Info = DEntryToDInfo.find(Cur)->second.get();
    const DirectiveEnvironment &DirEnv = Info->DirEnv;
    dbgs() << std::string(Depth*PrintSpaceMultiplier, ' ') << "[" << Depth << "] ";
    dbgs() << DirEnv.getDirectiveNameAsStr();
    dbgs() << " ";
    Cur->printAsOperand(dbgs(), false);

    std::string SpaceMultiplierStr = std::string((Depth + 1) * PrintSpaceMultiplier, ' ');
    if (PrintVerboseLevel == PV_Uses) {
      for (auto *V : AnalysisInfo.UsesBeforeEntry) {
        dbgs() << "\n";
        dbgs() << SpaceMultiplierStr
               << "[Before] ";
        V->printAsOperand(dbgs(), false);
      }
      for (auto *V : AnalysisInfo.UsesAfterExit) {
        dbgs() << "\n";
        dbgs() << SpaceMultiplierStr
               << "[After] ";
        V->printAsOperand(dbgs(), false);
      }
    }
    else if (PrintVerboseLevel == PV_DsaMissing) {
      for (auto *V : AnalysisInfo.UsesBeforeEntry) {
        if (!valueInDSABundles(DirEnv.DSAInfo, V)) {
          dbgs() << "\n";
          dbgs() << SpaceMultiplierStr;
          V->printAsOperand(dbgs(), false);
        }
      }
    }
    else if (PrintVerboseLevel == PV_DsaVLADimsMissing) {
      // Count VLAs and DSAs, Well-formed VLA must have a DSA and dimensions.
      // That is, it must have a frequency of 2
      std::map<const Value *, size_t> DSAVLADimsFreqMap;
      for (Value *V : DirEnv.DSAInfo.Shared) DSAVLADimsFreqMap[V]++;
      for (Value *V : DirEnv.DSAInfo.Private) DSAVLADimsFreqMap[V]++;
      for (Value *V : DirEnv.DSAInfo.Firstprivate) DSAVLADimsFreqMap[V]++;

      for (const auto &VLAWithDimsMap : DirEnv.VLADimsInfo) {
        DSAVLADimsFreqMap[VLAWithDimsMap.first]++;
      }
      for (const auto &Pair : DSAVLADimsFreqMap) {
        // It's expected to have only two VLA bundles, the DSA and dimensions
        if (Pair.second != 2) {
          dbgs() << "\n";
          dbgs() << SpaceMultiplierStr;
          Pair.first->printAsOperand(dbgs(), false);
        }
      }
    }
    else if (PrintVerboseLevel == PV_VLADimsCaptureMissing) {
      for (const auto &VLAWithDimsMap : DirEnv.VLADimsInfo) {
        for (auto *V : VLAWithDimsMap.second) {
          if (!valueInCapturedBundle(DirEnv.CapturedInfo, V)) {
            dbgs() << "\n";
            dbgs() << SpaceMultiplierStr;
            V->printAsOperand(dbgs(), false);
          }
        }
      }
    }
    else if (PrintVerboseLevel == PV_NonPODDSAMissing) {
      for (const auto &InitsPair : DirEnv.NonPODsInfo.Inits) {
        auto It = find(DirEnv.DSAInfo.Private, InitsPair.first);
        if (It == DirEnv.DSAInfo.Private.end()) {
          dbgs() << "\n";
          dbgs() << SpaceMultiplierStr
                 << "[Init] ";
          InitsPair.first->printAsOperand(dbgs(), false);
        }
      }
      for (const auto &CopiesPair : DirEnv.NonPODsInfo.Copies) {
        auto It = find(DirEnv.DSAInfo.Firstprivate, CopiesPair.first);
        if (It == DirEnv.DSAInfo.Firstprivate.end()) {
          dbgs() << "\n";
          dbgs() << SpaceMultiplierStr
                 << "[Copy] ";
          CopiesPair.first->printAsOperand(dbgs(), false);
        }
      }
      for (const auto &DeinitsPair : DirEnv.NonPODsInfo.Deinits) {
        auto PrivateIt = find(DirEnv.DSAInfo.Private, DeinitsPair.first);
        auto FirstprivateIt = find(DirEnv.DSAInfo.Firstprivate, DeinitsPair.first);
        if (FirstprivateIt == DirEnv.DSAInfo.Firstprivate.end()
            && PrivateIt == DirEnv.DSAInfo.Private.end()) {
          dbgs() << "\n";
          dbgs() << SpaceMultiplierStr
                 << "[Deinit] ";
          DeinitsPair.first->printAsOperand(dbgs(), false);
        }
      }
    }
    else if (PrintVerboseLevel == PV_ReductionInitsCombiners) {
      for (const auto &RedInfo : DirEnv.ReductionsInitCombInfo) {
        dbgs() << "\n";
        dbgs() << SpaceMultiplierStr;
        RedInfo.first->printAsOperand(dbgs(), false);
        dbgs() << " ";
        RedInfo.second.Init->printAsOperand(dbgs(), false);
        dbgs() << " ";
        RedInfo.second.Comb->printAsOperand(dbgs(), false);
      }
    }
    dbgs() << "\n";
  }
  for (auto *II : DirectivesTree.lookup(Cur)) {
    print_verbose(II, Depth + 1, PrintSpaceMultiplier);
  }
}

DirectiveFunctionInfo& OmpSsRegionAnalysis::getFuncInfo() { return DirectiveFuncInfo; }

void OmpSsRegionAnalysis::print(raw_ostream &OS) const {
  print_verbose(nullptr, -1, PrintSpaceMultiplier);
}

// child directives will be placed before its parent directives
void OmpSsRegionAnalysis::convertDirectivesTreeToVectorImpl(
  Instruction *Cur, SmallVectorImpl<Instruction *> &Stack) {

  if (Cur)
    if (auto *II = dyn_cast<IntrinsicInst>(Cur))
      if (II->getIntrinsicID() == Intrinsic::directive_region_entry)
        Stack.push_back(Cur);

  // TODO: Why using operator[] does weird things?
  // for (auto II : DirectivesTree[Cur]) {
  for (auto II : DirectivesTree.lookup(Cur)) {
    convertDirectivesTreeToVectorImpl(II, Stack);
  }
  if (Cur) {
    DirectiveInfo *DI = DEntryToDInfo[Cur].get();

    if (auto *II = dyn_cast<IntrinsicInst>(Cur)) {
      if (II->getIntrinsicID() == Intrinsic::directive_region_entry) {
        Stack.pop_back();

        for (Instruction *I : Stack) {
          // Annotate the current directive as inner of all directives in stack
          DirectiveInfo *DIStack = DEntryToDInfo[I].get();
          DIStack->InnerDirectiveInfos.push_back(DI);
        }
      }
    }
    DirectiveFuncInfo.PostOrder.push_back(DI);
  }
}


void printTaskConcurrentBlocks(DirectiveInfo &Task) {
  if (PrintVerboseLevel == PV_AutoScoping ||
      PrintVerboseLevel == PV_AutoDependencies){
    dbgs() << "\n -------- TASK Line: " << Task.Entry->getDebugLoc().getLine()
           << " ----------- \n \n";
      }
  else
    LLVM_DEBUG(dbgs() << "\n -------- TASK Line: "
                      << Task.Entry->getDebugLoc().getLine()
                      << " ----------- \n \n");

  for (auto I = Task.PreSyncs.begin(), IE = Task.PreSyncs.end(); I != IE; ++I) {
    SyncInfo *sync = *I;
    LLVM_DEBUG(dbgs() << "Pre sync found: " << SyncTypeToString[sync->Type]
                      << " " << *(sync->Sync) << " \n");
  }
  for (auto I = Task.PostSyncs.begin(), IE = Task.PostSyncs.end(); I != IE;
       ++I) {
    SyncInfo *sync = *I;
    LLVM_DEBUG(dbgs() << "Post sync found: " << SyncTypeToString[sync->Type]
                      << " " << *(sync->Sync) << " \n");
  }

  LLVM_DEBUG(dbgs() << "\n");

  for (auto &concurrent : Task.ConcurrentBlocks) {
    LLVM_DEBUG(dbgs() << "----Concurrent block found---- \n "
                      << "-> Function: "
                      << concurrent.Entry->getParent()->getParent()->getName()
                      << "  " << concurrent.Entry->getParent()->getName()
                      << "<-"
                      << "\n"
                      << " Entry -> "
                      << " " << *concurrent.Entry << " \n");

    LLVM_DEBUG(dbgs() << concurrent.Exit->getName() << " Exit-> "
                      << *concurrent.Exit << " \n");
  }
}

// child directives will be placed before its parent directives
void OmpSsRegionAnalysis::convertDirectivesTreeToVector() {
  SmallVector<Instruction *, 4> Stack;
  convertDirectivesTreeToVectorImpl(nullptr, Stack);
}

void generatePathsUtil(BasicBlock *SourceParent, BasicBlock *DestParent,
                       std::vector<BasicBlock *> &Visited,
                       std::vector<BasicBlock *> &Paths,
                       std::vector<std::vector<BasicBlock *>> &FinalPaths,
                       bool ReturnRedundantPath) {
  bool ContinuePath = true;
  Paths.push_back(SourceParent);

  if ((SourceParent == DestParent && Paths.size() > 1)) {
    FinalPaths.push_back(Paths);
    ContinuePath = false;
  } else if (ReturnRedundantPath) {
    for (auto &Node : Visited) {
      if (Node == SourceParent) {
        FinalPaths.push_back(Paths);
        ContinuePath = false;
        break;
      }
    }
  }

  if (Paths.size() > 1)
    Visited.push_back(SourceParent);
  if (ContinuePath) {

    for (succ_iterator pit = succ_begin(SourceParent),
                       pet = succ_end(SourceParent);
         pit != pet; ++pit) {
      bool IsVisited = false;
      for (auto &Node : Visited) {
        if (Node == *pit) {
          IsVisited = true;
          break;
        }
      }
      if (!IsVisited || ReturnRedundantPath)
        generatePathsUtil(*pit, DestParent, Visited, Paths, FinalPaths,
                          ReturnRedundantPath);
    }
  }

  Visited.pop_back();
  Paths.pop_back();
}

void getPaths(Instruction *Source, Instruction *Dest,
              std::vector<std::vector<BasicBlock *>> &FinalPaths,
              DominatorTree &DT, bool ReturnRedundantPath) {

  std::vector<BasicBlock *> Paths;

  if (Source->getParent() == Dest->getParent() && Source != Dest)
    if (orderedInstructions(DT, Source, Dest)) {
      Paths.push_back(Source->getParent());
      FinalPaths.push_back(Paths);
      return;
    }

  std::vector<BasicBlock *> Visited;

  generatePathsUtil(Source->getParent(), Dest->getParent(), Visited, Paths,
                    FinalPaths, ReturnRedundantPath);
}

// TODO mejorar esto con MemLoc
bool existsTaskDependency(DirectiveInfo &TaskA, DirectiveInfo &TaskB) {
  //Not taking into acount already dependencies
  return false;

  SmallVector<DependInfo, 4> Ins;
  SmallVector<DependInfo, 4> Outs;
  for (auto &Dep : TaskA.DirEnv.DependsInfo.List){
    if (Dep->DepType == DependInfo::DT_in || Dep->DepType == DependInfo::DT_inout)
      Ins.push_back(*Dep);
  }

  for (auto &Dep : TaskB.DirEnv.DependsInfo.List){
    if (Dep->DepType == DependInfo::DT_out || Dep->DepType == DependInfo::DT_inout)
      Outs.push_back(*Dep);
  }

  if (Ins.size() == 0)
    return false;

  for (DependInfo In : Ins) {
    bool exists = false;
    for (DependInfo Out : Outs) {
      if (In.Base == Out.Base) {
        exists = true;
        break;
      }
    }
    if (!exists)
      return false;
  }
  return true;
}


void setPreSync(DirectiveInfo &Task, SmallVectorImpl<SyncInfo> &SyncPoints,
                DominatorTree &DT,
                SmallVectorImpl<DirectiveInfo *> &PostOrder) {

  // Assign all reacheable sync points
  SmallVector<SyncInfo *, 4> PossibleSync;
  for (auto I = SyncPoints.rbegin(), IE = SyncPoints.rend(); I != IE; ++I) {
    SyncInfo *SyncPoint = &*I;
    if (isPotentiallyReachable(SyncPoint->Sync, Task.Entry)) {
      bool visible = true; // Discard invisible taskwaits
      for (auto &SelectedTask : PostOrder) {
        if (orderedInstructions(DT, SelectedTask->Entry, SyncPoint->Sync) &&
            !orderedInstructions(DT,SelectedTask->Exit, SyncPoint->Sync) &&
            !(orderedInstructions(DT,SelectedTask->Entry, Task.Entry) &&
              !orderedInstructions(DT,SelectedTask->Exit, Task.Entry))) {
          visible = false;
          break;
        }
      }
      if (visible)
        PossibleSync.push_back(SyncPoint);
    }
  }

  SmallVector<SyncInfo *, 4> Redundant;
  BasicBlock *TaskParent = Task.Entry->getParent();

  // Eliminate redundand syncs inside same block
  for (unsigned long i = 0; i < PossibleSync.size(); i++) {
    Instruction *First = PossibleSync[i]->Sync;
    BasicBlock *FirstParent = First->getParent();
    for (unsigned long j = i + 1; j < PossibleSync.size(); j++) {
      Instruction *Second = PossibleSync[j]->Sync;
      BasicBlock *SecondParent = Second->getParent();
      if ((FirstParent == SecondParent) && (TaskParent == FirstParent)) {
        if (orderedInstructions(DT,First, Second) && orderedInstructions(DT,First, Task.Entry) &&
            orderedInstructions(DT,Task.Entry, Second)) {
          Redundant.push_back(PossibleSync[i]);
        } else
          Redundant.push_back(PossibleSync[j]);

      } else if (FirstParent == SecondParent) {
        if (orderedInstructions(DT,First, Second))
          Redundant.push_back(PossibleSync[i]);
        else
          Redundant.push_back(PossibleSync[j]);
      }
    }
  }

  SmallVector<SyncInfo *, 4> SyncsForPathing;

  // Remove redundant syncs
  for (auto &Possible : PossibleSync) {
    bool Exists = false;
    for (auto &Red : Redundant) {
      if (Possible == Red) {
        Exists = true;
        break;
      }
    }
    if (!Exists)
      SyncsForPathing.push_back(Possible);
  }
  Redundant.clear();

  // Calculate paths and find more redundant sync points
  for (auto &SyncPoint : SyncsForPathing) {
    std::vector<std::vector<BasicBlock *>> FinalPaths;
    getPaths(SyncPoint->Sync, Task.Entry, FinalPaths, DT, false);
    int InvalidPaths = 0;
    for (auto &Node : FinalPaths) {
      bool Invalid = false;
      for (auto &BB : Node) {
        for (auto &PointFound : SyncsForPathing) {
          if (BB == PointFound->Sync->getParent() && PointFound != SyncPoint) {
            if (BB != Task.Entry->getParent() ||
                (BB == Task.Entry->getParent() &&
                 orderedInstructions(DT,PointFound->Sync, Task.Entry))) {
              Invalid = true;
              InvalidPaths++;
              break;
            }
          }
        }
        if (Invalid)
          break;
      }
    }
    if (InvalidPaths == (int)FinalPaths.size()) {
      Redundant.push_back(SyncPoint);
    }
  }

  // Remove redundant syncs again
  for (auto &Possible : SyncsForPathing) {
    bool Exists = false;
    for (auto &Red : Redundant) {
      if (Possible == Red) {
        Exists = true;
        break;
      }
    }
    if (!Exists)
      Task.PreSyncs.push_back(Possible);
  }


}


void setPostSync(DirectiveInfo &Task, SmallVectorImpl<SyncInfo> &SyncPoints,
                 DominatorTree &DT,
                 SmallVectorImpl<DirectiveInfo *> &PostOrder) {

  // Assign all reacheable sync points
  SmallVector<SyncInfo *, 4> PossibleSync;
  for (auto I = SyncPoints.rbegin(), IE = SyncPoints.rend(); I != IE; ++I) {
    SyncInfo *SyncPoint = &*I;
    if (isPotentiallyReachable(Task.Exit, SyncPoint->Sync)) {
      bool visible = true; // Discard invisible taskwaits
      for (auto &SelectedTask : PostOrder) {
        if (orderedInstructions(DT, SelectedTask->Entry, SyncPoint->Sync) &&
            !orderedInstructions(DT, SelectedTask->Exit, SyncPoint->Sync) &&
            !(orderedInstructions(DT, SelectedTask->Entry, Task.Entry) &&
              !orderedInstructions(DT, SelectedTask->Exit, Task.Entry))) {
          visible = false;
          break;
        }
      }
      if (visible)
        PossibleSync.push_back(SyncPoint);
    }
  }

  SmallVector<SyncInfo *, 4> Redundant;
  BasicBlock *TaskParent = Task.Entry->getParent();

  // Eliminate redundand syncs inside same block
  for (unsigned long i = 0; i < PossibleSync.size(); i++) {
    Instruction *First = PossibleSync[i]->Sync;
    BasicBlock *FirstParent = First->getParent();
    for (unsigned long j = i + 1; j < PossibleSync.size(); j++) {
      Instruction *Second = PossibleSync[j]->Sync;
      BasicBlock *SecondParent = Second->getParent();
      if ((FirstParent == SecondParent) && (TaskParent == FirstParent)) {
        if (orderedInstructions(DT, First, Second) && orderedInstructions(DT, First, Task.Entry) &&
            orderedInstructions(DT, Second, Task.Entry)) {
          Redundant.push_back(PossibleSync[j]);
        } else
          Redundant.push_back(PossibleSync[i]);

      } else if (FirstParent == SecondParent) {
        if (orderedInstructions(DT, First, Second))
          Redundant.push_back(PossibleSync[j]);
        else
          Redundant.push_back(PossibleSync[i]);
      }
    }
  }

  SmallVector<SyncInfo *, 4> SyncsForPathing;

  // Remove redundant syncs
  for (auto &Possible : PossibleSync) {
    bool Exists = false;
    for (auto &Red : Redundant) {
      if (Possible == Red) {
        Exists = true;
        break;
      }
    }
    if (!Exists)
      SyncsForPathing.push_back(Possible);
  }
  Redundant.clear();

  // Calculate paths and find more redundant sync points
  for (auto &SyncPoint : SyncsForPathing) {
    std::vector<std::vector<BasicBlock *>> FinalPaths;
    getPaths(Task.Exit, SyncPoint->Sync, FinalPaths, DT, false);
    int InvalidPaths = 0;
    for (auto &Node : FinalPaths) {
      bool Invalid = false;
      for (auto &BB : Node) {
        for (auto &PointFound : SyncsForPathing) {
          if (BB == PointFound->Sync->getParent() && PointFound != SyncPoint) {
            if (BB != Task.Exit->getParent() ||
                (BB == Task.Exit->getParent() &&
                 orderedInstructions(DT, Task.Exit, PointFound->Sync))) {
              Invalid = true;
              InvalidPaths++;
              break;
            }
          }
        }
        if (Invalid)
          break;
      }
    }
    if (InvalidPaths == (int)FinalPaths.size()) {
      Redundant.push_back(SyncPoint);
    }
  }

  // Remove redundant syncs again
  for (auto &Possible : SyncsForPathing) {
    bool Exists = false;
    for (auto &Red : Redundant) {
      if (Possible == Red) {
        Exists = true;
        break;
      }
    }
    if (!Exists)
      Task.PostSyncs.push_back(Possible);
  }
}


void setConcurrentTasks(DirectiveInfo &Task, DominatorTree &DT,
                        SmallVectorImpl<DirectiveInfo *> &PostOrder) {

  SmallVector<DirectiveInfo *, 4> ConcurrentTasks;
  for (auto &PreSyncs : Task.PreSyncs) {
    std::vector<std::vector<BasicBlock *>> FinalPaths;
    getPaths(PreSyncs->Sync, Task.Entry, FinalPaths, DT, false);
    for (auto &Path : FinalPaths) {
      for (auto &Node : Path) {
        for (auto &FocusedTask : PostOrder) {
          if (FocusedTask->Entry->getParent() == Node && FocusedTask != &Task) {
            if ((Node == PreSyncs->Sync->getParent() &&
                 Node != Task.Entry->getParent() &&
                 (orderedInstructions(DT,PreSyncs->Sync, FocusedTask->Entry))) ||
                (Node == PreSyncs->Sync->getParent() &&
                 Node == Task.Entry->getParent() &&
                 (orderedInstructions(DT,PreSyncs->Sync, FocusedTask->Entry) || PreSyncs->Sync==FocusedTask->Entry) &&
                 (orderedInstructions(DT,FocusedTask->Entry, Task.Entry))) ||
                (Node != PreSyncs->Sync->getParent() &&
                 Node == Task.Entry->getParent() &&
                 (orderedInstructions(DT,FocusedTask->Entry, Task.Entry))) ||
                (Node != PreSyncs->Sync->getParent() &&
                 Node != Task.Entry->getParent())) {

              int Found = 0;
              for (auto &ConcurrentTask : ConcurrentTasks)
                if (ConcurrentTask == FocusedTask) {
                  Found = 1;
                  break;
                }

              if ((!isPotentiallyReachable(Task.Exit, FocusedTask->Entry) &&
                   !Found) &&
                  !(orderedInstructions(DT,FocusedTask->Entry, Task.Entry) &&
                    !orderedInstructions(DT,FocusedTask->Exit, Task.Exit))) {

                if (!existsTaskDependency(Task, *FocusedTask))
                  ConcurrentTasks.push_back(FocusedTask);
              }
            }
          }
        }
      }
    }
  }

  if (isPotentiallyReachable(Task.Exit, Task.Entry))
    ConcurrentTasks.push_back(&Task);

  for (auto &ConcurrentTask : ConcurrentTasks) {
    Task.ConcurrentBlocks.push_back(
        {ConcurrentTask->Entry, ConcurrentTask->Exit});
  }
}

void setConcurrentSequential(DirectiveInfo &Task, DominatorTree &DT,
                             SmallVectorImpl<SyncInfo> &SyncPoints,
                             SmallVectorImpl<DirectiveInfo *> &PostOrder) {

  SmallVector<BasicBlock *, 10> ProcessedBlocks;
  SmallVector<ConcurrentBlock *, 10> ConcurrentBlocks;
  Instruction *Exit = Task.Exit;
  for (auto &PostSync : Task.PostSyncs) {
    std::vector<std::vector<BasicBlock *>> FinalPaths;
    getPaths(Exit, PostSync->Sync, FinalPaths, DT, false);
    for (auto &Path : FinalPaths) {
      for (auto &Node : Path) {
        bool Already = false;
        for (auto &Processed : ProcessedBlocks) {
          if (Processed == Node)
            Already = true;
        }
        if (!Already)
          ProcessedBlocks.push_back(Node);

        bool End = false;
        ConcurrentBlock Block;
        for (auto &SyncPoint : SyncPoints) {
          bool visible = true;
          for (auto &SelectedTask : PostOrder) {
            if (orderedInstructions(DT,SelectedTask->Entry, SyncPoint.Sync) &&
                !orderedInstructions(DT,SelectedTask->Exit, SyncPoint.Sync) &&
                !(orderedInstructions(DT,SelectedTask->Entry, Task.Entry) &&
                  !orderedInstructions(DT,SelectedTask->Exit, Task.Entry))) {
              visible = false;
              break;
            }
          }
          if (!visible)
            continue;

          if (Node == SyncPoint.Sync->getParent() &&
              ((Node != Exit->getParent() ||
                (Node == Exit->getParent() &&
                 orderedInstructions(DT,Exit, SyncPoint.Sync))) &&
               (Node != PostSync->Sync->getParent() ||
                (Node == PostSync->Sync->getParent() &&
                 orderedInstructions(DT,SyncPoint.Sync, PostSync->Sync))))) {
            Block.Entry = &*(Node->begin());
            Block.Exit = SyncPoint.Sync;
            if (!Already)
              Task.ConcurrentBlocks.push_back(Block);
            End = true;
            break;
          }
        }
        if (End)
          break;

        if (Already)
          continue;

        if (Node != Exit->getParent())
          Block.Entry = &*(Node->begin());
        else
          Block.Entry = Exit;

        if (Node != PostSync->Sync->getParent())
          Block.Exit = Node->getTerminator();
        else
          Block.Exit = PostSync->Sync;
        Task.ConcurrentBlocks.push_back(Block);
      }
    }
  }
  if (isPotentiallyReachable(Exit, Task.Entry)) {
    std::vector<std::vector<BasicBlock *>> FinalPaths;
    getPaths(Exit, Task.Entry, FinalPaths, DT, true);
    int State = 0;
    for (auto &Path : FinalPaths) {
      int Count = 0;
      for (auto &Node : Path) {

        bool Already = false;
        if (Node == Exit->getParent() &&
            Exit->getParent() == Task.Entry->getParent()) {

          Count++;
          if (State == 1 || (Count == 1 && State == 0))
            Already = true;
          if (Count > 1 && State == 0)
            State = 1;

        } else
          for (auto &Processed : ProcessedBlocks) {
            if (Processed == Node)
              Already = true;
          }

        if (!Already)
          ProcessedBlocks.push_back(Node);

        bool End = false;
        ConcurrentBlock Block;
        for (auto &SyncPoint : SyncPoints) {
          bool visible = true;
          for (auto &SelectedTask : PostOrder) {
            if (orderedInstructions(DT,SelectedTask->Entry, SyncPoint.Sync) &&
                !orderedInstructions(DT,SelectedTask->Exit, SyncPoint.Sync) &&
                !(orderedInstructions(DT,SelectedTask->Entry, Task.Entry) &&
                  !orderedInstructions(DT,SelectedTask->Exit, Task.Entry))) {
              visible = false;
              break;
            }
          }
          if (!visible)
            continue;
          bool Finish = false;
          if (((Node == SyncPoint.Sync->getParent()) &&
               (Exit->getParent() == Task.Entry->getParent()) &&
               Node == Exit->getParent()) &&
              ((Count == 1 && orderedInstructions(DT,Exit, SyncPoint.Sync)) ||
               (Count > 1 && orderedInstructions(DT,SyncPoint.Sync, Task.Entry))))
            Finish = true;

          if ((Node == SyncPoint.Sync->getParent()) &&
              ((Node != Exit->getParent() ||
                (Node == Exit->getParent() &&
                 orderedInstructions(DT,Exit, SyncPoint.Sync))) &&
               (Node != Task.Entry->getParent() ||
                (Node == Task.Entry->getParent() &&
                 orderedInstructions(DT,SyncPoint.Sync, Task.Entry)))))
            Finish = true;

          if (Finish) {
            Block.Entry = &*(Node->begin());
            Block.Exit = SyncPoint.Sync;
            if (!Already)
              Task.ConcurrentBlocks.push_back(Block);
            End = true;
            break;
          }
        }
        if (End)
          break;
        if (Already)
          continue;

        if (Node != Exit->getParent() ||
            (Node == Exit->getParent() && Count > 1))
          Block.Entry = &*(Node->begin());
        else

          Block.Entry = Exit;

        if (Node != Task.Entry->getParent() ||
            (Node == Task.Entry->getParent() && Count == 1))
          Block.Exit = Node->getTerminator();
        else
          Block.Exit = Task.Entry;
        Task.ConcurrentBlocks.push_back(Block);
      }
    }
  }
}

Value *getArgValue(Value *argVar) {
  for (User *U : argVar->users()) {
    if (auto *store = dyn_cast<StoreInst>(U)) {
      if (store->getValueOperand() == argVar) {
        Value *newargVar = store->getPointerOperand();
        return newargVar;
      }
    }
  }
  return nullptr;
}


Instruction *getArgIns(Value *argVar) {
  for (User *U : argVar->users()) {
    if (auto *store = dyn_cast<StoreInst>(U)) {
      return dyn_cast<Instruction>(store);
    }
  }
  return nullptr;
}


// Core recursive function for detecting a variable usage, given a variable and
// an instruction
void valueInInstruction(Instruction *I, Value *Var, VarUse &Usage,
                        bool GetFirstUsage,
                        SmallPtrSetImpl<Function *> *AnalyzedFunctions,
                        SmallVector<ValueAccess, 4> &UsedInValues, bool record,
                        ValueAccess &AccessToCheck, AAResults &BAA, AliveValue TypeOfAlive, bool ContinueSubblocks) {

  // Is a store instruction
  if (auto *store = dyn_cast<StoreInst>(I)) {
    if (store->getPointerOperand() == Var ) {
      if (Usage != UNKNOWN)
        Usage = WRITTEN;
      if (record)
        UsedInValues.push_back({I, MemoryLocation::get(I), WRITTEN, false,
                                false, false, NOT_USED, false, false});
    } else if (Usage == NOT_USED && store->getValueOperand() == Var) {
      Usage = READED;
      if (record)
        UsedInValues.push_back({I, MemoryLocation::get(I), READED, false, false,
                                false, NOT_USED, false, false});
    }
    // Is a load instruction
  } else if (auto *load = dyn_cast<LoadInst>(I)) {
    if (load->getPointerOperand() == Var) {
      if (Usage == NOT_USED){
        Usage = READED;
      }
      if (record)
        UsedInValues.push_back({I, MemoryLocation::get(I), READED, false, false,
                                false, NOT_USED, false, false});
    }
    // Is a getelement instruction
  } else if (auto *getElm = dyn_cast<GetElementPtrInst>(I)) {
    if (getElm->getPointerOperand() == Var) {
      Value *getElmValue = dyn_cast<Value>(getElm);

      if (GetFirstUsage) {
        Instruction *sec = I->getNextNode();
        while (Usage == NOT_USED && sec != &(sec->getParent()->back())) {
          valueInInstruction(sec, getElmValue, Usage, GetFirstUsage,
                             AnalyzedFunctions, UsedInValues, record,
                             AccessToCheck, BAA, TypeOfAlive, ContinueSubblocks);
          sec = sec->getNextNode();
        }
      } else {
        for (User *getElmUser : getElmValue->users()) {
          Instruction *userIns = dyn_cast<Instruction>(getElmUser);
          valueInInstruction(userIns, getElmValue, Usage, GetFirstUsage,
                             AnalyzedFunctions, UsedInValues, record,
                             AccessToCheck, BAA, TypeOfAlive, ContinueSubblocks);
        }
      }
    }
    // Is a call instruction
  } else if (auto *call = dyn_cast<CallInst>(I)) {

    Function *F = call->getCalledFunction();
    bool exists = !(AnalyzedFunctions->find(F) == AnalyzedFunctions->end());
    if (!exists)
        AnalyzedFunctions->insert(I->getFunction());
    if (F && !F->isDeclaration() && !exists) {
      Value *argVar =nullptr;
      Value *finalVar = nullptr;
      
      //Needed for nested variable, in the call should not be nested to go to the function
      Value *VarAlternative= Var;
      if(User *prueba = dyn_cast<User>(Var)){
        if(prueba->getNumOperands()){
          VarAlternative = prueba->getOperand(0);
        }
      }
     

      if (call->hasArgument(Var) || call->hasArgument(VarAlternative)) {
        for (unsigned int i = 0; i < call->getNumOperands(); i++) {
          if (call->getArgOperand(i) == Var || call->getArgOperand(i) == VarAlternative) {
            Argument *arg = (F->arg_begin()) + i;
            argVar = dyn_cast<Value>(arg);
            finalVar = getArgValue(argVar);
          }
        }
      }
      else if (dyn_cast<GlobalValue>(Var) || dyn_cast<GlobalValue>(VarAlternative) ){
        argVar= Var;
      }
      else
        return;

      if (!GetFirstUsage) {
          if (finalVar != nullptr){
                llvm_unreachable("mem2reg analsis required");
              }
              else {
                for (User *lU : argVar->users()) {
                  if (Instruction *finalVarI = dyn_cast<Instruction>(lU))
                    valueInInstruction(finalVarI, argVar, Usage, GetFirstUsage,
                                       AnalyzedFunctions, UsedInValues, true,
                                       AccessToCheck, BAA, TypeOfAlive, ContinueSubblocks);
                
                }
              }
            } else {
              if (finalVar != nullptr) {
                llvm_unreachable("mem2reg analsis required");

                
              } else {


                SmallPtrSet<BasicBlock *, 10> AnalizedBasicBlocks;

                analyzeFirstTaskUse(&F->getEntryBlock(), argVar,
                                F->back().getTerminator() , F->getEntryBlock().getFirstNonPHI(),
                                &AnalizedBasicBlocks, AnalyzedFunctions,
                                AccessToCheck, BAA, TypeOfAlive, ContinueSubblocks);
              }
            }
      } else {
        if (F && F->getIntrinsicID() != Intrinsic::directive_region_entry && F->getIntrinsicID() != Intrinsic::directive_region_exit &&
            !exists) {
          MemoryLocation dummy;
          UsedInValues.push_back(
              {I, dummy, UNKNOWN, false, false, false, NOT_USED, false, false});
          Usage = UNKNOWN;
        }
      }
  
    // Is a bitcast instruction
  } else if (auto *BitCast = dyn_cast<CastInst>(I)) {
    Value *BitCastValue = dyn_cast<Value>(BitCast);
    if (BitCast->getOperand(0) == Var)
      for (User *BitCastUser : I->users()) {
        Instruction *userIns = dyn_cast<Instruction>(BitCastUser);
        valueInInstruction(userIns, BitCastValue, Usage, GetFirstUsage,
                           AnalyzedFunctions, UsedInValues, record,
                           AccessToCheck, BAA, TypeOfAlive, ContinueSubblocks);
      }
  }
}

void analyzeFirstTaskUse(BasicBlock *BB, Value *Var, Instruction *Exit,
                         Instruction *Entry,
                         SmallPtrSetImpl<BasicBlock *> *Already,
                         SmallPtrSetImpl<Function *> *AnalyzedFunctions,
                         ValueAccess &AccessToCheck, AAResults &BAA, AliveValue TypeOfAlive, bool ContinueSubblocks) {

  // Dont analyze first uses of unknown.
  if (!AccessToCheck.MemLoc.Ptr)
    return;

  Instruction *InsPointer;
  Already->insert(BB);

  if (Entry->getParent() == BB)
    InsPointer = Entry;
  else
    InsPointer = &(BB->front());

  VarUse Usage = NOT_USED;
  VarUse LocalUsage;
  SmallVector<ValueAccess, 4> AccessFound;

  while (InsPointer != Exit && InsPointer != &(BB->back()) &&
         Usage == NOT_USED) {

    LocalUsage = NOT_USED;
    valueInInstruction(InsPointer, Var, LocalUsage, true, AnalyzedFunctions,
                       AccessFound, true, AccessToCheck, BAA, TypeOfAlive, true);

    if (LocalUsage == UNKNOWN)
      Usage = LocalUsage;

    // Alias analyses dont work through different functions
    else if (LocalUsage != NOT_USED && (AccessFound[0].I->getFunction() !=
                                        AccessToCheck.I->getFunction()))
      Usage = LocalUsage;
    else if (LocalUsage != NOT_USED &&
             BAA.alias(AccessFound[0].MemLoc, AccessToCheck.MemLoc) ==
                 AliasResult::MustAlias)
      Usage = LocalUsage;

    // If an usage is found, fill it with priority caution because of
    // recursive calls
    if (Usage != NOT_USED) {

      if(Usage == READED || Usage == UNKNOWN){
        if (TypeOfAlive == AFTER_SYNC)
          AccessToCheck.IsAliveAfterNextSync = true;
        else if (TypeOfAlive == BEFORE_ENTRY){
           AccessToCheck.IsAliveBeforeEntry = true;
        }
        else if (TypeOfAlive == AFTER_EXIT)
          AccessToCheck.IsAliveAfterExit = true;
      }

      if (AccessToCheck.FirstTaskUse == NOT_USED)
        AccessToCheck.FirstTaskUse = Usage;
      else if (AccessToCheck.FirstTaskUse == READED &&
               (Usage == WRITTEN || Usage == UNKNOWN))
        AccessToCheck.FirstTaskUse = Usage;
    }

    AccessFound.clear();

    InsPointer = InsPointer->getNextNode();
  }

  // Check child blocks
  if (InsPointer == &(BB->back()) && Usage == NOT_USED && ContinueSubblocks) {
    for (BasicBlock *Succ : successors(BB)) {
      if (Already->find(Succ) == Already->end()) {
        analyzeFirstTaskUse(Succ, Var, Exit, Entry, Already, AnalyzedFunctions,
                            AccessToCheck, BAA, TypeOfAlive, ContinueSubblocks);
      }
    }
  }
}

bool isMustAlias(MemoryLocation first, MemoryLocation second, AAResults &BAA) {
  AliasResult Result = BAA.alias(first, second);
  dbgs() << "Alias result is " << Result << " \n";
  if (Result == AliasResult::MustAlias)
    return true;
  return false;
}

bool compareMemoryLocations(MemoryLocation first, MemoryLocation second,
                            AAResults &BAA) {
  AliasResult Result = BAA.alias(first, second);
  dbgs() << "Alias result is " << Result << " \n";
  if (Result == AliasResult::NoAlias)
    return false;
  return true;
}

//Handle possible global variable access from other function reacheable from task
bool UseIsInExternalCall(Instruction *I, Instruction *Entry, Instruction *Exit, DominatorTree &DT, SmallPtrSetImpl<Function *> &AnalyzedFunctions){

 Function *FunctionUsed= I->getFunction();
 AnalyzedFunctions.insert(FunctionUsed);
 if(FunctionUsed != Entry->getFunction()){
      for(User *FunctionUser : FunctionUsed->users()){
          if(Instruction *NewI = dyn_cast<Instruction>(FunctionUser)){
            if (NewI->getFunction() == Entry->getFunction() && orderedInstructions(DT, Entry, NewI) &&
            !orderedInstructions(DT, Exit, NewI)){
              return true;
            }
            else if(NewI->getFunction() != Entry->getFunction()){


              bool exists = !(AnalyzedFunctions.find(NewI->getFunction()) == AnalyzedFunctions.end());
              if(!exists){
                  if(UseIsInExternalCall (NewI, Entry, Exit, DT, AnalyzedFunctions))
                  return true;
              }
            }
      }
    }
  }
  return false;
}

void obtainCallsInside(Instruction *I, Instruction *Entry, Instruction *Exit, DominatorTree &DT, SmallPtrSetImpl<Function *> &AnalyzedFunctions, SmallVectorImpl<Instruction *> &CallList){


 Function *FunctionUsed= I->getFunction();
 AnalyzedFunctions.insert(FunctionUsed);
 if(FunctionUsed != Entry->getFunction()){
      for(User *FunctionUser : FunctionUsed->users()){
          if(Instruction *NewI = dyn_cast<Instruction>(FunctionUser)){
            if (NewI->getFunction() == Entry->getFunction() && orderedInstructions(DT, Entry, NewI) &&
            !orderedInstructions(DT, Exit, NewI)){
               CallList.push_back(NewI);
            }
            else if(NewI->getFunction() != Entry->getFunction()){
              bool exists = !(AnalyzedFunctions.find(NewI->getFunction()) == AnalyzedFunctions.end());
              if(!exists){
                  obtainCallsInside (NewI, Entry, Exit, DT, AnalyzedFunctions, CallList);
              }
            }
      }
    }
  }
}

OmpSsRegionAnalysis::OmpSsRegionAnalysis(Function &F, DominatorTree &DT, AAResults &BAA) {

  MapVector<BasicBlock *, SmallVector<Instruction *, 4>> BBDirectiveStacks;
  SmallVector<BasicBlock*, 8> Worklist;
  SmallPtrSet<BasicBlock*, 8> Visited;

  SmallVector<SyncInfo, 4> SyncPoints;

// Add the function start as the first sync point
  SyncPoints.push_back({F.getEntryBlock().getFirstNonPHI() , FUNCTION_START, 0});
  Worklist.push_back(&F.getEntryBlock());
  Visited.insert(&F.getEntryBlock());
  while (!Worklist.empty()) {
    auto WIt = Worklist.begin();
    BasicBlock *BB = *WIt;
    Worklist.erase(WIt);

    SmallVectorImpl<Instruction *> &Stack = BBDirectiveStacks[BB];

    for (Instruction &I : *BB) {

      // Add function end as the last sync point
      if (dyn_cast<ReturnInst>(&I))
        SyncPoints.push_back({&I, FUNCTION_END, (int)Stack.size()});

      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        if (II->getIntrinsicID() == Intrinsic::directive_region_entry) {
          assert(II->hasOneUse() && "Directive entry has more than one user.");

          Instruction *Exit = cast<Instruction>(II->user_back());
          // This should not happen because it will crash before this pass
          assert(orderedInstructions(DT, II, Exit) && "Directive entry does not dominate exit.");

          // directive.region pushes into the stack
          if (Stack.empty()) {
            // outer directive, insert into nullptr
            DirectivesTree[nullptr].push_back(II);
          } else {
            DirectivesTree[Stack.back()].push_back(II);
          }
          Stack.push_back(II);

          auto Dir = std::make_unique<DirectiveInfo>(II, Exit);
          if (!DisableChecks)
            Dir->DirEnv.verify();

          DEntryToDInfo.insert({II, std::move(Dir)});

        } else if (II->getIntrinsicID() == Intrinsic::directive_region_exit) {
          if (Stack.empty())
            llvm_unreachable("Directive exit hit without and entry.");

          Instruction *StackEntry = Stack.back();
          Instruction *StackExit = cast<Instruction>(StackEntry->user_back());
          assert(StackExit == II && "unexpected directive exit instr.");

          Stack.pop_back();
        } else if (II->getIntrinsicID() == Intrinsic::directive_marker) {
          SyncPoints.push_back({&I, TASKWAIT, (int)Stack.size()});
          // directive_marker does not push into the stack
          if (Stack.empty()) {
            // outer directive, insert into nullptr
            DirectivesTree[nullptr].push_back(II);
          } else {
            DirectivesTree[Stack.back()].push_back(II);
          }

          auto Dir = std::make_unique<DirectiveInfo>(II);
          if (!DisableChecks)
            Dir->DirEnv.verify();

          DEntryToDInfo.insert({II, std::move(Dir)});
        }
      } else if (!Stack.empty()) {
        Instruction *StackEntry = Stack.back();
        Instruction *StackExit = cast<Instruction>(StackEntry->user_back());
        DirectiveAnalysisInfo &DAI = DEntryToDAnalysisInfo[StackEntry];
        const DirectiveInfo *DI = DEntryToDInfo[StackEntry].get();
        const DirectiveEnvironment &DirEnv = DI->DirEnv;
        for (Use &U : I.operands()) {
          if (Instruction *I2 = dyn_cast<Instruction>(U.get())) {
            if (orderedInstructions(DT, I2, StackEntry)) {
              DAI.UsesBeforeEntry.insert(I2);
            if (!DisableChecks
                && !valueInDSABundles(DirEnv.DSAInfo, I2)
                && !valueInCapturedBundle(DirEnv.CapturedInfo, I2)) {
                llvm_unreachable("Value supposed to be inside directive entry "
                                 "OperandBundle not found.");
              }
            }
          } else if (Argument *A = dyn_cast<Argument>(U.get())) {
            DAI.UsesBeforeEntry.insert(A);
            if (!DisableChecks
                && !valueInDSABundles(DirEnv.DSAInfo, A)
                && !valueInCapturedBundle(DirEnv.CapturedInfo, A)) {
              llvm_unreachable("Value supposed to be inside directive entry "
                               "OperandBundle not found.");
            }
          }
        }
        for (User *U : I.users()) {
          if (Instruction *I2 = dyn_cast<Instruction>(U)) {
            if (orderedInstructions(DT, StackExit, I2)) {
              DAI.UsesAfterExit.insert(&I);
              if (!DisableChecks) {
                llvm_unreachable("Value inside the directive body used after it.");
              }
            }
          }
        }
      }
    }

    std::unique_ptr<std::vector<Instruction *>> StackCopy;

    for (auto It = succ_begin(BB); It != succ_end(BB); ++It) {
      if (!Visited.count(*It)) {
        Worklist.push_back(*It);
        Visited.insert(*It);
        // Forward Stack, since we are setting visited here
        // we do this only once per BB
        if (!StackCopy) {
          // We need to copy Stacki, otherwise &Stack as an iterator would be
          // invalidated after BBDirectiveStacks[*It].
          StackCopy.reset(
              new std::vector<Instruction *>(Stack.begin(), Stack.end()));
        }

        BBDirectiveStacks[*It].append(StackCopy->begin(), StackCopy->end());
      }
    }
  }

  convertDirectivesTreeToVector();

  int GnumUnknown = 0, GnumShared = 0, GnumFirstPrivate = 0, GnumPrivate = 0;

  for (auto &SelectedTask : DirectiveFuncInfo.PostOrder) {

    // Activate Autodependency
    bool AutoDeps = true;

    // Find pre sync points
    setPreSync(*SelectedTask, SyncPoints, DT, DirectiveFuncInfo.PostOrder);
    // Find post sync points
    setPostSync(*SelectedTask, SyncPoints, DT, DirectiveFuncInfo.PostOrder);
    // Find concurrent tasks
    setConcurrentTasks(*SelectedTask, DT, DirectiveFuncInfo.PostOrder);
    // Find concurrent sequential code
    setConcurrentSequential(*SelectedTask, DT, SyncPoints,
                            DirectiveFuncInfo.PostOrder);

     // Print concurrent code blocks
    printTaskConcurrentBlocks(*SelectedTask);

    SmallVector<std::pair<Value *, DSAValue>, 10> ScopedVariables;

    // Clean scope of variables
    for (Value *Var : SelectedTask->DirEnv.DSAInfo.Firstprivate)
      ScopedVariables.push_back(
          std::pair<Value *, DSAValue>(Var, FIRSTPRIVATE));
    SelectedTask->DirEnv.DSAInfo.Firstprivate.clear();

    for (Value *Var : SelectedTask->DirEnv.DSAInfo.Private)
      ScopedVariables.push_back(std::pair<Value *, DSAValue>(Var, PRIVATE));
    SelectedTask->DirEnv.DSAInfo.Private.clear();

    for (Value *Var : SelectedTask->DirEnv.DSAInfo.Shared)
      ScopedVariables.push_back(std::pair<Value *, DSAValue>(Var, SHARED));
    SelectedTask->DirEnv.DSAInfo.Shared.clear();


    //Requires so global variables used in the task are added as variables to analyze, even they are not detected by Clang
    for (auto &Global : F.getParent()->getGlobalList()){
        Value *Variable = dyn_cast<Value>(&Global);

        for (User *U : Variable->users()){
          SmallVector<Instruction *, 4> CallList;
          SmallPtrSet<Function *, 10> AnalyzedFunctions;

          if(Instruction *I = dyn_cast<Instruction>(U)){
            obtainCallsInside(I, SelectedTask->Entry, SelectedTask->Exit ,DT, AnalyzedFunctions, CallList);
          }
          else{
             for (User *UNew : U->users()) {
               if(Instruction *I = dyn_cast<Instruction>(UNew)){
                  obtainCallsInside(I, SelectedTask->Entry, SelectedTask->Exit ,DT, AnalyzedFunctions, CallList);
               }
             }
          }
          //If there is a call, that means that the variable is used inside the task
          if(CallList.size()){
             bool exists= false;
             for (auto AlreadyVariable : ScopedVariables){
               if(AlreadyVariable.first==Variable){
                 exists=true;
                 break;
               }
             }
             if(!exists)
              ScopedVariables.push_back(std::pair<Value *, DSAValue>(Variable, SHARED));

             break;
          }
        }
    }


    // Count scope of variables
    int numUnknown = 0, numShared = 0, numFirstPrivate = 0, numPrivate = 0;
    // Iterate over task variables
    for (auto Variable : ScopedVariables) {

      Value *Var = Variable.first;
      dbgs() << "\n\033[1mAnalyzing variable: \033[0m";
      Var->printAsOperand(dbgs(), false); 
      dbgs() << "\n";

      // Declare important attributes for the task, to be filled and used in
      // the algorithm
      bool AutoDepsActivated = false;
      DSAValue OriginalDSA = Variable.second;
      DSAValue CorrectedDSA = UNDEF;
      VarUse UsedInConcurrent = NOT_USED;
      VarUse UsedInTask = NOT_USED;
      bool IsGlobalVariable = false;
      bool IsComposited;

      // Find if the variable is composited
      if (Var->getType()->isPointerTy()){
       auto *VarType=
            dyn_cast<Type>(Var->getType()->getPointerElementType());
            if(VarType->isStructTy() || VarType->isArrayTy() || VarType->isPointerTy() || VarType->isVectorTy())
              IsComposited= true;
      }
      else
        IsComposited = false;

      // Find if the variable is a Pointer
      bool IsPointer;
      if (Var->getType()->isPointerTy())
        IsPointer = Var->getType()->getContainedType(0)->isPointerTy();

      // Vector for storing variable uses in concurrent blocks, inside and
      // outside tasks
      SmallVector<ValueAccess, 4> UsedInTaskValues;
      SmallVector<ValueAccess, 4> UsedInConcurrentValues;

      // Dummy ValueAccess, temporal fix
      ValueAccess dummy;


      for (User *U : Var->users()) {
        if (Instruction *I = dyn_cast<Instruction>(U)) {

          SmallPtrSet<Function *, 10> AnalyzedFunctions;

          int TaskContainsCall=UseIsInExternalCall(I, SelectedTask->Entry, SelectedTask->Exit, DT, AnalyzedFunctions );
  
          // Use is inside the task
          if ((orderedInstructions(DT, SelectedTask->Entry, I) &&
              !orderedInstructions(DT, SelectedTask->Exit, I)) || TaskContainsCall) {
            AnalyzedFunctions.clear();
            valueInInstruction(I, Var, UsedInTask, false, &AnalyzedFunctions,
                               UsedInTaskValues, true, dummy, BAA, NONE, true);
          }
          for (ConcurrentBlock &Block : SelectedTask->ConcurrentBlocks) {

            AnalyzedFunctions.clear();
            bool TaskContainsCall=UseIsInExternalCall(I, Block.Entry, Block.Exit ,DT, AnalyzedFunctions);

            // Use is inside a concurrent block
            if ((orderedInstructions(DT, Block.Entry, I) &&
                 !orderedInstructions(DT, Block.Exit, I)) ||
                I == Block.Entry || I == Block.Exit || TaskContainsCall) {
              AnalyzedFunctions.clear();
              int previousSize = UsedInConcurrentValues.size();
              VarUse PreviousUsedInConcurrent = UsedInConcurrent;
              valueInInstruction(I, Var, UsedInConcurrent, false,
                                 &AnalyzedFunctions, UsedInConcurrentValues,
                                 true, dummy, BAA, NONE, true);
              int numAdditions= UsedInConcurrentValues.size() - previousSize;
              // Check if the use was inside a task or not
              if(Block.Entry->getFunction() == I->getFunction()){
                for (auto ThisTask : DirectiveFuncInfo.PostOrder) {
                  
                  if ((orderedInstructions(DT, ThisTask->Entry, I) &&
                      !orderedInstructions(DT, ThisTask->Exit, I))) {
                    /*
                    bool isPrivatized= false;
                    for(auto VarScope : ThisTask.DSAInfo.Private){
                      if(VarScope == Var) isPrivatized =true;
                    }
                    for(auto VarScope : ThisTask.DSAInfo.Firstprivate){
                      if(VarScope == Var) isPrivatized =true;
                    }
                    */
                    //Check if exists a taskdependency, if it exists we remove the use since is no concurrent anymore
                    if(isPotentiallyReachable(SelectedTask->Exit, ThisTask->Entry) && !existsTaskDependency(*ThisTask, *SelectedTask)){
                      UsedInConcurrentValues.back().ConcurrentUseInTask = true;
                      break;
                    }
                    else{
                      UsedInConcurrentValues.pop_back();
                      break;
                    }
                  }
                }
              }
              else{
                //Obtain a vector of all calls inside the concurrent block
                SmallVector<Instruction *, 4> CallList;
                AnalyzedFunctions.clear();
                obtainCallsInside(I, Block.Entry, Block.Exit ,DT, AnalyzedFunctions, CallList);
                //See if it exists a call in the vector that is not inside in a task
                bool AllInsideTask=true;
                SmallVector<DirectiveInfo *,4> TaskWithCall;
                for(Instruction *Call : CallList){
                   bool InsideTask=false;
                   for (auto ThisTask : DirectiveFuncInfo.PostOrder) {
                       if ((orderedInstructions(DT, ThisTask->Entry, Call) &&
                      !orderedInstructions(DT, ThisTask->Exit, Call))){
                        TaskWithCall.push_back(ThisTask);
                        InsideTask=true;
                        break;
                      }
                   }

                   if(!InsideTask){
                     AllInsideTask=false;
                     break;
                   }
                }
                //If all are in tasks, check if all the tasks are syncronized with the main
                if(AllInsideTask){
                  bool AllSynchronized= true;
                  for( auto ThisTask: TaskWithCall){
                    /*
                    bool isPrivatized= false;
                    for(auto VarScope : ThisTask.DSAInfo.Private){
                      if(VarScope == Var) isPrivatized =true;
                    }
                    for(auto VarScope : ThisTask.DSAInfo.Firstprivate){
                      if(VarScope == Var) isPrivatized =true;
                    }
                    if(isPrivatized) continue;
                    */

                    if(!existsTaskDependency(*ThisTask, *SelectedTask)){
                      AllSynchronized=false;
                      break;
                    }
                  }
                  // If they are syncronized, erase the last values added, else mark ConcurrentUseInTask as true
                  if(AllSynchronized){
                    //Restore
                    UsedInConcurrent =PreviousUsedInConcurrent;
                    for(int i=0; i< numAdditions; i++)
                      UsedInConcurrentValues.pop_back();
                  }
                  else{
                     for(int i=0; i< numAdditions; i++)
                      UsedInConcurrentValues[UsedInConcurrentValues.size()-1-i].ConcurrentUseInTask=true;
                  }
                }
              }
            }
          }
        } else {
          // In case the use is not an instruction, check for its uses again
          for (User *UNew : U->users()) {
            if (Instruction *I = dyn_cast<Instruction>(UNew)) {
              SmallPtrSet<Function *, 10> AnalyzedFunctions;
              int TaskContainsCall=UseIsInExternalCall(I, SelectedTask->Entry, SelectedTask->Exit, DT, AnalyzedFunctions);

              if ((orderedInstructions(DT, SelectedTask->Entry, I) &&
                  !orderedInstructions(DT, SelectedTask->Exit, I)) || TaskContainsCall) {
                AnalyzedFunctions.clear();
                valueInInstruction(I, U, UsedInTask, false, &AnalyzedFunctions,
                                   UsedInTaskValues, true, dummy, BAA, NONE , true);
              }
              for (ConcurrentBlock &Block : SelectedTask->ConcurrentBlocks) {

                AnalyzedFunctions.clear();
                int TaskContainsCall=UseIsInExternalCall(I, Block.Entry, Block.Exit ,DT, AnalyzedFunctions);

                // Use is inside a concurrent block
                if ((orderedInstructions(DT, Block.Entry, I) &&
                    !orderedInstructions(DT, Block.Exit, I)) ||
                    I == Block.Entry || I == Block.Exit || TaskContainsCall) {
                  AnalyzedFunctions.clear();
                  int previousSize = UsedInConcurrentValues.size();
                  VarUse PreviousUsedInConcurrent = UsedInConcurrent;
                  valueInInstruction(I, U, UsedInConcurrent, false,
                                    &AnalyzedFunctions, UsedInConcurrentValues,
                                    true, dummy, BAA, NONE, true);

                  int numAdditions= UsedInConcurrentValues.size() - previousSize;

                  // Check if the use was inside a task or not
                  if(Block.Entry->getFunction() == I->getFunction()){

                    for (auto ThisTask : DirectiveFuncInfo.PostOrder) {

                      if ((orderedInstructions(DT, ThisTask->Entry, I) &&
                          !orderedInstructions(DT, ThisTask->Exit, I))) {
                        /*
                        bool isPrivatized= false;
                        for(auto VarScope : ThisTask.DSAInfo.Private){
                          if(VarScope == Var) isPrivatized =true;
                        }
                        for(auto VarScope : ThisTask.DSAInfo.Firstprivate){
                          if(VarScope == Var) isPrivatized =true;
                        }
                        */
                        //Check if exists a taskdependency, if it exists we remove the use since is no concurrent anymore
                        if(isPotentiallyReachable(SelectedTask->Exit, ThisTask->Entry) && !existsTaskDependency(*ThisTask, *SelectedTask)){
                          UsedInConcurrentValues.back().ConcurrentUseInTask = true;
                          break;
                        }
                        else{
                          UsedInConcurrentValues.pop_back();
                          break;
                        }
                      }
                    }
                  }
                  else{

                    //Obtain a vector of all calls inside the concurrent block
                    SmallVector<Instruction *, 4> CallList;
                    AnalyzedFunctions.clear();
                    obtainCallsInside(I, Block.Entry, Block.Exit ,DT, AnalyzedFunctions, CallList);
                    //See if it exists a call in the vector that is not inside in a task
                    bool AllInsideTask=true;
                    SmallVector<DirectiveInfo *,4> TaskWithCall;
                    for(Instruction *Call : CallList){

                      bool InsideTask=false;
                      for (auto ThisTask : DirectiveFuncInfo.PostOrder) {
                          if ((orderedInstructions(DT, ThisTask->Entry, Call) &&
                          !orderedInstructions(DT, ThisTask->Exit, Call))){
                            TaskWithCall.push_back(ThisTask);
                            InsideTask=true;
                            break;
                          }
                      }

                      if(!InsideTask){
                        AllInsideTask=false;
                        break;
                      }
                    }
                    //If all are in tasks, check if all the tasks are syncronized with the main
                    if(AllInsideTask){
                      bool AllSynchronized= true;
                      for( auto ThisTask: TaskWithCall){
                        /*
                        bool isPrivatized= false;
                        for(auto VarScope : ThisTask.DSAInfo.Private){
                          if(VarScope == Var) isPrivatized =true;
                        }
                        for(auto VarScope : ThisTask.DSAInfo.Firstprivate){
                          if(VarScope == Var) isPrivatized =true;
                        }
                        if(isPrivatized) continue;
                        */
                        if(!existsTaskDependency(*ThisTask, *SelectedTask)){
                          AllSynchronized=false;
                          break;
                        }
                      }
                      // If they are syncronized, erase the last values added, else mark ConcurrentUseInTask as true
                      if(AllSynchronized){
                        //Restore
                        UsedInConcurrent =PreviousUsedInConcurrent;
                        for(int i=0; i< numAdditions; i++)
                          UsedInConcurrentValues.pop_back();
                      }
                      else{
                        for(int i=0; i< numAdditions; i++)
                          UsedInConcurrentValues[UsedInConcurrentValues.size()-1-i].ConcurrentUseInTask=true;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
     // Auxiliar vectors for storing already visited functions and blocks
      SmallPtrSet<BasicBlock *, 10> AnalizedBasicBlocks;
      SmallPtrSet<Function *, 10> AnalyzedFunctions;

      // Analyze the first use inside the task, to know if it is a read or a
      // write
      for (User *U : Var->users()) {
        if (!dyn_cast<Instruction>(U)) {
          // Case the use is not an instruction
          for (auto &AccessToCheck : UsedInTaskValues) {
            AnalizedBasicBlocks.clear();
            AnalyzedFunctions.clear();
            analyzeFirstTaskUse(SelectedTask->Entry->getParent(), U,
                                SelectedTask->Exit, SelectedTask->Entry,
                                &AnalizedBasicBlocks, &AnalyzedFunctions,
                                AccessToCheck, BAA, BEFORE_ENTRY, true);
          }
        }
      }

      // Case the use is an instruction
      for (auto &AccessToCheck : UsedInTaskValues) {
        AnalizedBasicBlocks.clear();
        AnalyzedFunctions.clear();
        analyzeFirstTaskUse(SelectedTask->Entry->getParent(), Var,
                            SelectedTask->Exit, SelectedTask->Entry,
                            &AnalizedBasicBlocks, &AnalyzedFunctions,
                            AccessToCheck, BAA , BEFORE_ENTRY, true);
      }

      // Analyze variable is alive after next sync
      for (SyncInfo *nextSync : SelectedTask->PostSyncs)
        for (auto &AccessToCheck : UsedInTaskValues) {
          AnalizedBasicBlocks.clear();
          AnalyzedFunctions.clear();
          for (auto U : Var->users())
            if (!dyn_cast<Instruction>(U))
              analyzeFirstTaskUse(nextSync->Sync->getParent(), U, nullptr,
                             nextSync->Sync, &AnalizedBasicBlocks,
                             &AnalyzedFunctions, AccessToCheck, BAA, AFTER_SYNC,
                             true);
          AnalizedBasicBlocks.clear();
          AnalyzedFunctions.clear();
          analyzeFirstTaskUse(nextSync->Sync->getParent(), Var, nullptr,
                         nextSync->Sync, &AnalizedBasicBlocks,
                         &AnalyzedFunctions, AccessToCheck, BAA, AFTER_SYNC,
                         true);
        }

      // Analyze variable is alive after task exit
      for (auto &SyncPoint : SelectedTask->PostSyncs) {
        std::vector<std::vector<BasicBlock *>> FinalPaths;
        getPaths(SelectedTask->Exit, SyncPoint->Sync, FinalPaths, DT, false);
        for (auto &Node : FinalPaths) {
          for (auto &BB : Node) {
            for (auto &AccessToCheck : UsedInTaskValues) {
              AnalizedBasicBlocks.clear();
              AnalyzedFunctions.clear();
              for (auto U : Var->users())
                if (!dyn_cast<Instruction>(U))
                  analyzeFirstTaskUse(BB, U, SyncPoint->Sync, SelectedTask->Exit,
                                 &AnalizedBasicBlocks, &AnalyzedFunctions,
                                 AccessToCheck, BAA, AFTER_EXIT, false);

              AnalizedBasicBlocks.clear();
              AnalyzedFunctions.clear();
              analyzeFirstTaskUse(BB, Var, SyncPoint->Sync, SelectedTask->Exit,
                             &AnalizedBasicBlocks, &AnalyzedFunctions,
                             AccessToCheck, BAA, AFTER_EXIT, false);
            }
          }
        }
      }

      // Check if the variable is global
      if (dyn_cast<GlobalValue>(Var))
        IsGlobalVariable = true;

      LLVM_DEBUG(dbgs() << "Var " << Var->getName() << " is "
                        << VarUseToString[UsedInConcurrent]
                        << " in a concurrent region and "
                        << VarUseToString[UsedInTask] << " inside the task"
                        << ", global: " << IsGlobalVariable << " Composited: "
                        << IsComposited << " Pointer " << IsPointer << " \n");
      SmallVector<ValueAccess, 4> DefinitiveUsedValues;
      SmallVector<ValueAccess, 4> ProcessedValueAccess;

      SmallVector<ValueAccess, 4> CopyUsedInTaskValues = UsedInTaskValues;
      // Eliminate uses of same memory address
      for (ValueAccess FirstTaskMemUse : UsedInTaskValues) {

        bool Already = false;
        for (auto &VA : ProcessedValueAccess) {
          if ((FirstTaskMemUse.MemLoc.Ptr != nullptr &&
               VA.MemLoc.Ptr != nullptr) &&
              ((FirstTaskMemUse.I->getFunction() != VA.I->getFunction()) ||
               (BAA.alias(FirstTaskMemUse.MemLoc, VA.MemLoc) != AliasResult::NoAlias))) {
            Already = true;
            break;
          }
        }
        if (Already)
          continue;
        ProcessedValueAccess.push_back(FirstTaskMemUse);

        SmallVector<VarUse, 4> Scopes;
        Scopes.push_back(FirstTaskMemUse.Use);

        for (ValueAccess SecondTaskMemUse : UsedInTaskValues) {
          if (FirstTaskMemUse.I != SecondTaskMemUse.I &&
              FirstTaskMemUse.MemLoc.Ptr != nullptr &&
              SecondTaskMemUse.MemLoc.Ptr != nullptr) {
            if ((FirstTaskMemUse.I->getFunction() !=
                 SecondTaskMemUse.I->getFunction()) ||
                (BAA.alias(FirstTaskMemUse.MemLoc, SecondTaskMemUse.MemLoc) !=
                 AliasResult::NoAlias)) {
              Scopes.push_back(SecondTaskMemUse.Use);
            }
          }
        }

        VarUse FinalUse = NOT_USED;
        for (auto TypeOfUse : Scopes) {
          if (TypeOfUse == UNKNOWN) {
            FinalUse = TypeOfUse;
            break;
          }
          if (FinalUse == NOT_USED ||
              (TypeOfUse == WRITTEN && FinalUse == READED))
            FinalUse = TypeOfUse;
        }
        FirstTaskMemUse.Use = FinalUse;
        DefinitiveUsedValues.push_back(FirstTaskMemUse);
      }
      UsedInTaskValues = DefinitiveUsedValues;

      DefinitiveUsedValues.clear();
      ProcessedValueAccess.clear();

      // Eliminate uses of same memory address
      for (ValueAccess &FirstConcurrentMemUse : UsedInConcurrentValues) {

        bool Already = false;
        for (auto &VA : ProcessedValueAccess) {
          if ((FirstConcurrentMemUse.MemLoc.Ptr != nullptr &&
               VA.MemLoc.Ptr != nullptr) &&
              ((FirstConcurrentMemUse.I->getFunction() !=
                VA.I->getFunction()) ||
               (BAA.alias(FirstConcurrentMemUse.MemLoc, VA.MemLoc) !=
                AliasResult::NoAlias))) {
            Already = true;
            break;
          }
        }
        if (Already)
          continue;
        ProcessedValueAccess.push_back(FirstConcurrentMemUse);

        SmallVector<VarUse, 4> Scopes;
        Scopes.push_back(FirstConcurrentMemUse.Use);

        for (ValueAccess SecondConcurrentMemUse : UsedInConcurrentValues) {
          if (FirstConcurrentMemUse.I != SecondConcurrentMemUse.I &&
              FirstConcurrentMemUse.MemLoc.Ptr != nullptr &&
              SecondConcurrentMemUse.MemLoc.Ptr != nullptr) {
            if ((FirstConcurrentMemUse.I->getFunction() !=
                 SecondConcurrentMemUse.I->getFunction()) ||
                (BAA.alias(FirstConcurrentMemUse.MemLoc,
                           SecondConcurrentMemUse.MemLoc) != AliasResult::NoAlias)) {
              Scopes.push_back(SecondConcurrentMemUse.Use);
              //Required to know if there is a concurrent use outside the task, since we are eliminating it
              if(FirstConcurrentMemUse.ConcurrentUseInTask == true
              && SecondConcurrentMemUse.ConcurrentUseInTask == false){
                FirstConcurrentMemUse.ConcurrentUseInTask= false;
              } 
            }
          }
        }

        VarUse FinalUse = NOT_USED;
        for (auto TypeOfUse : Scopes) {
          if (TypeOfUse == UNKNOWN) {
            FinalUse = TypeOfUse;
            break;
          }
          if (FinalUse == NOT_USED ||
              (TypeOfUse == WRITTEN && FinalUse == READED))
            FinalUse = TypeOfUse;
        }
        FirstConcurrentMemUse.Use = FinalUse;
        DefinitiveUsedValues.push_back(FirstConcurrentMemUse);
      }
      UsedInConcurrentValues = DefinitiveUsedValues;

      // Analyze variables that satisfy the conditions
      if (UsedInTask != UNKNOWN && UsedInTask != NOT_USED &&
          UsedInConcurrent != UNKNOWN) {

        SmallVector<MemoryAccessDescription, 4> AllUses;
        SmallVector<DSAValue, 4> DSA;

        for (ValueAccess TaskMemUse : UsedInTaskValues) {

          VarUse LocalUsedInTask = TaskMemUse.Use;
          VarUse LocalUsedInConcurrent = NOT_USED;
          VarUse FirstTaskUse = TaskMemUse.FirstTaskUse;
          bool IsAliveAfterNextSync = TaskMemUse.IsAliveAfterNextSync;
          bool Found = false;
          LLVM_DEBUG(dbgs() << "Alive before entry: " << TaskMemUse.IsAliveBeforeEntry << " Alive after Exit: " << TaskMemUse.IsAliveAfterExit << " Alive after next sync: " << TaskMemUse.IsAliveAfterNextSync << " \n");

          for (ValueAccess ConcurrentMemUse : UsedInConcurrentValues) {

            if ((TaskMemUse.I->getFunction() !=
                 ConcurrentMemUse.I->getFunction()) ||
                BAA.alias(TaskMemUse.MemLoc, ConcurrentMemUse.MemLoc) !=
                    AliasResult::NoAlias) {
              Found = true;
              LocalUsedInConcurrent = ConcurrentMemUse.Use;
              AllUses.push_back({LocalUsedInTask, LocalUsedInConcurrent,
                                 FirstTaskUse, IsAliveAfterNextSync,
                                 ConcurrentMemUse.ConcurrentUseInTask});
            }
          }
          if (!Found) {
            AllUses.push_back({LocalUsedInTask, LocalUsedInConcurrent,
                               FirstTaskUse, IsAliveAfterNextSync, false});
          }
        }

        for (auto ThisUse : AllUses) {
          VarUse TaskUse = ThisUse.UsedInTask;
          VarUse ConcurrentUse = ThisUse.UsedInConcurrent;
          VarUse FirstInTask = ThisUse.FirstTaskUse;
          bool IsAliveAfterNextSync = ThisUse.IsAliveAfterNextSync;
          bool ConcurrentUseInTask = ThisUse.ConcurrentUseInTask;

          LLVM_DEBUG(dbgs()
                     << " Use found, task use " << VarUseToString[TaskUse]
                     << " ,concurrent use " << VarUseToString[ConcurrentUse]
                     << " ,first in task " << VarUseToString[FirstInTask]
                     << " , is alive " << IsAliveAfterNextSync
                     << " , ConcurrentUseInTask " << ConcurrentUseInTask
                     << " \n");

          if (ConcurrentUse == NOT_USED) {
            if (TaskUse == READED)
              DSA.push_back(SHARED_OR_FIRSTPRIVATE);
            else if (TaskUse == WRITTEN) {
              if (IsGlobalVariable || IsAliveAfterNextSync)
                DSA.push_back(SHARED);
              else if (FirstInTask == WRITTEN)
                DSA.push_back(PRIVATE);
              else if (FirstInTask == READED || FirstInTask == UNKNOWN)
                DSA.push_back(SHARED_OR_FIRSTPRIVATE);
            }
          } else {
            if (TaskUse == READED && ConcurrentUse == READED)
              DSA.push_back(SHARED_OR_FIRSTPRIVATE);
            else if (TaskUse == WRITTEN || ConcurrentUse == WRITTEN) {
              // TODO: No Data Race (critical section)
              if (AutoDeps) {
                if (IsAliveAfterNextSync && !ConcurrentUseInTask)
                  DSA.push_back(UNDEF);
                else if (!ConcurrentUseInTask) {
                  if (FirstInTask == WRITTEN)
                    DSA.push_back(RACEPRIVATE);
                  else
                    DSA.push_back(RACEFIRSTPRIVATE);
                } else {
                  dbgs()<< "Possible race condition with other tasks, running Autodeps \n";
                  AutoDepsActivated=true;
                  DSA.push_back(SHARED);
                }
              } else {
                if (IsAliveAfterNextSync)
                  DSA.push_back(UNDEF);
                else if (FirstInTask == WRITTEN)
                  DSA.push_back(RACEPRIVATE);
                else
                  DSA.push_back(RACEFIRSTPRIVATE);
              }
            }
          }
        }

        if (IsComposited) {

          SmallVector<std::pair<DSAValue, DSAValue>, 4> DSAPairs;

          for (int i = 0; i < (int)DSA.size(); i++)
            for (int j = i; j < (int)DSA.size(); j++)
              DSAPairs.push_back({DSA[i], DSA[j]});

          bool AllEqual = true;
          bool AllRaces = true;
          bool ExistsUndefined = false;
          bool ConditionA = false;
          bool ConditionB = true;
          bool ConditionC = true;
          bool ConditionC1 = true;
          bool ConditionD = false;

          for (auto Pair : DSAPairs) {
            if (Pair.first != Pair.second)
              AllEqual = false;

            if (Pair.first == UNDEF || Pair.second == UNDEF)
              ExistsUndefined = true;

            if (((Pair.first == RACEFIRSTPRIVATE ||
                  Pair.first == RACEPRIVATE) &&
                 Pair.second == SHARED) ||
                (((Pair.second == RACEFIRSTPRIVATE ||
                   Pair.second == RACEPRIVATE) &&
                  Pair.first == SHARED)))
              ConditionA = true;

            if (!((Pair.first == SHARED_OR_FIRSTPRIVATE ||
                   Pair.first == SHARED) &&
                  (Pair.second == SHARED_OR_FIRSTPRIVATE ||
                   Pair.second == SHARED)))
              ConditionB = false;

            if (!((Pair.first == RACEPRIVATE || Pair.first == PRIVATE) &&
                  (Pair.second == RACEPRIVATE || Pair.second == PRIVATE)))
              ConditionC = false;

            if (!((Pair.first == RACEFIRSTPRIVATE || Pair.first == PRIVATE) &&
                  (Pair.second == RACEFIRSTPRIVATE || Pair.second == PRIVATE)))
              ConditionC1 = false;

            if ((Pair.first == SHARED_OR_FIRSTPRIVATE &&
                 (Pair.second == RACEPRIVATE ||
                  Pair.second == RACEFIRSTPRIVATE || Pair.second == PRIVATE)) ||
                (Pair.second == SHARED_OR_FIRSTPRIVATE &&
                 (Pair.first == RACEPRIVATE || Pair.first == RACEFIRSTPRIVATE ||
                  Pair.first == PRIVATE)))
              ConditionD = true;

            if (!((Pair.first == RACEPRIVATE ||
                   Pair.first == RACEFIRSTPRIVATE) &&
                  (Pair.second == RACEPRIVATE ||
                   Pair.second == RACEFIRSTPRIVATE)))
              AllRaces = false;
          }

          if (AllEqual || DSA.size() == 1)
            CorrectedDSA = DSA[0];
          else if (ExistsUndefined || ConditionA)
            CorrectedDSA = UNDEF;
          else if (ConditionB)
            CorrectedDSA = SHARED;
          else if (ConditionC)
            CorrectedDSA = PRIVATE;
          else if (ConditionD || ConditionC1 || AllRaces)
            CorrectedDSA = FIRSTPRIVATE;
          else
            assert(false && "Error: DSA not found");
        } else {

          DSAValue FinalDSA = UNINITIALIZED;

          for (auto ThisDSA : DSA) {
            if (ThisDSA == UNDEF) {
              FinalDSA = ThisDSA;
              break;
            }
            if (ThisDSA == FIRSTPRIVATE || ThisDSA == RACEFIRSTPRIVATE) {
              FinalDSA = ThisDSA;
            }
            if ((ThisDSA == PRIVATE || ThisDSA == RACEPRIVATE) &&
                (FinalDSA == SHARED_OR_FIRSTPRIVATE || FinalDSA == SHARED ||
                 FinalDSA == UNINITIALIZED))
              FinalDSA = ThisDSA;

            if (ThisDSA == SHARED && (FinalDSA == UNINITIALIZED ||
                                      FinalDSA == SHARED_OR_FIRSTPRIVATE))
              FinalDSA = ThisDSA;

            if (ThisDSA == SHARED_OR_FIRSTPRIVATE &&
                (FinalDSA == UNINITIALIZED))
              FinalDSA = ThisDSA;
          }

          CorrectedDSA = FinalDSA;

          assert(CorrectedDSA != UNINITIALIZED && "Error: DSA not found");
        }

        if (CorrectedDSA == RACEPRIVATE){
          CorrectedDSA = PRIVATE;
          dbgs() << "Race condition detected, can not be solved with autodeps, privatizing the variable \n";
        }
        else if (CorrectedDSA == RACEFIRSTPRIVATE){
          dbgs() << "Race condition detected, can not be solved with autodeps, firstprivatizing the variable \n";
          CorrectedDSA = FIRSTPRIVATE;
        }
      }

      bool isUnknown = false;
      ;
      if (CorrectedDSA == UNDEF) {
        if (UsedInTask != NOT_USED) {
          isUnknown = true;
          numUnknown++;
        }
        CorrectedDSA = OriginalDSA;
      } else if (CorrectedDSA == SHARED)
        numShared++;
      else if (CorrectedDSA == PRIVATE)
        numPrivate++;
      else if (CorrectedDSA == FIRSTPRIVATE)
        numFirstPrivate++;
      else if (CorrectedDSA == SHARED_OR_FIRSTPRIVATE) {
        if (IsComposited || IsPointer) {
          numShared++;
        } else {
          numFirstPrivate++;
        }
      }

      if (CorrectedDSA == SHARED)
        SelectedTask->DirEnv.DSAInfo.Shared.insert(Var);
      else if (CorrectedDSA == PRIVATE)
        SelectedTask->DirEnv.DSAInfo.Private.insert(Var);
      else if (CorrectedDSA == FIRSTPRIVATE)
        SelectedTask->DirEnv.DSAInfo.Firstprivate.insert(Var);
      else if (CorrectedDSA == SHARED_OR_FIRSTPRIVATE) {
        if (IsComposited || IsPointer) {
          SelectedTask->DirEnv.DSAInfo.Shared.insert(Var);
          CorrectedDSA = SHARED;
        } else {
          SelectedTask->DirEnv.DSAInfo.Firstprivate.insert(Var);
          CorrectedDSA = FIRSTPRIVATE;
        }
      }

      if (PrintVerboseLevel == PV_AutoScoping) {
        Var->printAsOperand(dbgs(), false);
        if (isUnknown)
          dbgs() << " detected scope: UNKNOWN";
        else
          dbgs() << " detected scope: " << DSAToString[CorrectedDSA];

        if (CorrectedDSA != OriginalDSA)
          dbgs() << ", modified original scope was: "
                 << DSAToString[OriginalDSA];

        dbgs() << " \n";
      }
    
      if (AutoDeps && AutoDepsActivated && CorrectedDSA == SHARED) {

         int VarDependency = DependInfo::DT_unknown;

        for (auto VarUse : CopyUsedInTaskValues) {

          bool UsedBeforeEntry = false;
          bool UsedAfterExit = VarUse.IsAliveAfterExit;
          bool ConditionA = true;
    
          if (VarUse.IsAliveBeforeEntry) {

            UsedBeforeEntry = true;
            bool UsedInConcurrent = false;
            bool UsedInATask = false;

            for (auto ConcurrentUse : UsedInConcurrentValues)
              if (ConcurrentUse.I == VarUse.I)
                UsedInConcurrent = true;

            // TODO ONLY TASK at same level
            for (auto ThisTask : DirectiveFuncInfo.PostOrder) {
              if (orderedInstructions(DT, ThisTask->Entry, VarUse.I) &&
                  !orderedInstructions(DT, ThisTask->Exit, VarUse.I))
                UsedInATask = true;
            }

            if (!UsedInATask && UsedInConcurrent) {
              ConditionA = false;
            }
          }

          if (UsedBeforeEntry && UsedAfterExit && UsedInTask==WRITTEN) {
            VarDependency= DependInfo::DT_inout;
            //dbgs() << "VAR SHOULD BE INOUT \n";
          } else if (UsedAfterExit && UsedInTask==WRITTEN) {
             if(VarDependency == DependInfo::DT_unknown)
              VarDependency = DependInfo::DT_out;
             else if(VarDependency==DependInfo::DT_in)
              VarDependency= DependInfo::DT_inout;
              //dbgs() << "VAR SHOULD BE OUT \n";

          } else if (UsedBeforeEntry) {
            if (ConditionA){
              if(VarDependency == DependInfo::DT_unknown)
              VarDependency = DependInfo::DT_in;
             else if(VarDependency==DEP_OUT)
              VarDependency=DependInfo::DT_inout;
              //dbgs() << "VAR SHOULD BE INPUT \n";
            }
          }

          //dbgs() << " Before entry " << UsedBeforeEntry << " After Exit "
           //      << UsedAfterExit << " Condition A  " << ConditionA << " \n";
        }

        if (VarDependency != DependInfo::DT_unknown) {
          bool correctDEP = false;
          for (auto &Dep : SelectedTask->DirEnv.DependsInfo.List) {
            if (Var == Dep->Base && Dep->DepType == VarDependency) {
              correctDEP = true;
              break;
            }
          }
          if (!correctDEP) {
            dbgs() << "\033[1;31mPossible ERROR: ";
            Var->printAsOperand(dbgs(), false);
            dbgs() << " should be " << VarDependencyToString[VarDependency]
                   << " \033[0m\n";
          } else {
            dbgs() << "\033[1;32mDependencies seems OK! \033[0m\n";
          }
        }
      } else if (!AutoDepsActivated) {
        bool depExists= false;
        for (auto &Dep : SelectedTask->DirEnv.DependsInfo.List) {
          if (Var == Dep->Base) {
            depExists = true;
            break;
          }
        }
        
        if(depExists){
          dbgs() << "\033[1;35mPossible Warning: ";
          Var->printAsOperand(dbgs(), false);
          dbgs() << " should not be a dependency \033[0m\n";
        }
      }
    }
    LLVM_DEBUG(dbgs() << "Num SHARED " << numShared << " Num PRIVATE "
                      << numPrivate << " Num FIRSTPRIVATE " << numFirstPrivate
                      << " Num UNDEF " << numUnknown << " \n");

    GnumShared += numShared;
    GnumPrivate += numPrivate;
    GnumFirstPrivate += numFirstPrivate;
    GnumUnknown += numUnknown;
  }
  if ((GnumShared + GnumPrivate + GnumUnknown + GnumFirstPrivate) > 0)
    LLVM_DEBUG(dbgs() << "GLOBAL Num SHARED " << GnumShared << " Num PRIVATE "
                      << GnumPrivate << " Num FIRSTPRIVATE " << GnumFirstPrivate
                      << " Num UNDEF " << GnumUnknown << " \n");
}

// OmpSsRegionAnalysisLegacyPass
//
bool OmpSsRegionAnalysisLegacyPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &BAA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  ORA = OmpSsRegionAnalysis(F, DT, BAA);

  return false;
}

void OmpSsRegionAnalysisLegacyPass::releaseMemory() {
  ORA = OmpSsRegionAnalysis();
}

void OmpSsRegionAnalysisLegacyPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<AAResultsWrapperPass>();
}

char OmpSsRegionAnalysisLegacyPass::ID = 0;

OmpSsRegionAnalysisLegacyPass::OmpSsRegionAnalysisLegacyPass() : FunctionPass(ID) {
  initializeOmpSsRegionAnalysisLegacyPassPass(*PassRegistry::getPassRegistry());
}

void OmpSsRegionAnalysisLegacyPass::print(raw_ostream &OS, const Module *M) const {
  ORA.print(OS);
}

OmpSsRegionAnalysis& OmpSsRegionAnalysisLegacyPass::getResult() { return ORA; }

INITIALIZE_PASS_BEGIN(OmpSsRegionAnalysisLegacyPass, "ompss-2-regions",
                      "Classify OmpSs-2 inside region uses", false, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(OmpSsRegionAnalysisLegacyPass, "ompss-2-regions",
                    "Classify OmpSs-2 inside region uses", false, true)


// OmpSsRegionAnalysisPass
//
AnalysisKey OmpSsRegionAnalysisPass::Key;

OmpSsRegionAnalysis OmpSsRegionAnalysisPass::run(
    Function &F, FunctionAnalysisManager &FAM) {
  auto *DT = &FAM.getResult<DominatorTreeAnalysis>(F);
  auto *BAA = &FAM.getResult<AAManager>(F);

  return OmpSsRegionAnalysis(F, *DT, *BAA);
}

// OmpSsRegionPrinterPass
//
OmpSsRegionPrinterPass::OmpSsRegionPrinterPass(raw_ostream &OS) : OS(OS) {}

PreservedAnalyses OmpSsRegionPrinterPass::run(
    Function &F, FunctionAnalysisManager &FAM) {
  FAM.getResult<OmpSsRegionAnalysisPass>(F).print(OS);

  return PreservedAnalyses::all();
}
