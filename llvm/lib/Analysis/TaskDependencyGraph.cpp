//===- TaskDependencyGraph.cpp - Generation of a static task dependency graph
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Detects single entry single exit regions in the control flow graph.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TaskDependencyGraph.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include <fstream>
#include <iostream>

using namespace llvm;
const StringRef color_names[] = {
    "aquamarine3", "crimson",         "chartreuse",  "blue2",
    "darkorchid3", "darkgoldenrod1",  "deeppink4",   "gray19",
    "indigo",      "indianred",       "forestgreen", "navy",
    "orangered2",  "slateblue3",      "yellowgreen", "salmon",
    "purple",      "mediumturquoise", "slategray3"};

#define DEBUG_TYPE "task-dependency-graph"

TaskDependencyGraphPass::TaskDependencyGraphPass() : ModulePass(ID) {
  initializeTaskDependencyGraphPassPass(*PassRegistry::getPassRegistry());
}

TaskDependencyGraphPass::~TaskDependencyGraphPass() = default;

// Fill TaskFound dependency information
void TaskDependencyGraphData::obtainTaskInfo(TaskInfo &TaskFound,
                                             CallInst &TaskCallInst,
                                             DominatorTree &DT) {

  // Get dependency info from the task call, operand 4
  Instruction *DepStructCast =
      dyn_cast<Instruction>(TaskCallInst.getArgOperand(4));

  // Must be a bitcast
  if (BitCastInst *DepStructCastType = dyn_cast<BitCastInst>(DepStructCast)) {

    // Get the number of dependencies
    Instruction *DepStruct =
        dyn_cast<Instruction>(DepStructCastType->getOperand(0));
    auto Array =
        dyn_cast<ArrayType>(DepStruct->getType()->getPointerElementType());
    int TotalNumDeps = Array->getNumElements();

    // Vector to store the dependency info of each dep
    std::vector<std::vector<Value *>> DepInfo(
        TotalNumDeps,
        std::vector<Value *>(3)); // 0: Base Adress, 1 : Size , 2: Type of dep

    // Find instructions that target the dependency info struct, and store them
    for (User *U : DepStruct->users()) {
      if (GetElementPtrInst *GetDepField = dyn_cast<GetElementPtrInst>(U)) {

        int DepNum, DepField;
        DepNum = DepField = 0;

        if (ConstantInt *CI =
                dyn_cast<ConstantInt>(GetDepField->getOperand(2))) {
          if (CI->getBitWidth() <= 64) {
            DepNum = CI->getSExtValue();
          }
        }
        if (ConstantInt *CI =
                dyn_cast<ConstantInt>(GetDepField->getOperand(3))) {
          if (CI->getBitWidth() <= 32) {
            DepField = CI->getSExtValue();
          }
        }
        DepInfo[DepNum][DepField] = dyn_cast<Value>(U);
      }
    }

    // Vector to store task dep info
    SmallVector<TaskDependInfo, 2> AllTaskDepInfo;

    for (int i = 0; i < TotalNumDeps; i++) {
      // Look for the base
      TaskDependInfo CurrentTaskDepInfo;
      for (User *U : DepInfo[i][0]->users()) {

        Instruction *BaseUse = dyn_cast<Instruction>(U);
        Instruction *TaskCall = dyn_cast<Instruction>(&TaskCallInst);

        // Check that the next task call is the one we are looking for
        CallInst *SelectedTaskCall = nullptr;
        Instruction *NextIns = BaseUse->getNextNode();
        while (!SelectedTaskCall) {
          SelectedTaskCall = dyn_cast<CallInst>(NextIns);
          NextIns = NextIns->getNextNode();
        }
        if (SelectedTaskCall->getCalledFunction() &&
            SelectedTaskCall->getCalledFunction()->getName() ==
                "__kmpc_omp_task_with_deps") {
          if (SelectedTaskCall != TaskCall) {
            continue;
          }
        }

        if (StoreInst *BaseStore = dyn_cast<StoreInst>(U)) {
          if (PtrToIntOperator *BasePtrToInt =
                  dyn_cast<PtrToIntOperator>(BaseStore->getValueOperand())) {
            if (GEPOperator *GEP =
                    dyn_cast<GEPOperator>(BasePtrToInt->getPointerOperand())) {

              if (LoadInst *BaseLoad = dyn_cast<LoadInst>(GEP->getOperand(0))) {
                CurrentTaskDepInfo.base = BaseLoad->getPointerOperand();
              } else {
                CurrentTaskDepInfo.base = GEP->getOperand(0);
              }

              CurrentTaskDepInfo.isArray = true;
              for (int i = 1; i < (int)GEP->getNumOperands(); i++) {
                if (ConstantInt *CI =
                        dyn_cast<ConstantInt>(GEP->getOperand(i))) {
                  CurrentTaskDepInfo.index.push_back(CI->getZExtValue());
                } else {
                  dbgs() << "Not constant access, can not compute task "
                            "dependencies, check that loops are correctly "
                            "unrolled \n";
                }
              }
            } else {
              if (LoadInst *BaseLoad =
                      dyn_cast<LoadInst>(BasePtrToInt->getPointerOperand())) {

                CurrentTaskDepInfo.base = BaseLoad->getPointerOperand();
                CurrentTaskDepInfo.isArray = false;

              } else {
                // Store base
                CurrentTaskDepInfo.base = BasePtrToInt->getPointerOperand();
                CurrentTaskDepInfo.isArray = false;
              }
            }
          }
        }
      }
      // Store type of dep (in=1, out=2, inout=3)
      for (User *U : DepInfo[i][2]->users()) {
        Instruction *TypeUse = dyn_cast<Instruction>(U);
        Instruction *TaskCall = dyn_cast<Instruction>(&TaskCallInst);

        // Check that the next task call is the one we are looking for
        CallInst *SelectedTaskCall = nullptr;
        Instruction *NextIns = TypeUse->getNextNode();
        while (!SelectedTaskCall) {
          SelectedTaskCall = dyn_cast<CallInst>(NextIns);
          NextIns = NextIns->getNextNode();
        }
        if (SelectedTaskCall->getCalledFunction() &&
            SelectedTaskCall->getCalledFunction()->getName() ==
                "__kmpc_omp_task_with_deps") {
          if (SelectedTaskCall != TaskCall) {
            continue;
          }
        }

        if (StoreInst *TypeStore = dyn_cast<StoreInst>(U)) {
          if (ConstantInt *CI =
                  dyn_cast<ConstantInt>(TypeStore->getValueOperand())) {
            CurrentTaskDepInfo.type = CI->getZExtValue();
          } else {
            dbgs() << "Type is not constant, something is wrong\n";
          }
        }
      }
      AllTaskDepInfo.push_back(CurrentTaskDepInfo);
      TaskFound.TaskDepInfo = AllTaskDepInfo;
    }
  }
}

// Returns true if a dependency exists between two tasks, otherwise returns
// false
bool TaskDependencyGraphData::checkDependency(TaskDependInfo &Source,
                                              TaskDependInfo &Dest) {

  if (Source.base != Dest.base || Source.isArray != Dest.isArray ||
      Source.type == 1 || Dest.type == 2 ||
      Source.index.size() != Dest.index.size())
    return false;

  if (!Source.isArray && !Dest.isArray)
    return true;

  for (int i = 0; i < (int)Source.index.size(); i++) {
    if (Source.index[i] != Dest.index[i])
      return false;
  }
  return true;
}

// Depth First Search to look for transitive edges
void TaskDependencyGraphData::traverse_node(
    SmallVectorImpl<uint64_t> &edges_to_check, int node, int nesting_level,
    std::vector<bool> &Visited) {

  Visited[node] = true;

  for (int i = 0; i < (int)FunctionTasks[node].successors.size(); i++) {
    int successor = FunctionTasks[node].successors[i];
    for (int j = 0; j < (int)edges_to_check.size(); j++) {
      int edge = edges_to_check[j];
      if (edge == successor) {
        // Remove edge
        edges_to_check.erase(edges_to_check.begin() + j);
        break;
      }
    }
    if (Visited[successor] == false && nesting_level < MaxNesting)
      traverse_node(edges_to_check, successor, nesting_level + 1, Visited);
  }
}

void TaskDependencyGraphData::erase_transitive_edges() {
  for (int i = 0; i < (int)FunctionTasks.size(); i++) {

    // Skip tasks with no succesors
    if (!FunctionTasks[i].successors.size())
      continue;

    std::vector<bool> Visited(FunctionTasks.size());
    Visited[i] = true;

    for (int j = 0; j < (int)FunctionTasks[i].successors.size(); j++) {
      traverse_node(FunctionTasks[i].successors, FunctionTasks[i].successors[j],
                    0, Visited);
    }
  }
}
// Obtain the task identation string
void TaskDependencyGraphData::obtainTaskIdent(TaskInfo &TaskFound,
                                              CallInst &TaskCall) {
  GlobalVariable *IdentContainer =
      dyn_cast<GlobalVariable>(TaskCall.getArgOperand(0));
  Constant *InitValue = IdentContainer->getInitializer();
  GEPOperator *IdentGep =
      dyn_cast<GEPOperator>(InitValue->getAggregateElement(4));
  GlobalVariable *IdentString =
      dyn_cast<GlobalVariable>(IdentGep->getPointerOperand());
  ConstantDataArray *IdentStringArray =
      dyn_cast<ConstantDataArray>(IdentString->getInitializer());
  TaskFound.ident = IdentStringArray->getAsCString();
}

void TaskDependencyGraphData::print_tdg() {
  for (int i = 0; i < (int)FunctionTasks.size(); i++) {

    dbgs() << "TASK: " << FunctionTasks[i].id << " Successors: ";
    for (int j = 0; j < (int)FunctionTasks[i].successors.size(); j++) {
      dbgs() << " " << FunctionTasks[i].successors[j] << " ";
    }
    // printf(" Predecessors : %d ", RecordMap[i].npredecessors);
    dbgs() << " \n";
  }
}

void TaskDependencyGraphData::print_tdg_to_dot(StringRef ModuleName) {

  std::string fileName = ModuleName.str();
  size_t lastindex = fileName.find_last_of(".");
  std::string rawFileName = fileName.substr(0, lastindex);

  std::ofstream tdgfile(rawFileName + "_tdg.dot");

  if (!tdgfile.is_open()) {
    dbgs() << "Error Opening TDG file \n";
    exit(1);
  }

  tdgfile << "digraph TDG {\n";
  tdgfile << "   compound=true\n";
  tdgfile << "   subgraph cluster_0 {\n";
  tdgfile << "      label=TDG_0\n";

  for (auto &Task : FunctionTasks) {

    StringRef color = "";
    StringRef ident = Task.ident;

    for (auto &identColor : ColorMap) {
      if (identColor.ident == ident)
        color = identColor.color;
    }

    if (color.equals("")) {
      color = color_names[ColorsUsed];
      ColorMap.push_back({ident, color_names[ColorsUsed]});
      ColorsUsed++;
    }

    tdgfile << "      " << Task.id << "[color=" << color.str()
            << ",style=bold]\n";
  }
  tdgfile << "   }\n";

  for (int i = 0; i < (int)FunctionTasks.size(); i++) {

    int nsuccessors = FunctionTasks[i].successors.size();
    if (nsuccessors) {
      for (int j = 0; j < nsuccessors; j++) {
        tdgfile << "   " << FunctionTasks[i].id << " -> "
                << FunctionTasks[i].successors[j] << " \n";
      }
    } else {
      tdgfile << "   " << FunctionTasks[i].id << " \n";
    }
  }

  tdgfile << "   node [shape=plaintext];\n";
  tdgfile << "    subgraph cluster_1000 {\n";
  tdgfile << "      label=\"User functions:\"; style=\"rounded\";\n";
  tdgfile << " user_funcs [label=<<table border=\"0\" cellspacing=\"10\" "
             "cellborder=\"0\">\n";

  for (auto &identColor : ColorMap) {
    tdgfile << "      <tr>\n";
    tdgfile << "         <td bgcolor=\"" << identColor.color.str()
            << "\" width=\"15px\" border=\"1\"></td>\n";

    tdgfile << "         <td>" << identColor.ident.str() << "</td>\n";
    tdgfile << "      </tr>\n";
  }
  tdgfile << "      </table>>]\n";
  tdgfile << "}}\n";
  tdgfile.close();
}

void TaskDependencyGraphData::generate_tdg_file(StringRef ModuleName) {
  std::string fileName = ModuleName.str();
  size_t lastindex = fileName.find_last_of(".");
  std::string rawFileName = fileName.substr(0, lastindex);

  std::ofstream tdgfile(rawFileName + "_tdg.c");
  SmallVector<int, 10> InputList;
  SmallVector<int, 10> OutputList;
  for (int i = 0; i < (int)FunctionTasks.size(); i++) {
    InputList.insert(InputList.end(), FunctionTasks[i].predecessors.begin(),
                     FunctionTasks[i].predecessors.end());
    OutputList.insert(OutputList.end(), FunctionTasks[i].successors.begin(),
                      FunctionTasks[i].successors.end());
  }
  int offin = 0;
  int offout = 0;

  if (!tdgfile.is_open()) {
    dbgs() << "Error Opening TDG file \n";
    exit(1);
  }
  tdgfile << "struct kmp_task_t;\nstruct kmp_tdg\n{\n";
  tdgfile << "  unsigned int id;\n  struct kmp_task_t task;\n  unsigned int "
             "offin;\n  unsigned int offout;\n  "
             "unsigned int nin;\n  unsigned int nout;\n};\n";
  tdgfile << "struct kmp_tdg kmp_tdg_0[" << FunctionTasks.size() << "] = {";
  for (int i = 0; i < (int)FunctionTasks.size(); i++) {
    tdgfile << "{ .id =" << FunctionTasks[i].id
            << ", .task = 0, .offin =" << offin << ", .offout =" << offout
            << ", .nin =" << FunctionTasks[i].predecessors.size()
            << ", nout =" << FunctionTasks[i].successors.size() << "}";

    offin += FunctionTasks[i].predecessors.size();
    offout += FunctionTasks[i].successors.size();
    if (i != (int)FunctionTasks.size() - 1)
      tdgfile << ",";
    else
      tdgfile << "};\n";
  }
  tdgfile << "unsigned int kmp_tdg_ins_0[" << InputList.size() << "] = {";
  for (int i = 0; i < (int)InputList.size(); i++) {
    tdgfile << " " << InputList[i];
    if (i != (int)InputList.size() - 1)
      tdgfile << ",";
    else
      tdgfile << "};\n";
  }
  tdgfile << "unsigned int kmp_tdg_outs_0[" << OutputList.size() << "] = {";
  for (int i = 0; i < (int)OutputList.size(); i++) {
    tdgfile << " " << OutputList[i];
    if (i != (int)OutputList.size() - 1)
      tdgfile << ",";
    else
      tdgfile << "};\n";
  }
  tdgfile << "unsigned int gomp_tdg_ntasks = " << FunctionTasks.size() << ";\n";
  tdgfile.close();
}

void TaskDependencyGraphData::findOpenMPTasks(Function &F, DominatorTree &DT) {

  // Iterate over the BB
  SmallVector<BasicBlock *, 8> Worklist;
  SmallPtrSet<BasicBlock *, 8> Visited;
  Worklist.push_back(&F.getEntryBlock());
  Visited.insert(&F.getEntryBlock());

  // Incremental task id
  int NumTasks = 0;

  while (!Worklist.empty()) {
    auto WIt = Worklist.begin();
    BasicBlock *BB = *WIt;
    Worklist.erase(WIt);

    for (Instruction &I : *BB) {
      // Look for task with deps calls
      if (CallInst *II = dyn_cast<CallInst>(&I)) {
        if (II->getCalledFunction() &&
            II->getCalledFunction()->getName() == "__kmpc_omp_task_with_deps") {
          TaskInfo TaskFound;
          TaskFound.id = NumTasks;
          NumTasks++;
          // Fill Task Deps Info
          obtainTaskInfo(TaskFound, *II, DT);
          obtainTaskIdent(TaskFound, *II);
          // Store Task Info
          FunctionTasks.push_back(TaskFound);
        }
      }
      // Look for task without deps

      if (CallInst *II = dyn_cast<CallInst>(&I)) {
        if (II->getCalledFunction() &&
            II->getCalledFunction()->getName() == "__kmpc_omp_task") {
          TaskInfo TaskFound;
          TaskFound.id = NumTasks;
          NumTasks++;
          // Fill Task Deps Info
          obtainTaskIdent(TaskFound, *II);
          // Store Task Info
          FunctionTasks.push_back(TaskFound);
        }
      }
    }
    // Do not revisite BB
    for (auto It = succ_begin(BB); It != succ_end(BB); ++It) {
      if (!Visited.count(*It)) {
        Worklist.push_back(*It);
        Visited.insert(*It);
      }
    }
  }

  // Check dependencies between all the tasks found, and fill
  // succesors/predecessors
  for (int i = 0; i < (int)FunctionTasks.size(); i++) {
    for (int j = i + 1; j < (int)FunctionTasks.size(); j++) {
      for (auto FirstTaskDepInfo : FunctionTasks[i].TaskDepInfo) {
        for (auto SecondTaskDepInfo : FunctionTasks[j].TaskDepInfo) {
          if (checkDependency(FirstTaskDepInfo, SecondTaskDepInfo)) {
            FunctionTasks[i].successors.push_back(FunctionTasks[j].id);
            FunctionTasks[j].predecessors.push_back(FunctionTasks[i].id);
            break;
          }
        }
      }
    }
  }

  // Remove transisitve edges
  erase_transitive_edges();

  print_tdg();
  if (FunctionTasks.size()) {
    print_tdg_to_dot(F.getParent()->getSourceFileName());
    generate_tdg_file(F.getParent()->getSourceFileName());
  }
  FunctionTasks.clear();
}

bool TaskDependencyGraphPass::runOnModule(Module &M) {
  TaskDependencyGraphData TDG;

  for (Function &F : M) {
    if (F.isDeclaration() || F.empty())
      continue;
    // Only check functions with tasks
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    //if (F.hasFnAttribute("llvm.omp.taskgraph"))
      TDG.findOpenMPTasks(F, DT);
  }
  return false;
}

void TaskDependencyGraphPass::releaseMemory() {}

void TaskDependencyGraphPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<DominatorTreeWrapperPass>();
}

void TaskDependencyGraphPass::print(raw_ostream &OS, const Module *) const {}

char TaskDependencyGraphPass::ID = 0;

INITIALIZE_PASS_BEGIN(TaskDependencyGraphPass, "task-dependency-graph",
                      "Generate static task dependency graph", true, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(TaskDependencyGraphPass, "task-dependency-graph",
                    "Generate static task dependency graph", true, true)

// Create methods available outside of this file, to use them
// "include/llvm/LinkAllPasses.h". Otherwise the pass would be deleted by
// the link time optimization.

ModulePass *llvm::createTaskDependencyGraphPass() {
  return new TaskDependencyGraphPass();
}

// New PM pass
AnalysisKey TaskDependencyGraphAnalysis::Key;

TaskDependencyGraphData
TaskDependencyGraphAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  for (Function &F : M) {
    if (F.isDeclaration() || F.empty())
      continue;
    auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
    // Only check functions with tasks
    //if (F.hasFnAttribute("llvm.omp.taskgraph"))
      TDG.findOpenMPTasks(F, DT);
  }
  return TDG;
}

PreservedAnalyses
TaskDependencyGraphAnalysisPass::run(Module &M, ModuleAnalysisManager &AM) {

  AM.getResult<TaskDependencyGraphAnalysis>(M);
  // Analysis should never change the LLVM IR code so all
  // results of other analyses are still valid!
  return PreservedAnalyses::all();
}
