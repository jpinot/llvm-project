//===- TaskDependencyGraph.cpp - Generation of a static OpenMP task dependency
//graph
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Generates an static OpenMP task dependency graph.
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

using namespace llvm;
const StringRef Color_names[] = {
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
                  llvm_unreachable(
                      "not constant access, can not compute task "
                      "dependencies, check that loops are correctly "
                      "unrolled \n");
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
            llvm_unreachable("type is not constant, something is wrong\n");
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
    SmallVectorImpl<uint64_t> &Edges_to_check, int Node, int Master,
    int Nesting_level, std::vector<bool> &Visited) {

  Visited[Node] = true;

  for (int i = 0; i < (int)FunctionTasks[Node].successors.size(); i++) {
    int Successor = FunctionTasks[Node].successors[i];
    for (int j = 0; j < (int)Edges_to_check.size(); j++) {
      int edge = Edges_to_check[j];
      if (edge == Successor) {
        // Remove edge
        Edges_to_check.erase(Edges_to_check.begin() + j);
        for (int x = 0; x < (int)FunctionTasks[edge].predecessors.size(); x++) {
          if ((int)FunctionTasks[edge].predecessors[x] == Master) {
            FunctionTasks[edge].predecessors.erase(
                FunctionTasks[edge].predecessors.begin() + x);
            break;
          }
        }
        break;
      }
    }
    if (Visited[Successor] == false && Nesting_level < MaxNesting)
      traverse_node(Edges_to_check, Successor, Master, Nesting_level + 1,
                    Visited);
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
                    i, 0, Visited);
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

  // std::string fileName = ModuleName.str();
  // size_t lastindex = fileName.find_last_of(".");
  // std::string rawFileName = fileName.substr(0, lastindex);
  std::error_code EC;
  llvm::raw_fd_ostream Tdgfile("tdg.dot", EC);

  if (!Tdgfile.has_error()) {
    llvm_unreachable("Error Opening TDG file \n");
  }

  Tdgfile << "digraph TDG {\n";
  Tdgfile << "   compound=true\n";
  Tdgfile << "   subgraph cluster_0 {\n";
  Tdgfile << "      label=TDG_0\n";

  for (auto &Task : FunctionTasks) {

    StringRef Color = "";
    StringRef Ident = Task.ident;

    for (auto &identColor : ColorMap) {
      if (identColor.ident == Ident)
        Color = identColor.color;
    }

    if (Color.equals("")) {
      Color = Color_names[ColorsUsed];
      ColorMap.push_back({Ident, Color_names[ColorsUsed]});
      size_t Ncolors = sizeof(Color_names) / sizeof(Color_names[0]);
      ColorsUsed = (ColorsUsed + 1) % Ncolors;
    }

    Tdgfile << "      " << Task.id << "[color=" << Color.str()
            << ",style=bold]\n";
  }
  Tdgfile << "   }\n";

  for (int i = 0; i < (int)FunctionTasks.size(); i++) {

    int nsuccessors = FunctionTasks[i].successors.size();
    if (nsuccessors) {
      for (int j = 0; j < nsuccessors; j++) {
        Tdgfile << "   " << FunctionTasks[i].id << " -> "
                << FunctionTasks[i].successors[j] << " \n";
      }
    } else {
      Tdgfile << "   " << FunctionTasks[i].id << " \n";
    }
  }

  Tdgfile << "   node [shape=plaintext];\n";
  Tdgfile << "    subgraph cluster_1000 {\n";
  Tdgfile << "      label=\"User functions:\"; style=\"rounded\";\n";
  Tdgfile << " user_funcs [label=<<table border=\"0\" cellspacing=\"10\" "
             "cellborder=\"0\">\n";

  for (auto &identColor : ColorMap) {
    Tdgfile << "      <tr>\n";
    Tdgfile << "         <td bgcolor=\"" << identColor.color.str()
            << "\" width=\"15px\" border=\"1\"></td>\n";

    Tdgfile << "         <td>" << identColor.ident.str() << "</td>\n";
    Tdgfile << "      </tr>\n";
  }
  Tdgfile << "      </table>>]\n";
  Tdgfile << "}}\n";
  Tdgfile.close();
}

void TaskDependencyGraphData::generate_analysis_tdg_file(StringRef ModuleName) {
  /*
  std::string fileName = ModuleName.str();
  size_t lastindex = fileName.find_last_of(".");
  std::string rawFileName = fileName.substr(0, lastindex);
  */
  std::error_code EC;
  llvm::raw_fd_ostream Tdgfile("analysis_tdg.c", EC);

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

  if (!Tdgfile.has_error()) {
    llvm_unreachable("Error Opening TDG file \n");
  }
  Tdgfile << "struct tdg\n{\n";
  Tdgfile << "  unsigned int id;\n  unsigned int "
             "offin;\n  unsigned int offout;\n  "
             "unsigned int nin;\n  unsigned int nout;\n};\n";
  Tdgfile << "struct tdg tdg_tasks[" << FunctionTasks.size() << "] = {";
  for (int i = 0; i < (int)FunctionTasks.size(); i++) {
    Tdgfile << "{ .id =" << FunctionTasks[i].id << ", .offin =" << offin
            << ", .offout =" << offout
            << ", .nin =" << FunctionTasks[i].predecessors.size()
            << ", nout =" << FunctionTasks[i].successors.size() << "}";

    offin += FunctionTasks[i].predecessors.size();
    offout += FunctionTasks[i].successors.size();
    if (i != (int)FunctionTasks.size() - 1)
      Tdgfile << ",\n";
    else
      Tdgfile << "};\n";
  }
  Tdgfile << "unsigned int tdg_ins[" << InputList.size() << "] = {";
  for (int i = 0; i < (int)InputList.size(); i++) {
    Tdgfile << " " << InputList[i];
    if (i != (int)InputList.size() - 1)
      Tdgfile << ",";
    else
      Tdgfile << "};\n";
  }
  Tdgfile << "unsigned int tdg_outs[" << OutputList.size() << "] = {";
  for (int i = 0; i < (int)OutputList.size(); i++) {
    Tdgfile << " " << OutputList[i];
    if (i != (int)OutputList.size() - 1)
      Tdgfile << ",";
    else
      Tdgfile << "};\n";
  }
  Tdgfile << "unsigned int tdg_ntasks = " << FunctionTasks.size() << ";\n";
  Tdgfile.close();
}

void TaskDependencyGraphData::generate_runtime_tdg_file(StringRef ModuleName) {
  /*
  std::string fileName = ModuleName.str();
  size_t lastindex = fileName.find_last_of(".");
  std::string rawFileName = fileName.substr(0, lastindex);
  */
  std::error_code EC;
  llvm::raw_fd_ostream Tdgfile("tdg.c", EC);

  if (!Tdgfile.has_error()) {
    llvm_unreachable("Error Opening TDG file \n");
  }

  SmallVector<int, 10> OutputList;
  SmallVector<int, 2> TdgRoots;
  for (int i = 0; i < (int)FunctionTasks.size(); i++) {
    if (FunctionTasks[i].predecessors.size() == 0)
      TdgRoots.push_back(i);

    OutputList.insert(OutputList.end(), FunctionTasks[i].successors.begin(),
                      FunctionTasks[i].successors.end());
  }

  int offout = 0;
  Tdgfile << "#include <stddef.h>\n";
  Tdgfile << "struct kmp_task_t;\nstruct kmp_record_info\n{\n";
  Tdgfile << "  int static_id;\n  struct kmp_task_t *task;\n  int "
             "* succesors;\n  int nsuccessors;\n  "
             "int npredecessors_counter;\n  int npredecessors;\n  int "
             "successors_size;\n};\n";
  Tdgfile << "extern void __kmpc_set_tdg(struct kmp_record_info *tdg, int "
             "ntasks, int *roots, int nroots);\n";
  Tdgfile << "int kmp_tdg_outs_0[" << OutputList.size() << "] = {";
  if (OutputList.size()) {
    for (int i = 0; i < (int)OutputList.size(); i++) {
      Tdgfile << " " << OutputList[i];
      if (i != (int)OutputList.size() - 1)
        Tdgfile << ",";
      else
        Tdgfile << "};\n";
    }
  } else {
    Tdgfile << "};\n";
  }
  Tdgfile << "struct kmp_record_info kmp_tdg_0[" << FunctionTasks.size()
          << "] = {";
  for (int i = 0; i < (int)FunctionTasks.size(); i++) {
    Tdgfile << "{ .static_id =" << FunctionTasks[i].id
            << ", .task = NULL, .succesors = &kmp_tdg_outs_0[" << offout << "]"
            << ", .nsuccessors =" << FunctionTasks[i].successors.size()
            << ", .npredecessors_counter ="
            << FunctionTasks[i].predecessors.size()
            << ", .npredecessors = " << FunctionTasks[i].predecessors.size()
            << ", .successors_size = 0"
            << "}";

    offout += FunctionTasks[i].successors.size();
    if (i != (int)FunctionTasks.size() - 1)
      Tdgfile << ",\n";
    else
      Tdgfile << "};\n";
  }
  Tdgfile << "int kmp_tdg_roots[" << TdgRoots.size() << "] = {";
  for (int i = 0; i < (int)TdgRoots.size(); i++) {
    Tdgfile << " " << TdgRoots[i];
    if (i != (int)TdgRoots.size() - 1)
      Tdgfile << ",";
    else
      Tdgfile << "};\n";
  }
  Tdgfile << "void kmp_set_tdg()\n{\n  __kmpc_set_tdg(kmp_tdg_0, "
          << FunctionTasks.size() << ", kmp_tdg_roots, " << TdgRoots.size()
          << ");\n};";
  Tdgfile.close();
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
          ++NumTasks;
          // Fill Task Deps and Ident info
          obtainTaskInfo(TaskFound, *II, DT);
          obtainTaskIdent(TaskFound, *II);
          // Store Task info
          FunctionTasks.push_back(TaskFound);
        }
      }
      // Look for task without deps
      if (CallInst *II = dyn_cast<CallInst>(&I)) {
        if (II->getCalledFunction() &&
            II->getCalledFunction()->getName() == "__kmpc_omp_task") {
          TaskInfo TaskFound;
          TaskFound.id = NumTasks;
          ++NumTasks;
          // Fill Task Ident info
          obtainTaskIdent(TaskFound, *II);
          // Store Task info
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
        bool DepExists = false;
        for (auto SecondTaskDepInfo : FunctionTasks[j].TaskDepInfo) {
          if (checkDependency(FirstTaskDepInfo, SecondTaskDepInfo)) {
            FunctionTasks[i].successors.push_back(FunctionTasks[j].id);
            FunctionTasks[j].predecessors.push_back(FunctionTasks[i].id);
            DepExists = true;
            break;
          }
        }
        if (DepExists)
          break;
      }
    }
  }

  // Remove transisitve edges
  erase_transitive_edges();

  // print_tdg();
  if (FunctionTasks.size()) {
    print_tdg_to_dot(F.getParent()->getSourceFileName());
    generate_analysis_tdg_file(F.getParent()->getSourceFileName());
    generate_runtime_tdg_file(F.getParent()->getSourceFileName());
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
    if (F.hasFnAttribute("llvm.omp.taskgraph.static"))
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
    if (F.hasFnAttribute("llvm.omp.taskgraph.static"))
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
