//===- TaskDependencyGraph.cpp - Generation of a static OpenMP task dependency
// graph
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

  if (Tdgfile.has_error()) {
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

  if (Tdgfile.has_error()) {
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

std::string
TaskDependencyGraphData::get_task_layout(std::string LongestPrivateName,
                                         std::string LongestSharedName) {
  // Define routine entry
  std::string Result =
      "typedef int32_t (*kmp_routine_entry_t)(int32_t, void *);\n\n";

  Result += "struct kmp_space_indexer_node {\n"
            "  void *task;\n"
            "  void *next;\n"
            "};\n\n";
  // Define kmp_cmplrdata
  Result += "typedef union kmp_cmplrdata {\n"
            "  int32_t priority;\n"
            "  kmp_routine_entry_t destructors;\n"
            "} kmp_cmplrdata_t;\n\n";

  // Define kmp_r_sched_t
  Result += "typedef union kmp_r_sched {\n"
            "  struct {\n"
            "    int32_t r_sched_type;\n"
            "    int chunk;\n"
            "  };\n"
            "  int64_t sched;\n"
            "} kmp_r_sched_t;\n\n";

  // Define kmp_proc_bind_t
  Result += "typedef enum kmp_proc_bind_t {\n"
            "  proc_bind_false = 0,\n"
            "  proc_bind_true,\n"
            "  proc_bind_master,\n"
            "  proc_bind_close,\n"
            "  proc_bind_spread,\n"
            "  proc_bind_intel,\n"
            "  proc_bind_default\n"
            "} kmp_proc_bind_t;\n\n";

  // Define internal control
  Result += "typedef struct kmp_internal_control {\n"
            "  int serial_nesting_level; \n"
            "  int8_t dynamic; \n"
            "  int8_t bt_set; \n"
            "  int blocktime;\n"
            "  int nproc;\n"
            "  int thread_limit;\n"
            "  int max_active_levels;\n"
            "  kmp_r_sched_t sched; \n"
            "  kmp_proc_bind_t proc_bind; \n"
            "  int32_t default_device;\n"
            "  void *next;\n"
            "} kmp_internal_control_t;\n\n";

  // Define kmp_event_t
  Result += "typedef struct {\n"
            "  int32_t pending_events_count;\n"
            "  union {\n"
            "    void *task;\n"
            "  } ed;\n"
            "} kmp_event_t;\n\n";

  // Define ompt_data_t
  Result += "typedef union ompt_data_t {\n"
            "  uint64_t value;\n"
            "  void *ptr;\n"
            "} ompt_data_t;\n\n";

  // Define ompt_frame_t
  Result += "typedef struct ompt_frame_t {\n"
            "  ompt_data_t exit_frame;\n"
            "  ompt_data_t enter_frame;\n"
            "  int exit_frame_flags;\n"
            "  int enter_frame_flags;\n"
            "} ompt_frame_t;\n\n";

  // Define ompt_task_info_t
  Result += "typedef struct {\n"
            "  ompt_frame_t frame;\n"
            "  ompt_data_t task_data;\n"
            "  void *scheduling_parent;\n"
            "  int thread_num;\n"
            "} ompt_task_info_t;\n\n";

  // Define taskdata
  Result += "struct kmp_task {\n"
            "  int32_t td_task_id;\n"
            "  int32_t td_flags;\n"
            "  void *td_team;\n"
            "  void *td_alloc_thread;\n"
            "  void *td_parent; \n"
            "  int32_t td_level;\n"
            "  std::atomic<int32_t> td_untied_count;\n"
            "  void *td_ident;\n"
            "  void *td_taskwait_ident;\n"
            "  uint32_t td_taskwait_counter;\n"
            "  int32_t td_taskwait_thread;\n"
            "  __attribute__((aligned(64))) kmp_internal_control_t td_icvs;\n"
            "  __attribute__((aligned(64))) std::atomic<int32_t> "
            "td_allocated_child_tasks;\n"
            "  std::atomic<int32_t> td_incomplete_child_tasks;\n"
            "  void *td_taskgroup;\n"
            "  void *td_dephash;\n"
            "  void *td_depnode;\n"
            "  void *td_task_team;\n"
            "  size_t td_size_alloc;\n"
            "  int32_t td_size_loop_bounds;\n"
            "  void *td_last_tied;\n"
            "  void (*td_copy_func)(void *, void *);\n"
            "  kmp_event_t td_allow_completion_event;\n"
            "  ompt_task_info_t ompt_task_info; \n"
            "  int is_taskgraph = 0;\n"
            "  void *indexer_node;\n";

  // Define shared data
  Result += "  struct " + LongestSharedName + " shared_data;\n";

  // Define task structure
  Result += "  void *shareds; \n"
            "  kmp_routine_entry_t routine;\n"
            "  int32_t part_id;\n"
            "  kmp_cmplrdata_t data1;\n"
            "  kmp_cmplrdata_t data2;\n";

  // Define private data
  Result += "  struct " + LongestPrivateName + " private_data;\n";
  Result += "  struct " + LongestPrivateName + " private_data_2;\n};\n";

  return Result;
}

std::string TaskDependencyGraphData::get_c_struct_from_types(
    SmallVectorImpl<Type *> &types, int pragma_id, bool is_private) {
  std::string Result;

  if (is_private)
    Result += "struct private_data_" + std::to_string(pragma_id) + "{\n";
  else
    Result += "struct shared_data_" + std::to_string(pragma_id) + "{\n";
  int num_members = 0;
  for (Type *this_type : types) {
    switch (this_type->getTypeID()) {
    case Type::HalfTyID:
      Result += "  short r_" + std::to_string(num_members) + ";\n";
      num_members++;
      break;

    case Type::BFloatTyID:
      Result += "  short r_" + std::to_string(num_members) + ";\n";
      num_members++;
      break;

    case Type::FloatTyID:
      Result += "  float r_" + std::to_string(num_members) + ";\n";
      num_members++;
      break;

    case Type::DoubleTyID:
      Result += "  double r_" + std::to_string(num_members) + ";\n";
      num_members++;
      break;

    case Type::IntegerTyID:
      Result += "  int" + std::to_string(this_type->getIntegerBitWidth()) +
                "_t r_" + std::to_string(num_members) + ";\n";
      num_members++;
      break;

    case Type::PointerTyID:
      Result += "  void * r_" + std::to_string(num_members) + ";\n";
      num_members++;
      break;

    default:
      dbgs() << "Data type not implemented or recognized "
             << this_type->getTypeID() << "\n";
      break;
    }
  }
  Result += "};\n";
  return Result;
}

void TaskDependencyGraphData::generate_runtime_tdg_file(StringRef ModuleName) {
  /*
  std::string fileName = ModuleName.str();
  size_t lastindex = fileName.find_last_of(".");
  std::string rawFileName = fileName.substr(0, lastindex);
  */
  std::error_code EC;
  llvm::raw_fd_ostream Tdgfile("tdg.cpp", EC);

  if (Tdgfile.has_error()) {
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
  Tdgfile << "#include <atomic>\n";
  if (Prealloc) {
    Tdgfile << "#include <stdint.h>\n";
    Tdgfile << "#include <stdio.h>\n";
  }

  Tdgfile << "struct kmp_task_t;\nstruct kmp_record_info\n{\n";
  Tdgfile << "  int static_id;\n  struct kmp_task_t *task;\n  int "
             "* succesors;\n  int nsuccessors;\n  "
             "std::atomic<int> npredecessors_counter;\n  int npredecessors;\n  int "
             "successors_size;\n  int static_thread;\n  int pragma_id;\n  void "
             "* private_data;\n  "
             "void * shared_data;\n  void * parent_task;\n  struct "
             "kmp_record_info * next_waiting_tdg;\n};\n";

  Tdgfile
      << "extern  \"C\" void __kmpc_set_tdg(struct kmp_record_info *tdg, int "
         "ntasks, int *roots, int nroots);\n";
  if (Prealloc) {
    for (int i = 0; i < (int)TasksAllocInfo.size(); i++) {
      Tdgfile << "extern  \"C\" int " << TasksAllocInfo[i].entryPoint->getName()
              << "(int, void *);\n";
    }
    Tdgfile << "struct kmp_task_alloc_info\n{\n";
    Tdgfile << "  int flags;\n  int sizeOfTask;\n  int sizeOfShareds;\n  void* "
               "taskEntry;\n  int *sharedDataPositions;\n  int *firstPrivateDataPositions;\n  int *firstPrivateDataOffsets;\n  int *firstPrivateDataSizes;\n  int numFirstPrivates;\n};\n";
    Tdgfile << "extern  \"C\"  void  __kmpc_prealloc_tasks(struct "
               "kmp_task_alloc_info *task_static_data, char "
               "*preallocated_tasks, void *preallocated_nodes, unsigned int "
               "n_task_constructs,unsigned int "
               "max_concurrent_tasks, unsigned int task_size);\n";
    int longest_pragma_size = 0;
    int longest_pragma;
    for (int i = 0; i < (int)TasksAllocInfo.size(); i++) {
      int pragma_size =
          TasksAllocInfo[i].sizeOfTask + TasksAllocInfo[i].sizeOfShareds;
      if (pragma_size > longest_pragma_size) {
        longest_pragma_size = pragma_size;
        longest_pragma = i;
      }
      Tdgfile << get_c_struct_from_types(TasksAllocInfo[i].privatesType,
                                         FunctionTasks[i].id, true);
      Tdgfile << get_c_struct_from_types(TasksAllocInfo[i].sharedsType,
                                         FunctionTasks[i].id, false);
    }
    for (int i = 0; i < (int)FunctionTasks.size(); i++) {
      Tdgfile << "struct private_data_" << FunctionTasks[i].pragmaId << " task_"
              << FunctionTasks[i].id << "_private_data={";
      for (int j = 0; j < (int)FunctionTasks[i].FirstPrivateData.size(); j++) {
        if(TasksAllocInfo[FunctionTasks[i].pragmaId].privatesType[j]->isPointerTy()){
          Tdgfile << "(void *) ";
        }
        Tdgfile << FunctionTasks[i].FirstPrivateData[j];
        if (j != (int)FunctionTasks[i].FirstPrivateData.size() - 1)
          Tdgfile << ", ";
        else
          Tdgfile << "};\n";
      }
      if(!FunctionTasks[i].FirstPrivateData.size())
         Tdgfile << "};\n";
      Tdgfile << "struct shared_data_" << FunctionTasks[i].pragmaId << " task_"
              << FunctionTasks[i].id << "_shared_data;\n";
    }
    Tdgfile << get_task_layout("private_data_" + std::to_string(longest_pragma),
                               "shared_data_" + std::to_string(longest_pragma));
    Tdgfile << "struct kmp_task preallocated_tasks[" << NumPreallocs << "];\n";
    Tdgfile << "struct kmp_space_indexer_node preallocated_nodes["
            << NumPreallocs << "];\n\n";
  }

  Tdgfile << "int kmp_tdg_outs_0[" << OutputList.size() << "] = {";
  if (OutputList.size()) {
    for (int i = 0; i < (int)OutputList.size(); i++) {
      Tdgfile << OutputList[i];
      if (i != (int)OutputList.size() - 1)
        Tdgfile << ", ";
      else
        Tdgfile << "};\n";
    }
  } else {
    Tdgfile << "};\n";
  }
  Tdgfile << "struct kmp_record_info kmp_tdg_0[" << FunctionTasks.size()
          << "] = {";
  for (int i = 0; i < (int)FunctionTasks.size(); i++) {
    Tdgfile << "{ .static_id = " << FunctionTasks[i].id
            << ", .task = NULL, .succesors = &kmp_tdg_outs_0[" << offout << "]"
            << ", .nsuccessors = " << FunctionTasks[i].successors.size()
            << ", .npredecessors_counter = {"
            << FunctionTasks[i].predecessors.size()
            << "}, .npredecessors = " << FunctionTasks[i].predecessors.size()
            << ", .successors_size = 0"
            << ", .static_thread = -1"
            << ", .pragma_id = " << FunctionTasks[i].pragmaId;
    if (Prealloc)
      Tdgfile << ", .private_data = &task_" << FunctionTasks[i].id
              << "_private_data"
              << ", .shared_data = &task_" << FunctionTasks[i].id
              << "_shared_data";
    else
      Tdgfile << ", .private_data = NULL"
              << ", .shared_data = NULL";

    Tdgfile << ", .parent_task = NULL"
            << ", .next_waiting_tdg = NULL"
            << "}";

    offout += FunctionTasks[i].successors.size();
    if (i != (int)FunctionTasks.size() - 1)
      Tdgfile << ",\n";
    else
      Tdgfile << "};\n";
  }
  Tdgfile << "int kmp_tdg_roots[" << TdgRoots.size() << "] = {";
  for (int i = 0; i < (int)TdgRoots.size(); i++) {
    Tdgfile << TdgRoots[i];
    if (i != (int)TdgRoots.size() - 1)
      Tdgfile << ", ";
    else
      Tdgfile << "};\n";
  }
  if (Prealloc) {
    /*
    Tdgfile << "int kmp_pragma_ids[" << FunctionTasks.size() << "] = {";
    for (int i = 0; i < (int)FunctionTasks.size(); i++) {

      Tdgfile << FunctionTasks[i].pragmaId;

      if (i != (int)FunctionTasks.size() - 1)
        Tdgfile << ", ";
      else
        Tdgfile << "};\n";

    Tdgfile << ""
    }
    */
    for (int i = 0; i < (int)TasksAllocInfo.size(); i++) {
      Tdgfile << "int shared_data_positions_" << i << "[] = {";
      for (int j = 0; j < (int)TasksAllocInfo[i].sharedDataPositions.size(); j++){
        Tdgfile << TasksAllocInfo[i].sharedDataPositions[j];
        if (j != (int)TasksAllocInfo[i].sharedDataPositions.size() - 1)
          Tdgfile << ", ";
        else
          Tdgfile << "};\n";
      }
      if(!TasksAllocInfo[i].sharedDataPositions.size())
        Tdgfile << "};\n";

      Tdgfile << "int firstprivate_data_positions_" << i << "[] = {";
      for (int j = 0; j < (int)TasksAllocInfo[i].firstPrivateDataPositions.size(); j++){
        Tdgfile << TasksAllocInfo[i].firstPrivateDataPositions[j];
        if (j != (int)TasksAllocInfo[i].firstPrivateDataPositions.size() - 1)
          Tdgfile << ", ";
        else
          Tdgfile << "};\n";
      }
      if(!TasksAllocInfo[i].firstPrivateDataPositions.size())
        Tdgfile << "};\n";

      Tdgfile << "int firstprivate_data_offsets_" << i << "[] = {";
      for (int j = 0; j < (int)TasksAllocInfo[i].firstPrivateDataOffsets.size(); j++){
        Tdgfile << TasksAllocInfo[i].firstPrivateDataOffsets[j];
        if (j != (int)TasksAllocInfo[i].firstPrivateDataOffsets.size() - 1)
          Tdgfile << ", ";
        else
          Tdgfile << "};\n";
      }
      if(!TasksAllocInfo[i].firstPrivateDataOffsets.size())
        Tdgfile << "};\n";

      Tdgfile << "int firstprivate_data_sizes_" << i << "[] = {";
      for (int j = 0; j < (int)TasksAllocInfo[i].firstPrivateDataSizes.size(); j++){
        Tdgfile << TasksAllocInfo[i].firstPrivateDataSizes[j];
        if (j != (int)TasksAllocInfo[i].firstPrivateDataSizes.size() - 1)
          Tdgfile << ", ";
        else
          Tdgfile << "};\n";
      }
      if(!TasksAllocInfo[i].firstPrivateDataSizes.size())
        Tdgfile << "};\n";


    }
    Tdgfile << "struct kmp_task_alloc_info task_static_data["
            << TasksAllocInfo.size() << "] = {";
    for (int i = 0; i < (int)TasksAllocInfo.size(); i++) {

      Tdgfile << "{ .flags = " << TasksAllocInfo[i].flags
              << ", .sizeOfTask = " << TasksAllocInfo[i].sizeOfTask
              << ", .sizeOfShareds = " << TasksAllocInfo[i].sizeOfShareds
              << ", .taskEntry = (void *) &"
              << TasksAllocInfo[i].entryPoint->getName() << ", .sharedDataPositions = (int *) &shared_data_positions_" << i 
              << ", .firstPrivateDataPositions = (int *) &firstprivate_data_positions_" << i
              << ", .firstPrivateDataOffsets = (int *) &firstprivate_data_offsets_" << i
              << ", .firstPrivateDataSizes = (int *) & firstprivate_data_sizes_" << i
              << ", .numFirstPrivates = " << TasksAllocInfo[i].firstPrivateDataSizes.size() << "}";

      if (i != (int)TasksAllocInfo.size() - 1)
        Tdgfile << ",\n";
      else
        Tdgfile << "};\n";

    }
  }

  Tdgfile << "extern \"C\" void kmp_set_tdg(int num_preallocs)\n{\n";
  if (Prealloc) {
    // Tdgfile << "printf(\" es: %d \", sizeof(struct kmp_task));\n";
    Tdgfile << "  __kmpc_prealloc_tasks(task_static_data, (char *) preallocated_tasks, "
               "preallocated_nodes, "
            << FunctionTasks.size()
            << ", num_preallocs, sizeof(struct kmp_task));\n";
  }
  Tdgfile << "  __kmpc_set_tdg(kmp_tdg_0, " << FunctionTasks.size()
          << ", kmp_tdg_roots, " << TdgRoots.size() << ");\n}";

  Tdgfile.close();
}

int getTypeSizeInBytes(Type *this_type){
   int result=0;
   switch (this_type->getTypeID()) {
    case Type::HalfTyID:
      result= sizeof(short);
      break;

    case Type::BFloatTyID:
      result= sizeof(float);
      break;

    case Type::FloatTyID:
      result= sizeof(float);
      break;

    case Type::DoubleTyID:
      result= sizeof(double);
      break;

    case Type::IntegerTyID:
      result= (this_type->getIntegerBitWidth())/8;
      break;

    case Type::PointerTyID:
      result= sizeof(void *);
      break;

    default:
      dbgs() << "Data type not implemented or recognized "
             << this_type->getTypeID() << "\n";
      break;
    }
    return result;
}

int TaskDependencyGraphData::findPragmaId(CallInst &TaskCallInst,
                                          TaskInfo &TaskFound, Function &F) {

  // Get task alloc from the task call, operand 2
  CallInst *TaskAlloc = dyn_cast<CallInst>(TaskCallInst.getArgOperand(2));

  int Flags = 0, SizeOfTask = 0, SizeOfShareds = 0;

  // Get flags from task alloc, operand 2
  if (ConstantInt *CI = dyn_cast<ConstantInt>(TaskAlloc->getArgOperand(2))) {
    if (CI->getBitWidth() <= 64) {
      Flags = CI->getSExtValue();
    }
  }

  // Get flags from task alloc, operand 3
  if (ConstantInt *CI = dyn_cast<ConstantInt>(TaskAlloc->getArgOperand(3))) {
    if (CI->getBitWidth() <= 64) {
      SizeOfTask = CI->getSExtValue();
    }
  }

  // Get flags from task alloc, operand 4
  if (ConstantInt *CI = dyn_cast<ConstantInt>(TaskAlloc->getArgOperand(4))) {
    if (CI->getBitWidth() <= 64) {
      SizeOfShareds = CI->getSExtValue();
    }
  }

  // Get entry point from task alloc, operand 5
  Value *EntryPoint = nullptr;

  if (auto BitC = dyn_cast<BitCastOperator>(TaskAlloc->getArgOperand(5))) {
    EntryPoint = BitC->getOperand(0);
  }

  // Get private data types
  SmallVector<Type *, 2> TaskPrivatesType;
  Function *EntryPointFunction = dyn_cast<Function>(EntryPoint);
  Type *TaskWithPrivatesType =
      ((EntryPointFunction->getArg(1))->getType())->getPointerElementType();
  if (TaskWithPrivatesType->getStructNumElements() > 1) {
    Type *PrivatesType = TaskWithPrivatesType->getStructElementType(1);
    for (int j = 0; j < (int)PrivatesType->getStructNumElements(); j++)
      TaskPrivatesType.push_back(PrivatesType->getStructElementType(j));
  }

  // Look for constant values for the private data
  SmallVector<int64_t> PrivateValues;
  SmallVector<int, 2> FinalFirstPrivateOffsets;
  SmallVector<int, 2> FirstPrivateSizes;
  SmallVector<int, 2> AllPositions;

  Instruction *Start = dyn_cast<Instruction>(TaskAlloc);
  Instruction *End = dyn_cast<Instruction>(&TaskCallInst);
  while (Start != End) {
    for (Value *TaskAllocUser : TaskAlloc->users()) {
      if (Start == dyn_cast<Instruction>(TaskAllocUser)) {
        if (auto *ThisGEP = dyn_cast<GEPOperator>(TaskAllocUser)) {
          Value *ValSize = ThisGEP->getOperand(1);
          int Size = (int) dyn_cast<ConstantInt>(ValSize)->getSExtValue() - 40;
          AllPositions.push_back(Size);

          //Check if there is a private variable in the middle!!!
          int BytesDiff, LastBytes, DiffCount;
          if(AllPositions.size() != 1){
            BytesDiff= AllPositions.back()-AllPositions[AllPositions.size()-2];
            LastBytes=  getTypeSizeInBytes(TaskPrivatesType[AllPositions.size()-2]);
          }
          else{
            BytesDiff= AllPositions.back();
            LastBytes = 0;
          }
          if(BytesDiff != LastBytes){
            int Index=0;
            DiffCount = LastBytes;
            while(DiffCount < BytesDiff){
              PrivateValues.push_back(-2);
              //dbgs()<< "Es " << DiffCount << " " << BytesDiff << " \n";
              if(LastBytes!=0)
                DiffCount+= 4;
              else
                DiffCount+= 4; //getTypeSizeInBytes(TaskPrivatesType[Index]);
              Index++;
            }
          }
          

          Value *ValStored = nullptr;
          if (StoreInst *ConstantStore =
                  dyn_cast<StoreInst>(Start->getNextNode()->getNextNode())) {
            ValStored = ConstantStore->getValueOperand();
          } else if (StoreInst *ConstantStore =
                         dyn_cast<StoreInst>(Start->getNextNode())) {
            ValStored = ConstantStore->getValueOperand();
          }
          else if (StoreInst *ConstantStore =
                         dyn_cast<StoreInst>(Start->getNextNode()->getNextNode()->getNextNode())) {
            ValStored = ConstantStore->getValueOperand();
          }
          if (ValStored) {
            if (ConstantInt *CI = dyn_cast<ConstantInt>(ValStored)) {
              PrivateValues.push_back(CI->getSExtValue());
            } else {

              if(PrivateValues.size()==0)
                FinalFirstPrivateOffsets.push_back(0);
              else{
                FinalFirstPrivateOffsets.push_back(AllPositions.back());
              }

              int CurrentPos = PrivateValues.size();
              PrivateValues.push_back(-1);
              FirstPrivateSizes.push_back(getTypeSizeInBytes(TaskPrivatesType[CurrentPos]));
            }
          }
        }
      }
    }
    Start = Start->getNextNode();
  }

  // Private values that are not in the middle are stored at the end
  int diff = TaskPrivatesType.size() - PrivateValues.size();
  for (int i = 0; i < diff; i++) {
    PrivateValues.push_back(-2);
  }
  
  TaskFound.FirstPrivateData = PrivateValues;
  

  // Get shared data types
  SmallVector<Type *, 2> TaskSharedsType;
  BasicBlock *FirstBB = &EntryPointFunction->getEntryBlock();
  if (auto *SharedsBitcast =
          dyn_cast<BitCastInst>(FirstBB->getFirstNonPHIOrDbgOrLifetime())) {
    if (auto *SharedsLoad = dyn_cast<LoadInst>(SharedsBitcast->getNextNode())) {
      Type *SharedsType = SharedsLoad->getPointerOperandType()
                              ->getPointerElementType()
                              ->getPointerElementType();
      for (int j = 0; j < (int)SharedsType->getStructNumElements(); j++)
        TaskSharedsType.push_back(SharedsType->getStructElementType(j));
    }
  }

  // Store position of shared data
  SmallVector<int, 2> FinalSharedPositions;
  SmallVector<int, 2> FinalFirstPrivatePositions;

  //TODO: Falla cuando el taskgraph y el task alloc no estan en la misma funcion
  Value *FuncArg = F.getArg(0);
  std::vector<Value *> ArgPositions(FuncArg->getNumUses());
  for (Value *ArgUser : FuncArg->users()) {
    if (GetElementPtrInst *GetArg = dyn_cast<GetElementPtrInst>(ArgUser)) {
      Value *PositionValue = GetArg->getOperand(2);
      int position = dyn_cast<ConstantInt>(PositionValue)->getSExtValue();
      if (LoadInst *ArgLoad = dyn_cast<LoadInst>(GetArg->getNextNode())) {
        ArgPositions[position] = ArgLoad;
      } else {
        Start = TaskCallInst.getPrevNode();
        bool Found = false;
        while (!Found) {
          if (LoadInst *ArgLoad = dyn_cast<LoadInst>(Start)) {
            if (ArgLoad->getPointerOperand() == GetArg){
              ArgPositions[position] = ArgLoad;
              Found = true;
            }
          }
          if(Start->getPrevNode()==nullptr)
            break;

          Start = Start->getPrevNode();
        }
      }
    }
  }
  Start = dyn_cast<Instruction>(TaskAlloc);
  End = dyn_cast<Instruction>(&TaskCallInst);

  while (Start != End) {
    for (int i = 0; i < (int)ArgPositions.size(); i++){
      if(ArgPositions[i] ==nullptr) continue;
      for (Value *ArgUser : ArgPositions[i]->users()) {
        if (Start == dyn_cast<Instruction>(ArgUser)) {
          if(dyn_cast<StoreInst>(ArgUser))
            FinalSharedPositions.push_back(i);
          else if(dyn_cast<StoreInst>(dyn_cast<Instruction>(ArgUser)->getNextNode()))
            FinalFirstPrivatePositions.push_back(i);
        }
      }
    }
    Start = Start->getNextNode();
  }

  int i;
  for (i = 0; i < (int)TasksAllocInfo.size(); i++) {
    if (TasksAllocInfo[i].entryPoint == EntryPoint) {
      break;
    }
  }

  if (i == (int)TasksAllocInfo.size())
    TasksAllocInfo.push_back({Flags, SizeOfTask, SizeOfShareds, EntryPoint,
                              TaskPrivatesType, TaskSharedsType,
                              FinalSharedPositions, FinalFirstPrivatePositions, FinalFirstPrivateOffsets, FirstPrivateSizes});

  return i;
}

void TaskDependencyGraphData::findOpenMPTasks(Function &F, DominatorTree &DT) {

  // Iterate over the BB
  SmallVector<BasicBlock *, 8> Worklist;
  SmallPtrSet<BasicBlock *, 8> Visited;
  Worklist.push_back(&F.getEntryBlock());
  Visited.insert(&F.getEntryBlock());

  // Incremental task id
  int NumTasks = 0;

  if (Prealloc) {
    for (Value *FunctionUse : F.users()) {
      for (Value *FunctionUseReal : FunctionUse->users()) {
        if (dyn_cast<CallInst>(FunctionUseReal)) {
          Instruction *TaskgraphCall = dyn_cast<Instruction>(FunctionUseReal);
          Instruction *SetTdg = TaskgraphCall->getPrevNode();
          while (!dyn_cast<CallInst>(SetTdg)) {
            SetTdg = SetTdg->getPrevNode();
          }
          CallInst *SetTdgCall = dyn_cast<CallInst>(SetTdg);
          ConstantInt *NumPreallocConstant =
              dyn_cast<ConstantInt>(SetTdgCall->getOperand(0));
          NumPreallocs = NumPreallocConstant->getSExtValue();
        }
      }
    }
  }

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
          if(Prealloc)
            TaskFound.pragmaId = findPragmaId(*II, TaskFound, F);
          else
            TaskFound.pragmaId = -1;
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
          if (Prealloc)
            TaskFound.pragmaId = findPragmaId(*II, TaskFound, F);
          else
            TaskFound.pragmaId = -1;
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

  if (Prealloc && (NumPreallocs == 0 || NumPreallocs > FunctionTasks.size()))
    NumPreallocs = FunctionTasks.size();
  // Remove transisitve edges
  erase_transitive_edges();

  // print_tdg();
  if (FunctionTasks.size()) {
    print_tdg_to_dot(F.getParent()->getSourceFileName());
    generate_analysis_tdg_file(F.getParent()->getSourceFileName());
    generate_runtime_tdg_file(F.getParent()->getSourceFileName());
  }
  TasksAllocInfo.clear();
  FunctionTasks.clear();
}

bool TaskDependencyGraphPass::runOnModule(Module &M) {
  TaskDependencyGraphData TDG;

  for (Function &F : M) {
    if (F.isDeclaration() || F.empty())
      continue;
    // Only check functions with tasks
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    if (F.hasFnAttribute("llvm.omp.taskgraph.static")) {
      if (F.hasFnAttribute("llvm.omp.taskgraph.prealloc"))
        TDG.setPrealloc();
      TDG.findOpenMPTasks(F, DT);
    }
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
    if (F.hasFnAttribute("llvm.omp.taskgraph.static")) {
      if (F.hasFnAttribute("llvm.omp.taskgraph.prealloc"))
        TDG.setPrealloc();
      TDG.findOpenMPTasks(F, DT);
    }
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
