#ifndef LLVM_ANALYSIS_TASK_DEPENDENCY_GRAPH_H
#define LLVM_ANALYSIS_TASK_DEPENDENCY_GRAPH_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

ModulePass *createTaskDependencyGraphPass();

struct TaskDependInfo {
  Value *base;                    // base Address
  bool isArray;                   // if the dependency is an array or not
  SmallVector<uint64_t, 2> index; // indexes in the subscript
  int type;                       // 1: in, 2:out, 3:inout
};

struct TaskAllocInfo {
  int flags;         // task flags
  int sizeOfTask;    // size of Task struct
  int sizeOfShareds; // size of Shareds
  Value *entryPoint; // entry point of the Task
  SmallVector<Type *, 2>
      privatesType; // Vector to store the type of private variables
  SmallVector<Type *, 2>
      sharedsType; // Vector to store the type of shared variables
  SmallVector<int, 2> sharedDataPositions;
  SmallVector<int, 2> firstPrivateDataPositions;
  SmallVector<int, 2> firstPrivateDataOffsets;
  SmallVector<int, 2> firstPrivateDataSizes;
};

struct TaskInfo {
  int id; // Task id
  int pragmaId;
  StringRef ident;                            // Task ident line
  SmallVector<uint64_t, 2> successors;        // Ids of successors
  SmallVector<uint64_t, 2> predecessors;      // Ids of predecessors
  SmallVector<TaskDependInfo, 2> TaskDepInfo; // Task dependency information
  SmallVector<int64_t> FirstPrivateData;
};

struct ident_color {
  StringRef ident;
  StringRef color;
};

class TaskDependencyGraphData {
  // Info used by the transform pass
  SmallVector<ident_color, 10> ColorMap;
  int MaxNesting = 3;
  int ColorsUsed = 0;
  bool Prealloc = false;
  uint NumPreallocs = 0;
  SmallVector<TaskInfo, 10> FunctionTasks; // Vector to store all tasks found
  SmallVector<TaskAllocInfo, 8>
      TasksAllocInfo; // Vector to store all tasks alloc info found

public:
  void traverse_node(SmallVectorImpl<uint64_t> &edges_to_check, int node,
                     int master, int nesting_level, std::vector<bool> &Visited);
  void print_tdg();
  void print_tdg_to_dot(StringRef ModuleName, int ntdgs, Function &F);
  void generate_analysis_tdg_file(StringRef ModuleName);
  void generate_runtime_tdg_file(StringRef ModuleName, Function &F, int ntdgs);
  void clear();
  void obtainTaskIdent(TaskInfo &TaskFound, CallInst &TaskCall);
  // static so that it can be used in cuda generation
  static std::string get_c_struct_from_types(SmallVectorImpl<Type *> &types,
                                      int struct_id, bool is_private, std::string name = "");
  std::string get_task_layout(std::string LongestPrivateName,
                              std::string LongestSharedName);
  void erase_transitive_edges();
  bool checkDependency(TaskDependInfo &Source, TaskDependInfo &Dest);
  void findOpenMPTasks(Function &F, DominatorTree &DT, int ntdgs);
  void obtainTaskInfo(TaskInfo &TaskFound, CallInst &TaskCall,
                      DominatorTree &DT);
  int findPragmaId(CallInst &TaskCallInst, TaskInfo &TaskFound, Function &F, DominatorTree &DT);
  void setPrealloc() { Prealloc = true; }
  bool invalidate(Module &, const PreservedAnalyses &,
                  ModuleAnalysisManager::Invalidator &) {
    return false;
  }
};

class TaskDependencyGraphPass : public ModulePass {

public:
  static char ID;
  explicit TaskDependencyGraphPass();
  ~TaskDependencyGraphPass() override;

  /// @name FunctionPass interface
  //@{
  bool runOnModule(Module &M) override;
  void releaseMemory() override;
  // void verifyAnalysis() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void print(raw_ostream &OS, const Module *) const override;
  //@}
};

// New PassManager Analysis
class TaskDependencyGraphAnalysis
    : public AnalysisInfoMixin<TaskDependencyGraphAnalysis> {
  friend AnalysisInfoMixin<TaskDependencyGraphAnalysis>;
  static AnalysisKey Key;
  TaskDependencyGraphData TDG;

public:
  /// Provide the result typedef for this analysis pass.
  using Result = TaskDependencyGraphData;
  TaskDependencyGraphData run(Module &M, ModuleAnalysisManager &AM);
};

class TaskDependencyGraphAnalysisPass
    : public AnalysisInfoMixin<TaskDependencyGraphAnalysisPass> {

public:
  /// Provide the result typedef for this analysis pass.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_TASK_DEPENDENCY_GRAPH_H
