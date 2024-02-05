/*
 * kmp_taskdeps.cpp
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//#define KMP_SUPPORT_GRAPH_OUTPUT 1

#include "kmp.h"
#include "kmp_io.h"
#include "kmp_wait_release.h"
#include "kmp_taskdeps.h"
#if OMPT_SUPPORT
#include "ompt-specific.h"
#endif

// TODO: Improve memory allocation? keep a list of pre-allocated structures?
// allocate in blocks? re-use list finished list entries?
// TODO: don't use atomic ref counters for stack-allocated nodes.
// TODO: find an alternate to atomic refs for heap-allocated nodes?
// TODO: Finish graph output support
// TODO: kmp_lock_t seems a tad to big (and heavy weight) for this. Check other
// runtime locks
// TODO: Any ITT support needed?

bool KMP_TDG_TRACING =
    0; // set to 1 to turn on the printf throughout the library
bool KMP_TDG_TIMING =
    0; // set to 1 to turn on time measurements, needed to sort tasks
extern bool enable_tdg_scheduling;
void debug_print(const char *format, ...);
void __kmp_enable_tasking(kmp_task_team_t *task_team, kmp_info_t *this_thr);
void sync_tdg_tasks_for_task_team(kmp_int32, kmp_task_team_t *,
                                  kmp_task_team_t *, kmp_int32);
int __kmp_realloc_task_threads_data(kmp_info_t *thread,
                                           kmp_task_team_t *task_team);
#if LIBOMP_TASKGRAPH
#include <utility>

// Colors for the graphviz output
const char *ColorNames[] = {
    "aquamarine3", "crimson",         "chartreuse",  "blue2",
    "darkorchid3", "darkgoldenrod1",  "deeppink4",   "gray19",
    "indigo",      "indianred",       "forestgreen", "navy",
    "orangered2",  "slateblue3",      "yellowgreen", "salmon",
    "purple",      "mediumturquoise", "slategray3"};

extern size_t __kmp_round_up_to_val(size_t size, size_t val);
extern void __kmp_free_task_and_ancestors(kmp_int32 gtid,
                                          kmp_taskdata_t *taskdata,
                                          kmp_info_t *thread);
kmp_task_t *kmp_init_lazy_task(int static_id,
                               kmp_int32 gtid, kmp_node_info *thisRecordMap, kmp_tdg_info *globalTdg);
void kmp_insert_task_in_indexer(kmp_task_t *task);

#endif // LIBOMP_TASKGRAPH

#ifdef KMP_SUPPORT_GRAPH_OUTPUT
static std::atomic<kmp_int32> kmp_node_id_seed = 0;
#endif

static void __kmp_init_node(kmp_depnode_t *node) {
  node->dn.successors = NULL;
  node->dn.task = NULL; // will point to the right task
  // once dependences have been processed
  for (int i = 0; i < MAX_MTX_DEPS; ++i)
    node->dn.mtx_locks[i] = NULL;
  node->dn.mtx_num_locks = 0;
  __kmp_init_lock(&node->dn.lock);
  KMP_ATOMIC_ST_RLX(&node->dn.nrefs, 1); // init creates the first reference
#ifdef KMP_SUPPORT_GRAPH_OUTPUT
  node->dn.id = KMP_ATOMIC_INC(&kmp_node_id_seed);
#endif
#if USE_ITT_BUILD && USE_ITT_NOTIFY
  __itt_sync_create(node, "OMP task dep node", NULL, 0);
#endif
}

static inline kmp_depnode_t *__kmp_node_ref(kmp_depnode_t *node) {
  KMP_ATOMIC_INC(&node->dn.nrefs);
  return node;
}

enum { KMP_DEPHASH_OTHER_SIZE = 97, KMP_DEPHASH_MASTER_SIZE = 997 };

size_t sizes[] = {997, 2003, 4001, 8191, 16001, 32003, 64007, 131071, 270029};
const size_t MAX_GEN = 8;

static inline size_t __kmp_dephash_hash(kmp_intptr_t addr, size_t hsize) {
  // TODO alternate to try: set = (((Addr64)(addrUsefulBits * 9.618)) %
  // m_num_sets );
  return ((addr >> 6) ^ (addr >> 2)) % hsize;
}

static kmp_dephash_t *__kmp_dephash_extend(kmp_info_t *thread,
                                           kmp_dephash_t *current_dephash) {
  kmp_dephash_t *h;

  size_t gen = current_dephash->generation + 1;
  if (gen >= MAX_GEN)
    return current_dephash;
  size_t new_size = sizes[gen];

  size_t size_to_allocate =
      new_size * sizeof(kmp_dephash_entry_t *) + sizeof(kmp_dephash_t);

#if USE_FAST_MEMORY
  h = (kmp_dephash_t *)__kmp_fast_allocate(thread, size_to_allocate);
#else
  h = (kmp_dephash_t *)__kmp_thread_malloc(thread, size_to_allocate);
#endif

  h->size = new_size;
  h->nelements = current_dephash->nelements;
  h->buckets = (kmp_dephash_entry **)(h + 1);
  h->generation = gen;
  h->nconflicts = 0;
  h->last_all = current_dephash->last_all;

  // make sure buckets are properly initialized
  for (size_t i = 0; i < new_size; i++) {
    h->buckets[i] = NULL;
  }

  // insert existing elements in the new table
  for (size_t i = 0; i < current_dephash->size; i++) {
    kmp_dephash_entry_t *next, *entry;
    for (entry = current_dephash->buckets[i]; entry; entry = next) {
      next = entry->next_in_bucket;
      // Compute the new hash using the new size, and insert the entry in
      // the new bucket.
      size_t new_bucket = __kmp_dephash_hash(entry->addr, h->size);
      entry->next_in_bucket = h->buckets[new_bucket];
      if (entry->next_in_bucket) {
        h->nconflicts++;
      }
      h->buckets[new_bucket] = entry;
    }
  }

  // Free old hash table
#if USE_FAST_MEMORY
  __kmp_fast_free(thread, current_dephash);
#else
  __kmp_thread_free(thread, current_dephash);
#endif

  return h;
}

static kmp_dephash_t *__kmp_dephash_create(kmp_info_t *thread,
                                           kmp_taskdata_t *current_task) {
  kmp_dephash_t *h;

  size_t h_size;

  if (current_task->td_flags.tasktype == TASK_IMPLICIT)
    h_size = KMP_DEPHASH_MASTER_SIZE;
  else
    h_size = KMP_DEPHASH_OTHER_SIZE;

  size_t size = h_size * sizeof(kmp_dephash_entry_t *) + sizeof(kmp_dephash_t);

#if USE_FAST_MEMORY
  h = (kmp_dephash_t *)__kmp_fast_allocate(thread, size);
#else
  h = (kmp_dephash_t *)__kmp_thread_malloc(thread, size);
#endif
  h->size = h_size;

  h->generation = 0;
  h->nelements = 0;
  h->nconflicts = 0;
  h->buckets = (kmp_dephash_entry **)(h + 1);
  h->last_all = NULL;

  for (size_t i = 0; i < h_size; i++)
    h->buckets[i] = 0;

  return h;
}

static kmp_dephash_entry *__kmp_dephash_find(kmp_info_t *thread,
                                             kmp_dephash_t **hash,
                                             kmp_intptr_t addr) {
  kmp_dephash_t *h = *hash;
  if (h->nelements != 0 && h->nconflicts / h->size >= 1) {
    *hash = __kmp_dephash_extend(thread, h);
    h = *hash;
  }
  size_t bucket = __kmp_dephash_hash(addr, h->size);

  kmp_dephash_entry_t *entry;
  for (entry = h->buckets[bucket]; entry; entry = entry->next_in_bucket)
    if (entry->addr == addr)
      break;

  if (entry == NULL) {
// create entry. This is only done by one thread so no locking required
#if USE_FAST_MEMORY
    entry = (kmp_dephash_entry_t *)__kmp_fast_allocate(
        thread, sizeof(kmp_dephash_entry_t));
#else
    entry = (kmp_dephash_entry_t *)__kmp_thread_malloc(
        thread, sizeof(kmp_dephash_entry_t));
#endif
    entry->addr = addr;
    if (!h->last_all) // no predecessor task with omp_all_memory dependence
      entry->last_out = NULL;
    else // else link the omp_all_memory depnode to the new entry
      entry->last_out = __kmp_node_ref(h->last_all);
    entry->last_set = NULL;
    entry->prev_set = NULL;
    entry->last_flag = 0;
    entry->mtx_lock = NULL;
    entry->next_in_bucket = h->buckets[bucket];
    h->buckets[bucket] = entry;
    h->nelements++;
    if (entry->next_in_bucket)
      h->nconflicts++;
  }
  return entry;
}

static kmp_depnode_list_t *__kmp_add_node(kmp_info_t *thread,
                                          kmp_depnode_list_t *list,
                                          kmp_depnode_t *node) {
  kmp_depnode_list_t *new_head;

#if USE_FAST_MEMORY
  new_head = (kmp_depnode_list_t *)__kmp_fast_allocate(
      thread, sizeof(kmp_depnode_list_t));
#else
  new_head = (kmp_depnode_list_t *)__kmp_thread_malloc(
      thread, sizeof(kmp_depnode_list_t));
#endif

  new_head->node = __kmp_node_ref(node);
  new_head->next = list;

  return new_head;
}

static inline void __kmp_track_dependence(kmp_int32 gtid, kmp_depnode_t *source,
                                          kmp_depnode_t *sink,
                                          kmp_task_t *sink_task) {

kmp_taskdata_t *task_source = KMP_TASK_TO_TASKDATA(source->dn.task);
kmp_taskdata_t *task_sink = KMP_TASK_TO_TASKDATA(sink_task);
#if LIBOMP_TASKGRAPH
  if((source->dn.task && sink_task) && ((task_source->is_taskgraph && !task_sink->is_taskgraph) || (!task_source->is_taskgraph && task_sink->is_taskgraph))){
    printf("Internal OpenMP error: task dependency detected between a task inside a taskgraph and a task outside, this is not supported \n");
  }
  if (task_sink->is_taskgraph && task_sink->tdg->tdgStatus == KMP_TDG_RECORDING) {
    kmp_node_info *sourceNode = &task_sink->tdg->recordMap[task_source->td_task_id];
    bool exists = false;
    for (int i = 0; i < sourceNode->nsuccessors; i++) {
      if (sourceNode->successors[i] == task_sink->td_task_id) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      if (sourceNode->nsuccessors >= sourceNode->successors_size) {
        sourceNode->successors_size = sourceNode->successors_size * 2;
        sourceNode->successors = (kmp_int32 *)realloc(
            sourceNode->successors,
            sourceNode->successors_size * sizeof(kmp_int32));
      }

      sourceNode->successors[sourceNode->nsuccessors] = task_sink->td_task_id;
      sourceNode->nsuccessors++;

      kmp_node_info *sinkNode = &(task_sink->tdg->recordMap[task_sink->td_task_id]);
      sinkNode->npredecessors++;
    }
  }
#endif

#ifdef KMP_SUPPORT_GRAPH_OUTPUT
  // do not use sink->dn.task as that is only filled after the dependences
  // are already processed!

  __kmp_printf("%d(%s) -> %d(%s)\n", source->dn.id,
               task_source->td_ident->psource, sink->dn.id,
               task_sink->td_ident->psource);
#endif
#if OMPT_SUPPORT && OMPT_OPTIONAL
  /* OMPT tracks dependences between task (a=source, b=sink) in which
     task a blocks the execution of b through the ompt_new_dependence_callback
     */
  if (ompt_enabled.ompt_callback_task_dependence) {
    kmp_taskdata_t *task_source = KMP_TASK_TO_TASKDATA(source->dn.task);
    ompt_data_t *sink_data;
    if (sink_task)
      sink_data = &(KMP_TASK_TO_TASKDATA(sink_task)->ompt_task_info.task_data);
    else
      sink_data = &__kmp_threads[gtid]->th.ompt_thread_info.task_data;

    ompt_callbacks.ompt_callback(ompt_callback_task_dependence)(
        &(task_source->ompt_task_info.task_data), sink_data);
  }
#endif /* OMPT_SUPPORT && OMPT_OPTIONAL */
}

static inline kmp_int32
__kmp_depnode_link_successor(kmp_int32 gtid, kmp_info_t *thread,
                             kmp_task_t *task, kmp_depnode_t *node,
                             kmp_depnode_list_t *plist) {
  if (!plist)
    return 0;
  kmp_int32 npredecessors = 0;
  // link node as successor of list elements
  for (kmp_depnode_list_t *p = plist; p; p = p->next) {
    kmp_depnode_t *dep = p->node;
#if LIBOMP_TASKGRAPH
    kmp_tdg_status tdgStatus = KMP_TDG_NONE;
    if (task){
      if(KMP_TASK_TO_TASKDATA(task)->is_taskgraph)
        tdgStatus = KMP_TASK_TO_TASKDATA(task)->tdg->tdgStatus;
      if (tdgStatus == KMP_TDG_RECORDING)
        __kmp_track_dependence(gtid, dep, node, task);
    }
#endif
    if (dep->dn.task) {
      KMP_ACQUIRE_DEPNODE(gtid, dep);
      if (dep->dn.task) {
#if LIBOMP_TASKGRAPH
        if (!(tdgStatus == KMP_TDG_RECORDING) && task)
#endif
          __kmp_track_dependence(gtid, dep, node, task);
#if LIBOMP_TASKGRAPH
        else if (KMP_TASK_TO_TASKDATA(dep->dn.task)->td_flags.onced) {
          KMP_RELEASE_DEPNODE(gtid, dep);
          continue;
        }
#endif
        dep->dn.successors = __kmp_add_node(thread, dep->dn.successors, node);
        KA_TRACE(40, ("__kmp_process_deps: T#%d adding dependence from %p to "
                      "%p\n",
                      gtid, KMP_TASK_TO_TASKDATA(dep->dn.task),
                      KMP_TASK_TO_TASKDATA(task)));
        npredecessors++;
      }
      KMP_RELEASE_DEPNODE(gtid, dep);
    }
  }
  return npredecessors;
}

static inline kmp_int32 __kmp_depnode_link_successor(kmp_int32 gtid,
                                                     kmp_info_t *thread,
                                                     kmp_task_t *task,
                                                     kmp_depnode_t *source,
                                                     kmp_depnode_t *sink) {
  if (!sink)
    return 0;
  kmp_int32 npredecessors = 0;
#if LIBOMP_TASKGRAPH
  kmp_tdg_status tdgStatus = KMP_TDG_NONE;
  if (task){
    if(KMP_TASK_TO_TASKDATA(task)->is_taskgraph)
      tdgStatus = KMP_TASK_TO_TASKDATA(task)->tdg->tdgStatus;
    if (tdgStatus == KMP_TDG_RECORDING && sink->dn.task)
      __kmp_track_dependence(gtid, sink, source, task);
  }
#endif
  if (sink->dn.task) {
    // synchronously add source to sink' list of successors
    KMP_ACQUIRE_DEPNODE(gtid, sink);
    if (sink->dn.task) {
#if LIBOMP_TASKGRAPH
      if (!(tdgStatus == KMP_TDG_RECORDING) && task)
#endif
        __kmp_track_dependence(gtid, sink, source, task);
#if LIBOMP_TASKGRAPH
        else if (KMP_TASK_TO_TASKDATA(sink->dn.task)->td_flags.onced) {
          KMP_RELEASE_DEPNODE(gtid, sink);
          return npredecessors;
        }
#endif
      sink->dn.successors = __kmp_add_node(thread, sink->dn.successors, source);
      KA_TRACE(40, ("__kmp_process_deps: T#%d adding dependence from %p to "
                    "%p\n",
                    gtid, KMP_TASK_TO_TASKDATA(sink->dn.task),
                    KMP_TASK_TO_TASKDATA(task)));
      npredecessors++;
    }
    KMP_RELEASE_DEPNODE(gtid, sink);
  }
  return npredecessors;
}

static inline kmp_int32
__kmp_process_dep_all(kmp_int32 gtid, kmp_depnode_t *node, kmp_dephash_t *h,
                      bool dep_barrier, kmp_task_t *task) {
  KA_TRACE(30, ("__kmp_process_dep_all: T#%d processing dep_all, "
                "dep_barrier = %d\n",
                gtid, dep_barrier));
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_int32 npredecessors = 0;

  // process previous omp_all_memory node if any
  npredecessors +=
      __kmp_depnode_link_successor(gtid, thread, task, node, h->last_all);
  __kmp_node_deref(thread, h->last_all);
  if (!dep_barrier) {
    h->last_all = __kmp_node_ref(node);
  } else {
    // if this is a sync point in the serial sequence, then the previous
    // outputs are guaranteed to be completed after the execution of this
    // task so the previous output nodes can be cleared.
    h->last_all = NULL;
  }

  // process all regular dependences
  for (size_t i = 0; i < h->size; i++) {
    kmp_dephash_entry_t *info = h->buckets[i];
    if (!info) // skip empty slots in dephash
      continue;
    for (; info; info = info->next_in_bucket) {
      // for each entry the omp_all_memory works as OUT dependence
      kmp_depnode_t *last_out = info->last_out;
      kmp_depnode_list_t *last_set = info->last_set;
      kmp_depnode_list_t *prev_set = info->prev_set;
      if (last_set) {
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_set);
        __kmp_depnode_list_free(thread, last_set);
        __kmp_depnode_list_free(thread, prev_set);
        info->last_set = NULL;
        info->prev_set = NULL;
        info->last_flag = 0; // no sets in this dephash entry
      } else {
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_out);
      }
      __kmp_node_deref(thread, last_out);
      if (!dep_barrier) {
        info->last_out = __kmp_node_ref(node);
      } else {
        info->last_out = NULL;
      }
    }
  }
  KA_TRACE(30, ("__kmp_process_dep_all: T#%d found %d predecessors\n", gtid,
                npredecessors));
  return npredecessors;
}

template <bool filter>
static inline kmp_int32
__kmp_process_deps(kmp_int32 gtid, kmp_depnode_t *node, kmp_dephash_t **hash,
                   bool dep_barrier, kmp_int32 ndeps,
                   kmp_depend_info_t *dep_list, kmp_task_t *task) {
  KA_TRACE(30, ("__kmp_process_deps<%d>: T#%d processing %d dependences : "
                "dep_barrier = %d\n",
                filter, gtid, ndeps, dep_barrier));

  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_int32 npredecessors = 0;
  for (kmp_int32 i = 0; i < ndeps; i++) {
    const kmp_depend_info_t *dep = &dep_list[i];

    if (filter && dep->base_addr == 0)
      continue; // skip filtered entries

    kmp_dephash_entry_t *info =
        __kmp_dephash_find(thread, hash, dep->base_addr);
    kmp_depnode_t *last_out = info->last_out;
    kmp_depnode_list_t *last_set = info->last_set;
    kmp_depnode_list_t *prev_set = info->prev_set;

    if (dep->flags.out) { // out or inout --> clean lists if any
      if (last_set) {
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_set);
        __kmp_depnode_list_free(thread, last_set);
        __kmp_depnode_list_free(thread, prev_set);
        info->last_set = NULL;
        info->prev_set = NULL;
        info->last_flag = 0; // no sets in this dephash entry
      } else {
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_out);
      }
      __kmp_node_deref(thread, last_out);
      if (!dep_barrier) {
        info->last_out = __kmp_node_ref(node);
      } else {
        // if this is a sync point in the serial sequence, then the previous
        // outputs are guaranteed to be completed after the execution of this
        // task so the previous output nodes can be cleared.
        info->last_out = NULL;
      }
    } else { // either IN or MTX or SET
      if (info->last_flag == 0 || info->last_flag == dep->flag) {
        // last_set either didn't exist or of same dep kind
        // link node as successor of the last_out if any
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_out);
        // link node as successor of all nodes in the prev_set if any
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, prev_set);
        if (dep_barrier) {
          // clean last_out and prev_set if any; don't touch last_set
          __kmp_node_deref(thread, last_out);
          info->last_out = NULL;
          __kmp_depnode_list_free(thread, prev_set);
          info->prev_set = NULL;
        }
      } else { // last_set is of different dep kind, make it prev_set
        // link node as successor of all nodes in the last_set
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_set);
        // clean last_out if any
        __kmp_node_deref(thread, last_out);
        info->last_out = NULL;
        // clean prev_set if any
        __kmp_depnode_list_free(thread, prev_set);
        if (!dep_barrier) {
          // move last_set to prev_set, new last_set will be allocated
          info->prev_set = last_set;
        } else {
          info->prev_set = NULL;
          info->last_flag = 0;
        }
        info->last_set = NULL;
      }
      // for dep_barrier last_flag value should remain:
      // 0 if last_set is empty, unchanged otherwise
      if (!dep_barrier) {
        info->last_flag = dep->flag; // store dep kind of the last_set
        info->last_set = __kmp_add_node(thread, info->last_set, node);
      }
      // check if we are processing MTX dependency
      if (dep->flag == KMP_DEP_MTX) {
        if (info->mtx_lock == NULL) {
          info->mtx_lock = (kmp_lock_t *)__kmp_allocate(sizeof(kmp_lock_t));
          __kmp_init_lock(info->mtx_lock);
        }
        KMP_DEBUG_ASSERT(node->dn.mtx_num_locks < MAX_MTX_DEPS);
        kmp_int32 m;
        // Save lock in node's array
        for (m = 0; m < MAX_MTX_DEPS; ++m) {
          // sort pointers in decreasing order to avoid potential livelock
          if (node->dn.mtx_locks[m] < info->mtx_lock) {
            KMP_DEBUG_ASSERT(!node->dn.mtx_locks[node->dn.mtx_num_locks]);
            for (int n = node->dn.mtx_num_locks; n > m; --n) {
              // shift right all lesser non-NULL pointers
              KMP_DEBUG_ASSERT(node->dn.mtx_locks[n - 1] != NULL);
              node->dn.mtx_locks[n] = node->dn.mtx_locks[n - 1];
            }
            node->dn.mtx_locks[m] = info->mtx_lock;
            break;
          }
        }
        KMP_DEBUG_ASSERT(m < MAX_MTX_DEPS); // must break from loop
        node->dn.mtx_num_locks++;
      }
    }
  }
  KA_TRACE(30, ("__kmp_process_deps<%d>: T#%d found %d predecessors\n", filter,
                gtid, npredecessors));
  return npredecessors;
}

#define NO_DEP_BARRIER (false)
#define DEP_BARRIER (true)

// returns true if the task has any outstanding dependence
static bool __kmp_check_deps(kmp_int32 gtid, kmp_depnode_t *node,
                             kmp_task_t *task, kmp_dephash_t **hash,
                             bool dep_barrier, kmp_int32 ndeps,
                             kmp_depend_info_t *dep_list,
                             kmp_int32 ndeps_noalias,
                             kmp_depend_info_t *noalias_dep_list) {
  int i, n_mtxs = 0, dep_all = 0;
#if KMP_DEBUG
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
#endif
  KA_TRACE(20, ("__kmp_check_deps: T#%d checking dependences for task %p : %d "
                "possibly aliased dependences, %d non-aliased dependences : "
                "dep_barrier=%d .\n",
                gtid, taskdata, ndeps, ndeps_noalias, dep_barrier));

  // Filter deps in dep_list
  // TODO: Different algorithm for large dep_list ( > 10 ? )
  for (i = 0; i < ndeps; i++) {
    if (dep_list[i].base_addr != 0 &&
        dep_list[i].base_addr != (kmp_intptr_t)KMP_SIZE_T_MAX) {
      KMP_DEBUG_ASSERT(
          dep_list[i].flag == KMP_DEP_IN || dep_list[i].flag == KMP_DEP_OUT ||
          dep_list[i].flag == KMP_DEP_INOUT ||
          dep_list[i].flag == KMP_DEP_MTX || dep_list[i].flag == KMP_DEP_SET);
      for (int j = i + 1; j < ndeps; j++) {
        if (dep_list[i].base_addr == dep_list[j].base_addr) {
          if (dep_list[i].flag != dep_list[j].flag) {
            // two different dependences on same address work identical to OUT
            dep_list[i].flag = KMP_DEP_OUT;
          }
          dep_list[j].base_addr = 0; // Mark j element as void
        }
      }
      if (dep_list[i].flag == KMP_DEP_MTX) {
        // limit number of mtx deps to MAX_MTX_DEPS per node
        if (n_mtxs < MAX_MTX_DEPS && task != NULL) {
          ++n_mtxs;
        } else {
          dep_list[i].flag = KMP_DEP_OUT; // downgrade mutexinoutset to inout
        }
      }
    } else if (dep_list[i].flag == KMP_DEP_ALL ||
               dep_list[i].base_addr == (kmp_intptr_t)KMP_SIZE_T_MAX) {
      // omp_all_memory dependence can be marked by compiler by either
      // (addr=0 && flag=0x80) (flag KMP_DEP_ALL), or (addr=-1).
      // omp_all_memory overrides all other dependences if any
      dep_all = 1;
      break;
    }
  }

  // doesn't need to be atomic as no other thread is going to be accessing this
  // node just yet.
  // npredecessors is set -1 to ensure that none of the releasing tasks queues
  // this task before we have finished processing all the dependences
  node->dn.npredecessors = -1;

  // used to pack all npredecessors additions into a single atomic operation at
  // the end
  int npredecessors;

  if (!dep_all) { // regular dependences
    npredecessors = __kmp_process_deps<true>(gtid, node, hash, dep_barrier,
                                             ndeps, dep_list, task);
    npredecessors += __kmp_process_deps<false>(
        gtid, node, hash, dep_barrier, ndeps_noalias, noalias_dep_list, task);
  } else { // omp_all_memory dependence
    npredecessors = __kmp_process_dep_all(gtid, node, *hash, dep_barrier, task);
  }

  node->dn.task = task;
  KMP_MB();

  // Account for our initial fake value
  npredecessors++;

  // Update predecessors and obtain current value to check if there are still
  // any outstanding dependences (some tasks may have finished while we
  // processed the dependences)
  npredecessors =
      node->dn.npredecessors.fetch_add(npredecessors) + npredecessors;

  KA_TRACE(20, ("__kmp_check_deps: T#%d found %d predecessors for task %p \n",
                gtid, npredecessors, taskdata));

  // beyond this point the task could be queued (and executed) by a releasing
  // task...
  return npredecessors > 0 ? true : false;
}

/*!
@ingroup TASKING
@param loc_ref location of the original task directive
@param gtid Global Thread ID of encountering thread
@param new_task task thunk allocated by __kmp_omp_task_alloc() for the ''new
task''
@param ndeps Number of depend items with possible aliasing
@param dep_list List of depend items with possible aliasing
@param ndeps_noalias Number of depend items with no aliasing
@param noalias_dep_list List of depend items with no aliasing

@return Returns either TASK_CURRENT_NOT_QUEUED if the current task was not
suspended and queued, or TASK_CURRENT_QUEUED if it was suspended and queued

Schedule a non-thread-switchable task with dependences for execution
*/
kmp_int32 __kmpc_omp_task_with_deps(ident_t *loc_ref, kmp_int32 gtid,
                                    kmp_task_t *new_task, kmp_int32 ndeps,
                                    kmp_depend_info_t *dep_list,
                                    kmp_int32 ndeps_noalias,
                                    kmp_depend_info_t *noalias_dep_list) {
  // debug_print("kmpc_omp_task_with_deps called with gtid %d, tid %d\n", gtid,
  // __kmp_get_tid());
  kmp_taskdata_t *new_taskdata = KMP_TASK_TO_TASKDATA(new_task);

  KA_TRACE(10, ("__kmpc_omp_task_with_deps(enter): T#%d loc=%p task=%p\n", gtid,
                loc_ref, new_taskdata));
  __kmp_assert_valid_gtid(gtid);
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *current_task = thread->th.th_current_task;

#if LIBOMP_TASKGRAPH
  kmp_tdg_info *task_tdg = new_taskdata->tdg;
  if (new_taskdata->is_taskgraph && task_tdg->tdgStatus == KMP_TDG_RECORDING) {
    // Extend Map Size if needed
    if (new_taskdata->td_task_id >= (int)task_tdg->mapSize) {
      __kmp_acquire_bootstrap_lock(&new_taskdata->tdg->graph_lock);
      // mapSize could have been updated by another thread if recursive taskloop
      if (new_taskdata->td_task_id >= (int)task_tdg->mapSize) {
        kmp_int32 oldSize = task_tdg->mapSize;
        kmp_uint newSize = oldSize * 2;
        // We malloc and move the data instead of realloc in order to avoid data
        // races
        kmp_node_info *oldRecord = task_tdg->recordMap;
        kmp_node_info *newRecord =
            (kmp_node_info *)malloc(newSize * sizeof(kmp_node_info));
        KMP_MEMCPY(newRecord, task_tdg->recordMap,
                   oldSize * sizeof(kmp_node_info));

        task_tdg->recordMap = newRecord;
        free(oldRecord);

        task_tdg->taskIdent = (kmp_ident_task *)realloc(
            task_tdg->taskIdent, newSize * sizeof(kmp_ident_task));

        for (kmp_uint32 i = oldSize; i < newSize; i++) {
          kmp_int32 *successorsList =
              (kmp_int32 *)malloc(__kmp_tdg_initial_successors_size * sizeof(kmp_int32));
          kmp_node_info *node = &new_taskdata->tdg->recordMap[i];
          node->static_id = 0;
          node->task = nullptr;
          node->successors = successorsList;
          node->nsuccessors = 0;
          node->npredecessors = 0;
          node->successors_size = __kmp_tdg_initial_successors_size;
          node->static_thread = -1;
          KMP_ATOMIC_ST_RLX(&node->npredecessors_counter, 0);
        }
        // update the size at the end, so that we avoid other
        // threads use OldRecordMap while mapSize is already updated
        task_tdg->mapSize = newSize;
      }
      __kmp_release_bootstrap_lock(&task_tdg->graph_lock);
    }

    kmp_node_info *node = &task_tdg->recordMap[new_taskdata->td_task_id];
    task_tdg->taskIdent[new_taskdata->td_task_id].td_ident =
        new_taskdata->td_ident->psource;
    node->static_id = new_taskdata->td_task_id;
    node->task = new_task;
    node->parent_task = new_taskdata->td_parent;
    KMP_ATOMIC_INC(&task_tdg->numTasks);
  }

  if (new_taskdata->is_taskgraph && task_tdg->tdgStatus == KMP_TDG_FILL_DATA) {
    kmp_node_info *node =
        &(new_taskdata->tdg->recordMap[new_taskdata->td_task_id]);
    node->task = new_task;
    node->parent_task = new_taskdata->td_parent;

    // Reset parent task counters, since task is not executed
    KMP_ATOMIC_ST_RLX(&new_taskdata->td_parent->td_incomplete_child_tasks, 0);
    if (new_taskdata->td_parent->td_taskgroup)
      KMP_ATOMIC_ST_RLX(&new_taskdata->td_parent->td_taskgroup->count, 0);
    // Only need to keep track of allocated child tasks for explicit tasks since
    // implicit not deallocated
    if (new_taskdata->td_parent->td_flags.tasktype == TASK_EXPLICIT) {
      KMP_ATOMIC_ST_RLX(&new_taskdata->td_parent->td_allocated_child_tasks, 0);
    }

    return TASK_CURRENT_NOT_QUEUED;
  }
#endif

#if OMPT_SUPPORT
  if (ompt_enabled.enabled) {
    if (!current_task->ompt_task_info.frame.enter_frame.ptr)
      current_task->ompt_task_info.frame.enter_frame.ptr =
          OMPT_GET_FRAME_ADDRESS(0);
    if (ompt_enabled.ompt_callback_task_create) {
      ompt_callbacks.ompt_callback(ompt_callback_task_create)(
          &(current_task->ompt_task_info.task_data),
          &(current_task->ompt_task_info.frame),
          &(new_taskdata->ompt_task_info.task_data),
          ompt_task_explicit | TASK_TYPE_DETAILS_FORMAT(new_taskdata), 1,
          OMPT_LOAD_OR_GET_RETURN_ADDRESS(gtid));
    }

    new_taskdata->ompt_task_info.frame.enter_frame.ptr =
        OMPT_GET_FRAME_ADDRESS(0);
  }

#if OMPT_OPTIONAL
  /* OMPT grab all dependences if requested by the tool */
  if (ndeps + ndeps_noalias > 0 && ompt_enabled.ompt_callback_dependences) {
    kmp_int32 i;

    int ompt_ndeps = ndeps + ndeps_noalias;
    ompt_dependence_t *ompt_deps = (ompt_dependence_t *)KMP_OMPT_DEPS_ALLOC(
        thread, (ndeps + ndeps_noalias) * sizeof(ompt_dependence_t));

    KMP_ASSERT(ompt_deps != NULL);

    for (i = 0; i < ndeps; i++) {
      ompt_deps[i].variable.ptr = (void *)dep_list[i].base_addr;
      if (dep_list[i].flags.in && dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_inout;
      else if (dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_out;
      else if (dep_list[i].flags.in)
        ompt_deps[i].dependence_type = ompt_dependence_type_in;
      else if (dep_list[i].flags.mtx)
        ompt_deps[i].dependence_type = ompt_dependence_type_mutexinoutset;
      else if (dep_list[i].flags.set)
        ompt_deps[i].dependence_type = ompt_dependence_type_inoutset;
    }
    for (i = 0; i < ndeps_noalias; i++) {
      ompt_deps[ndeps + i].variable.ptr = (void *)noalias_dep_list[i].base_addr;
      if (noalias_dep_list[i].flags.in && noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inout;
      else if (noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_out;
      else if (noalias_dep_list[i].flags.in)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_in;
      else if (noalias_dep_list[i].flags.mtx)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_mutexinoutset;
      else if (noalias_dep_list[i].flags.set)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inoutset;
    }
    ompt_callbacks.ompt_callback(ompt_callback_dependences)(
        &(new_taskdata->ompt_task_info.task_data), ompt_deps, ompt_ndeps);
    /* We can now free the allocated memory for the dependences */
    /* For OMPD we might want to delay the free until end of this function */
    KMP_OMPT_DEPS_FREE(thread, ompt_deps);
  }
#endif /* OMPT_OPTIONAL */
#endif /* OMPT_SUPPORT */

  bool serial = current_task->td_flags.team_serial ||
                current_task->td_flags.tasking_ser ||
                current_task->td_flags.final;
  kmp_task_team_t *task_team = thread->th.th_task_team;
  serial = serial &&
           !(task_team && (task_team->tt.tt_found_proxy_tasks ||
                           task_team->tt.tt_hidden_helper_task_encountered));

  if (!serial && (ndeps > 0 || ndeps_noalias > 0)) {
    /* if no dependences have been tracked yet, create the dependence hash */
    if (current_task->td_dephash == NULL)
      current_task->td_dephash = __kmp_dephash_create(thread, current_task);

#if USE_FAST_MEMORY
    kmp_depnode_t *node =
        (kmp_depnode_t *)__kmp_fast_allocate(thread, sizeof(kmp_depnode_t));
#else
    kmp_depnode_t *node =
        (kmp_depnode_t *)__kmp_thread_malloc(thread, sizeof(kmp_depnode_t));
#endif

    __kmp_init_node(node);

    new_taskdata->td_depnode = node;

    if (__kmp_check_deps(gtid, node, new_task, &current_task->td_dephash,
                         NO_DEP_BARRIER, ndeps, dep_list, ndeps_noalias,
                         noalias_dep_list)) {
      KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d task had blocking "
                    "dependences: "
                    "loc=%p task=%p, return: TASK_CURRENT_NOT_QUEUED\n",
                    gtid, loc_ref, new_taskdata));
#if OMPT_SUPPORT
      if (ompt_enabled.enabled) {
        current_task->ompt_task_info.frame.enter_frame = ompt_data_none;
      }
#endif
      return TASK_CURRENT_NOT_QUEUED;
    }
  } else {
    KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d ignored dependences "
                  "for task (serialized) loc=%p task=%p\n",
                  gtid, loc_ref, new_taskdata));
  }

  KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d task had no blocking "
                "dependences : "
                "loc=%p task=%p, transferring to __kmp_omp_task\n",
                gtid, loc_ref, new_taskdata));

  kmp_int32 ret = __kmp_omp_task(gtid, new_task, true);
#if OMPT_SUPPORT
  if (ompt_enabled.enabled) {
    current_task->ompt_task_info.frame.enter_frame = ompt_data_none;
  }
#endif
  return ret;
}

#if OMPT_SUPPORT
void __ompt_taskwait_dep_finish(kmp_taskdata_t *current_task,
                                ompt_data_t *taskwait_task_data) {
  if (ompt_enabled.ompt_callback_task_schedule) {
    ompt_callbacks.ompt_callback(ompt_callback_task_schedule)(
        taskwait_task_data, ompt_taskwait_complete, NULL);
  }
  current_task->ompt_task_info.frame.enter_frame.ptr = NULL;
  *taskwait_task_data = ompt_data_none;
}
#endif /* OMPT_SUPPORT */

#if LIBOMP_TASKGRAPH
void print_tdg(kmp_tdg_info *thisTdg) {
  for (kmp_int32 i = 0; i < KMP_ATOMIC_LD_RLX(&thisTdg->numTasks); i++) {
    printf("TASK: %d Successors: ", thisTdg->recordMap[i].static_id);
    for (int j = 0; j < thisTdg->recordMap[i].nsuccessors; j++) {
      printf(" %d ", thisTdg->recordMap[thisTdg->recordMap[i].successors[j]].static_id);
    }
    printf(" Predecessors : %d ", thisTdg->recordMap[i].npredecessors);
    printf(" \n");
  }
}

// Depth First Search to look for transitive edges
void traverse_node(kmp_int32 *edges_to_check, kmp_int32 *num_edges,
                   kmp_int32 node, kmp_int32 nesting_level, int visited[],
                   kmp_tdg_info *thisTdg) {
  kmp_int32 *successors = thisTdg->recordMap[node].successors;
  kmp_int32 nsuccessors = thisTdg->recordMap[node].nsuccessors;
  visited[node] = true;
  for (int i = 0; i < nsuccessors; i++) {
    kmp_int32 successor = successors[i];
    for (int j = 0; j < *num_edges; j++) {
      kmp_int32 edge = edges_to_check[j];
      if (edge == successor) {
        // Remove edge
        edges_to_check[j] = -1;
        for (int x = j; x < (*num_edges) - 1; x++) {
          edges_to_check[x] = edges_to_check[x + 1];
          edges_to_check[x + 1] = -1;
        }
        *num_edges = *num_edges - 1;
        thisTdg->recordMap[edge].npredecessors--;
        break;
      }
    }
    if (visited[successor] == false && nesting_level < __kmp_tdg_maxNesting)
      traverse_node(edges_to_check, num_edges, successor, nesting_level + 1,
                    visited, thisTdg);
  }
}

void erase_transitive_edges(kmp_tdg_info *thisTdg) {
  kmp_int32 thisNumTasks = KMP_ATOMIC_LD_RLX(&thisTdg->numTasks);
  for (kmp_int32 i = 0; i < thisNumTasks; i++) {

    kmp_int32 nsuccessors = thisTdg->recordMap[i].nsuccessors;

    if (!nsuccessors)
      continue;

    int visited[thisNumTasks];
    memset(visited, false, sizeof(int) * thisNumTasks);
    visited[i] = true;
    // Copy succesors, as they may be modified
    kmp_int32 *successors =
        (kmp_int32 *)malloc(sizeof(kmp_int32) * nsuccessors);
    memcpy(successors, thisTdg->recordMap[i].successors,
           sizeof(kmp_int32) * nsuccessors);

    for (int j = 0; j < nsuccessors; j++) {
      bool deleted = true;
      for (int x = 0; x < nsuccessors; x++) {
        if (thisTdg->recordMap[i].successors[x] == successors[j])
          deleted = false;
      }
      if (!deleted)
        traverse_node(thisTdg->recordMap[i].successors,
                      &thisTdg->recordMap[i].nsuccessors, successors[j], 0, visited,
                      thisTdg);
    }
    // free succesors
    free(successors);
  }
}

/*
void erase_taskgroup() {
  for (kmp_uint i = 0; i < MapSize; i++) {
    if (recordMap[i].task == nullptr) {
      continue;
    }
    kmp_task_t *task = recordMap[i].task;
    kmp_taskdata_t *td = KMP_TASK_TO_TASKDATA(task);
    if (td->td_taskgroup) {
      td->td_taskgroup = nullptr;
    }
  }
}*/

void print_tdg_to_dot(kmp_tdg_info *thisTdg) {
  kmp_int32 thisNumTasks = KMP_ATOMIC_LD_RLX(&thisTdg->numTasks);
  char FileName[20];
  sprintf(FileName, "tdg_%u.dot", thisTdg->tdgId);
  FILE *f = fopen(FileName, "w");

  if (f == NULL) {
    printf("Error opening file!\n");
    exit(1);
  }

  fprintf(f, "digraph TDG {\n");
  fprintf(f, "   compound=true\n");
  fprintf(f, "   subgraph cluster_0 {\n");
  fprintf(f, "      label=TDG_%u\n", thisTdg->tdgId);

  for (kmp_int32 i = 0; i < thisNumTasks; i++) {

    const char *color = nullptr;
    const char *ident = thisTdg->taskIdent[i].td_ident;
    for (int j = 0; j < thisTdg->colorMapSize; j++) {
      if (thisTdg->colorMap[j].td_ident == nullptr) {
        thisTdg->colorMap[j].td_ident = ident;
        thisTdg->colorMap[j].color = ColorNames[thisTdg->colorIndex];
        thisTdg->colorIndex++;
	color = thisTdg->colorMap[j].color;
	if(thisTdg->colorIndex>= (int) (sizeof(ColorNames)/sizeof(ColorNames[0])))
	  thisTdg->colorIndex = 0;
        break;
      } else if (thisTdg->colorMap[j].td_ident == ident) {
        color = thisTdg->colorMap[j].color;
        break;
      }
     }

     if (color == nullptr) {
	 int oldSize = thisTdg->colorMapSize;
	 thisTdg->colorMapSize = thisTdg->colorMapSize * 2;
	 kmp_ident_color *oldColorMap = thisTdg->colorMap;
	 kmp_ident_color *newColorMap = (kmp_ident_color *)malloc(thisTdg->colorMapSize * sizeof(kmp_ident_color));
	 KMP_MEMCPY(newColorMap, thisTdg->colorMap, oldSize *  sizeof(kmp_ident_color));

	 thisTdg->colorMap = newColorMap;
	 free(oldColorMap);

	 thisTdg->colorMap[oldSize].td_ident = ident;
         thisTdg->colorMap[oldSize].color = ColorNames[thisTdg->colorIndex];
         thisTdg->colorIndex++;
         color = thisTdg->colorMap[oldSize].color;
         if(thisTdg->colorIndex>= (int) (sizeof(ColorNames)/sizeof(ColorNames[0])))
		thisTdg->colorIndex = 0;
	 for (int j = oldSize+1; j < thisTdg->colorMapSize; j++) {
		thisTdg->colorMap[j]= {nullptr, nullptr};
	 }
    }
    if (color == nullptr) {
      printf("Unexpected error, color not found \n");
    } else {
      fprintf(f, "      %d[color=%s,style=bold]\n", thisTdg->recordMap[i].static_id,
              color);
    }
  }
  fprintf(f, "   }\n");
  for (kmp_int32 i = 0; i < thisNumTasks; i++) {
    kmp_int32 nsuccessors = thisTdg->recordMap[i].nsuccessors;
    kmp_int32 *successors = thisTdg->recordMap[i].successors;
    if (nsuccessors) {
      for (int j = 0; j < nsuccessors; j++) {
        fprintf(f, "   %d -> %d \n", thisTdg->recordMap[i].static_id,
                thisTdg->recordMap[successors[j]].static_id);
      }
    } else {
      fprintf(f, "   %d \n", thisTdg->recordMap[i].static_id);
    }
  }
  fprintf(f, "   node [shape=plaintext];\n");
  fprintf(f, "    subgraph cluster_1000 {\n");
  fprintf(f, "      label=\"User functions:\"; style=\"rounded\";\n");
  fprintf(f, " user_funcs [label=<<table border=\"0\" cellspacing=\"10\" "
             "cellborder=\"0\">\n");
  for (int i = 0; i < thisTdg->colorMapSize; i++) {
    if (thisTdg->colorMap[i].td_ident == nullptr)
      break;
    fprintf(f, "      <tr>\n");
    fprintf(f,
            "         <td bgcolor=\"%s\" width=\"15px\" border=\"1\"></td>\n",
            thisTdg->colorMap[i].color);
    fprintf(f, "         <td>%s</td>\n", thisTdg->colorMap[i].td_ident);
    fprintf(f, "      </tr>\n");
  }
  fprintf(f, "      </table>>]\n");
  fprintf(f, "}}\n");
  fclose(f);
}

// task_insert: distribute tasks among threads.
// Returns a list of kmp_int32 (corresponding to task_id) for each thread.
//
// nthreads: the number of threads taking tasks, TDG thread is already excluded,
//           i.e., nthreads = team.nproc - 1
// ntasks_per_thread: number of tasks assigned to each thread, update
//                    and return to caller
static kmp_int32 **task_insert(kmp_int32 nthreads, kmp_int32 *ntasks_per_thread, kmp_int32 tdg_index) {
  kmp_node_info *thisRecordMap;
  kmp_int32 thisNumRoots;
  kmp_int32 *thisRootTasks;

  thisRecordMap = __kmp_global_tdgs[tdg_index].recordMap;
  thisNumRoots = __kmp_global_tdgs[tdg_index].numRoots;
  thisRootTasks = __kmp_global_tdgs[tdg_index].rootTasks;

  int dst_th_ct = 0; // thread counter
  kmp_int32 **result = (kmp_int32 **)malloc(sizeof(kmp_int32 *) * nthreads);

  for (int i = 0; i < nthreads; ++i) {
    // we are not sure how many tasks are going to be in each thread,
    // allocate number of root tasks to be pessimistic.
    result[i] = (kmp_int32 *)malloc(sizeof(kmp_int32) * thisNumRoots);
    for (kmp_uint j = 0; j < (kmp_uint) thisNumRoots; ++j) {
      result[i][j] = -1;
    }
  }

  // distribute tasks in a round robin way without sorting their duration
  for (kmp_int i = 0; i < thisNumRoots; ++i) {
    result[dst_th_ct][ntasks_per_thread[dst_th_ct]] = thisRecordMap[thisRootTasks[i]].static_id;
    ntasks_per_thread[dst_th_ct]++;
    dst_th_ct++;
    dst_th_ct %= nthreads;
  }

  return result;
}

// distribute_tasks: after recording a TDG, this method is invoked to
// evenly distribute root tasks among available threads.
//
// gtid: global thread id of the caller (TDG thread)
static void distribute_tasks(kmp_int32 gtid, kmp_int32 tdg_index) {
  kmp_node_info *thisRecordMap;
  thisRecordMap = __kmp_global_tdgs[tdg_index].recordMap;

  __kmp_tdg_task_teams_sync = true;
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_task_team_t *task_team = thread->th.th_task_team;
  kmp_int32 nthreads;

  if (task_team)
    nthreads = task_team->tt.tt_nproc;
  else
    return;

  kmp_int32 *ntasks_per_thread;

  if (task_team == NULL || nthreads == 1)
    return;

  ntasks_per_thread = (kmp_int32 *)calloc(nthreads, sizeof(kmp_int32));

  kmp_int32 **schedule =
      task_insert(nthreads, ntasks_per_thread, tdg_index);

  for (kmp_int32 dst_tid = 0; dst_tid < nthreads; ++dst_tid) {
    kmp_thread_data *thread_data = &task_team->tt.tt_threads_data[dst_tid];
    if (thread_data == NULL) {
      __kmp_enable_tasking(task_team, thread);
      thread_data = &task_team->tt.tt_threads_data[dst_tid];
    }
    KMP_ASSERT(thread_data != NULL);

    if (ntasks_per_thread[dst_tid] > 0) {
      __kmp_alloc_tdg_tasks(gtid, thread_data, tdg_index,
                            ntasks_per_thread[dst_tid]);
      KMP_ASSERT(thread_data->td.td_tdg_tasks[tdg_index] != NULL);
      for (kmp_int32 task_ct = 0; task_ct < ntasks_per_thread[dst_tid];
          ++task_ct) {
        __kmp_insert_task_into_tdg(gtid, thread_data, tdg_index,
                                  thisRecordMap[schedule[dst_tid][task_ct]].task);
      }
    }
  }

  free(ntasks_per_thread);
  free(schedule);
}

// wait all threads to schedule their tasks.
void wait_all_threads_scheduled(kmp_int32 gtid, kmp_int32 nthreads) {
  volatile int proceed;
  do {
    proceed = 0;
    for (int i = 0; i < nthreads; ++i)
      proceed |= __kmp_tdg_replaying[i];
  } while (proceed);
}

void __kmpc_execute_tdg(ident_t *loc_ref, kmp_int32 gtid, kmp_int32 tdg_index, bool nowait) {
   kmp_node_info *thisRecordMap;
   kmp_int32 *thisRootTasks;
   kmp_int32 thisNumRoots;
   thisRecordMap = __kmp_global_tdgs[tdg_index].recordMap;
   thisNumRoots = __kmp_global_tdgs[tdg_index].numRoots;
   thisRootTasks = __kmp_global_tdgs[tdg_index].rootTasks;
   kmp_int32 thisNumTasks = KMP_ATOMIC_LD_RLX(&__kmp_global_tdgs[tdg_index].numTasks);
   
   kmp_info_t *thread = __kmp_threads[gtid];
   kmp_thread_data_t *thread_data;
   kmp_taskdata_t *parent_task = thread->th.th_current_task;

   kmp_task_team_t *task_team = thread->th.th_task_team;
   kmp_int32 nthreads = 1;
   kmp_int32 tid = __kmp_tid_from_gtid(gtid);

   bool prealloc = false;
   if (__kmp_global_tdgs[tdg_index].tdgStatus == KMP_TDG_PREALLOC)
    prealloc = true;

   if (task_team) {
    nthreads = task_team->tt.tt_nproc;
    if (!(&task_team->tt.tt_threads_data[tid]))
      __kmp_realloc_task_threads_data(thread, task_team);

    thread_data = &task_team->tt.tt_threads_data[tid];
    if (__kmp_tdg_task_teams_sync) {
      kmp_task_team *other_task_team =
          thread->th.th_team->t.t_task_team[1 - thread->th.th_task_state];
      if (other_task_team && other_task_team->tt.tt_threads_data)
        sync_tdg_tasks_for_task_team(gtid, other_task_team, task_team,
                                     nthreads);
      __kmp_tdg_task_teams_sync = false;
    }
   }

   if (__kmp_global_tdgs[tdg_index].rec_taskred_data) {
        __kmpc_taskred_init(gtid, __kmp_global_tdgs[tdg_index].rec_num_taskred,
                            __kmp_global_tdgs[tdg_index].rec_taskred_data);
   }
   // Reset remaining tasks
   KMP_ATOMIC_ST_RLX(&__kmp_global_tdgs[tdg_index].remainingTasks, thisNumTasks);

  for (kmp_int32 j = 0; j < thisNumTasks; j++) {
    kmp_taskdata_t *td = KMP_TASK_TO_TASKDATA(thisRecordMap[j].task);

    if (!prealloc)
      td->td_parent = parent_task;

    thisRecordMap[j].parent_task = parent_task;
    kmp_taskgroup_t *parentTaskgroup =
        thisRecordMap[j].parent_task->td_taskgroup;

    KMP_ATOMIC_ST_RLX(&thisRecordMap[j].npredecessors_counter,
                      thisRecordMap[j].npredecessors);

    KMP_ATOMIC_INC(&thisRecordMap[j].parent_task->td_incomplete_child_tasks);

    if (parentTaskgroup) {
        KMP_ATOMIC_INC(&parentTaskgroup->count);
        // The taskgroup is different so we must update it
        if (!prealloc)
          td->td_taskgroup = parentTaskgroup;
    } else if (!prealloc && td->td_taskgroup != nullptr) {
        // If the parent doesnt have a taskgroup, remove it from the task
        td->td_taskgroup = nullptr;
    }

    if (thisRecordMap[j].parent_task->td_flags.tasktype == TASK_EXPLICIT)
      KMP_ATOMIC_INC(&thisRecordMap[j].parent_task->td_allocated_child_tasks);
  }

  if (__kmp_global_tdgs[tdg_index].tdgStatus == KMP_TDG_PREALLOC) {
    for (kmp_int32 j = 0; j < thisNumRoots; j++) {
      thisRecordMap[thisRootTasks[j]].task = nullptr;
      kmp_task_t *task = kmp_init_lazy_task(
          thisRootTasks[j], gtid, thisRecordMap, &__kmp_global_tdgs[tdg_index]);
      if (task == nullptr) {
        insert_to_waiting_tdg(&thisRecordMap[thisRootTasks[j]], &__kmp_global_tdgs[tdg_index]);
      } else{
        __kmp_omp_task(gtid, task, true);
      }
    }
  } else if (thread->th.th_task_team == NULL || nowait ||
             __kmp_tdg_static_schedule || !enable_tdg_scheduling) {
    // tdgStatus != KMP_TDG_PREALLOC && (th_task_team == NULL || nowait)
    for (kmp_int32 j = 0; j < thisNumRoots; j++)
      __kmp_omp_task(gtid, thisRecordMap[thisRootTasks[j]].task, true);
  } else {
    // tdgStatus != KMP_TDG_PREALLOC && th_task_team != NULL && !nowait
    __kmp_curr_tdg_idx = tdg_index;
    memset(&__kmp_tdg_replaying[0], 1, sizeof(int) * MAX_NUM_PROC);
    // wake up threads to execute tasks
    if (UNLIKELY(!KMP_TASKING_ENABLED(task_team)))
      __kmp_enable_tasking(task_team, thread);
    for (kmp_int32 i = 0; i < thread_data->td.td_tdg_ntasks[tdg_index]; ++i) {
      __kmp_omp_task(gtid, thread_data->td.td_tdg_tasks[tdg_index][i], true);
    }
    __kmp_tdg_replaying[tid] = 0;
  }
}

void cleanTdgCreationInfo(kmp_int32 gtid) {
  for (int i = 0; i < __kmp_tdgCreationInfo_size; i++) {
    if (__kmp_tdgCreationInfo[i].gtid == gtid) {
      __kmp_tdgCreationInfo[i].gtid = -1;
       KMP_ATOMIC_ST_RLX(&__kmp_tdgCreationInfo[i].currentTaskGenID, -1);
      __kmp_tdgCreationInfo[i].tdg = nullptr;
    }
  }
}

kmp_int32 obtainTdgCreationInfo(kmp_int32 gtid) {
  if (__kmp_tdgCreationInfo_size == 0) {
    __kmp_tdgCreationInfo_size = 2;
    __kmp_tdgCreationInfo = (kmp_tdg_creation_info *)malloc(__kmp_tdgCreationInfo_size *
                                                  sizeof(kmp_tdg_creation_info));
    __kmp_tdgCreationInfo[0].gtid = gtid;
    __kmp_tdgCreationInfo[0].tdg = nullptr;
    KMP_ATOMIC_ST_RLX(&__kmp_tdgCreationInfo[0].currentTaskGenID,0);
   
    __kmp_ntdgsBeingCreated++;

    for (int i=1; i< __kmp_tdgCreationInfo_size; i++){
      __kmp_tdgCreationInfo[i].gtid = -1;
      __kmp_tdgCreationInfo[i].tdg = nullptr;
      KMP_ATOMIC_ST_RLX(&__kmp_tdgCreationInfo[i].currentTaskGenID, -1);
    }
    return 0;
  } else if (__kmp_ntdgsBeingCreated >= __kmp_tdgCreationInfo_size) {
    kmp_int32 oldsize = __kmp_tdgCreationInfo_size;
    __kmp_tdgCreationInfo_size += 2;
    __kmp_tdgCreationInfo = (kmp_tdg_creation_info *)realloc(
        __kmp_tdgCreationInfo, __kmp_tdgCreationInfo_size * sizeof(kmp_tdg_creation_info));


    __kmp_tdgCreationInfo[oldsize].gtid = gtid;
    __kmp_tdgCreationInfo[oldsize].tdg = nullptr;
    KMP_ATOMIC_ST_RLX(&__kmp_tdgCreationInfo[oldsize].currentTaskGenID, 0);

    __kmp_ntdgsBeingCreated++;
    for (int i=oldsize+1; i< __kmp_tdgCreationInfo_size; i++){
      __kmp_tdgCreationInfo[i].gtid = -1;
      __kmp_tdgCreationInfo[i].tdg = nullptr;
      KMP_ATOMIC_ST_RLX(&__kmp_tdgCreationInfo[i].currentTaskGenID, -1);
    }
    return oldsize;
  } else {
    for (int i = 0; i < __kmp_tdgCreationInfo_size; i++) {
      if (__kmp_tdgCreationInfo[i].gtid == -1) {

        __kmp_tdgCreationInfo[i].gtid = gtid;
        __kmp_tdgCreationInfo[i].tdg = nullptr;
        KMP_ATOMIC_ST_RLX(&__kmp_tdgCreationInfo[i].currentTaskGenID, 0);

        __kmp_ntdgsBeingCreated++;
        return i;
      }
    }
  }
  printf("Unreacheable: Error allocating tdg creation info \n");
  return -1;
}

void __kmpc_fill_data(ident_t *loc_ref, kmp_int32 gtid, void (*entry)(void *),
                      void *args) {

  entry(args);

  //Clean tdg creation info slot
  cleanTdgCreationInfo(gtid);
}

void __kmpc_set_tdg(struct kmp_node_info *tdg, kmp_int32 gtid, kmp_uint32 tdg_id, kmp_int32 ntasks, kmp_int32 *roots, kmp_int32 nroots) {

  // Skip tdgs that we already have
  for (int i = 0; i < __kmp_ntdgs; i++) {
    if (__kmp_global_tdgs[i].tdgId == tdg_id){
      return;
    }
  }

  __kmp_acquire_futex_lock(&__kmp_tdg_lock, 0);
  kmp_int32 tdgCreationIndex = obtainTdgCreationInfo(gtid);
  if(__kmp_ntdgs > NUM_TDG_LIMIT){
    printf("internal OpenMP error: Max number of TDGs exceeded \n");
  }
  int current_tdg_number = __kmp_ntdgs;
  __kmp_ntdgs++;
  __kmp_release_futex_lock(&__kmp_tdg_lock, 0);

  __kmp_tdgCreationInfo[tdgCreationIndex].tdg = &__kmp_global_tdgs[current_tdg_number];
  kmp_tdg_info *thisTdg =  &__kmp_global_tdgs[current_tdg_number];
  thisTdg->loc = "static";
  thisTdg->tdgId = tdg_id;
  thisTdg->mapSize = ntasks;
  thisTdg->numRoots = nroots;
  thisTdg->rootTasks = roots;
  thisTdg->recordMap = tdg;
  thisTdg->tdgStatus = KMP_TDG_FILL_DATA;

  KMP_ATOMIC_ST_RLX(&thisTdg->numTasks, ntasks);
  KMP_ATOMIC_ST_RLX(&thisTdg->remainingTasks,0);

  //printf("TDG set! \n");
}

kmp_int32 __kmpc_record(ident_t *loc_ref, kmp_int32 gtid, void (*entry)(void *),
                        void *args, kmp_int32 tdg_id, bool re_rec, bool nowait) {

  kmp_int32 thisMapSize = INIT_MAPSIZE;

  //Malloc and initialize TDG structures
  kmp_node_info *thisRecordMap = (kmp_node_info *)malloc(thisMapSize * sizeof(kmp_node_info));
  kmp_int32 thisColorMapSize = 20;
  kmp_int32 thisNumTasks;
  kmp_ident_color *thisColorMap = (kmp_ident_color *)malloc(thisColorMapSize * sizeof(kmp_ident_color));
  kmp_ident_task *thisTaskIdentMap = (kmp_ident_task *)malloc(thisMapSize * sizeof(kmp_ident_task));

  for (kmp_int32 i = 0; i < thisMapSize; i++) {
    thisTaskIdentMap[i] = {nullptr};
    kmp_int32 *successorsList =
        (kmp_int32 *)malloc(__kmp_tdg_initial_successors_size * sizeof(kmp_int32));
    kmp_node_info *node = &thisRecordMap[i];
    node->static_id = 0;
    node->task = nullptr;
    node->successors = successorsList;
    node->nsuccessors = 0;
    node->npredecessors = 0;
    node->successors_size = __kmp_tdg_initial_successors_size;
    node->static_thread = -1;
    KMP_ATOMIC_ST_RLX(&node->npredecessors_counter,0);
  }
  for (int i = 0; i < thisColorMapSize; i++) {
    thisColorMap[i] = {nullptr, nullptr};
  }

  //Lock tdg creation info and global tdg number counter
  __kmp_acquire_futex_lock(&__kmp_tdg_lock, 0);
  kmp_int32 tdgCreationIndex = obtainTdgCreationInfo(gtid);

  if(__kmp_ntdgs > NUM_TDG_LIMIT){
    printf("internal OpenMP error: Max number of TDGs exceeded \n");
  }
  int current_tdg_number = __kmp_ntdgs;

  if (!re_rec)
    __kmp_ntdgs++;
  else
    current_tdg_number--;

  __kmp_release_futex_lock(&__kmp_tdg_lock, 0);

  __kmp_tdgCreationInfo[tdgCreationIndex].tdg = &__kmp_global_tdgs[current_tdg_number];
  kmp_tdg_info *thisTdg =  &__kmp_global_tdgs[current_tdg_number];
  //Initialize tdg structure
  thisTdg->loc = loc_ref->psource;
  thisTdg->tdgId = tdg_id;
  thisTdg->mapSize = thisMapSize;
  thisTdg->numRoots = -1;
  thisTdg->rootTasks = nullptr;
  thisTdg->recordMap = thisRecordMap;
  thisTdg->taskIdent = thisTaskIdentMap;
  thisTdg->colorMap = thisColorMap;
  thisTdg->colorIndex = 0;
  thisTdg->colorMapSize = 20;
  thisTdg->tdgStatus = KMP_TDG_RECORDING;
  thisTdg->rec_num_taskred = 0;
  thisTdg->rec_taskred_data = nullptr;
  KMP_ATOMIC_ST_RLX(&thisTdg->numTasks, 0);
  KMP_ATOMIC_ST_RLX(&thisTdg->remainingTasks,0);

  __kmpc_taskgroup(loc_ref, gtid);
  //Start recording and wait to finish
  entry(args);
  __kmpc_end_taskgroup(loc_ref, gtid);

  thisNumTasks = KMP_ATOMIC_LD_RLX(&thisTdg->numTasks);
  
  //We have to update the mapsize and the record pointer, as it may change during task creation
  thisRecordMap = thisTdg->recordMap;
  thisMapSize = thisTdg->mapSize;

  // Store roots
  kmp_int32 *thisRootTasks = (kmp_int32 *)malloc(thisNumTasks * sizeof(kmp_int32));
  kmp_int32 thisNumRoots=0;
  for (kmp_int32 i = 0; i < thisNumTasks; i++) {
    if (thisRecordMap[i].npredecessors == 0) {
      thisRootTasks[thisNumRoots++] = i;
    }
  }

  //Update with roots info and mapsize
  thisTdg->mapSize = thisMapSize;
  thisTdg->numRoots = thisNumRoots;
  thisTdg->rootTasks = thisRootTasks;
  thisTdg->tdgStatus = KMP_TDG_NONE;
  thisTdg->spent_time = 0;

  //Static Mapping: distribute root tasks among threads.
  //Only activate when taskgraph is synchronous
  if (!nowait && !__kmp_tdg_static_schedule && enable_tdg_scheduling)
    distribute_tasks(gtid, current_tdg_number);
  //Clean tdg creation info slot
  cleanTdgCreationInfo(gtid);
  // printf("[OpenMP] Recording finished! \n");
  erase_transitive_edges(thisTdg);
  // print_tdg(&__kmp_global_tdgs[current_tdg_number]);
  // remove taskgroup from recorded tasks, not needed since taskwait is
  // called at the end of execute_tdg
  //erase_taskgroup();
  char *my_env_var = getenv("OMP_PRINT_TDG");
  if (my_env_var && strcmp(my_env_var, "TRUE") == 0) {
    // printf("[OpenMP] Dot file tdg.dot generated \n");
    print_tdg_to_dot(thisTdg);
  }

  //We have to clean the dephash after recording, to avoid conflicts
  kmp_info_t *thread = __kmp_threads[gtid];
  if(thread->th.th_current_task->td_dephash){
	 __kmp_dephash_free(thread, thread->th.th_current_task->td_dephash);
	 thread->th.th_current_task->td_dephash = NULL;
  }

  //Reset predecessor counter
  for (kmp_int32 i = 0; i < thisNumTasks; i++) {
      KMP_ATOMIC_ST_RLX(&thisRecordMap[i].npredecessors_counter, thisRecordMap[i].npredecessors);
  }

  return 1;
}

void freeTDG(int tdg_index, int gtid){  
   kmp_info_t *thread = __kmp_threads[gtid];

   for (kmp_int32 i = 0; i < KMP_ATOMIC_LD_RLX(&__kmp_global_tdgs[tdg_index].numTasks); i++) {
    free(__kmp_global_tdgs[tdg_index].recordMap[i].successors);
    kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(__kmp_global_tdgs[tdg_index].recordMap[i].task);
    taskdata->is_taskgraph = 0;
    taskdata->td_flags.executing = 0;
    taskdata->td_flags.complete = 1;
    taskdata->td_flags.freed = 0;
    __kmp_free_task_and_ancestors(gtid, taskdata, thread);
   }
   free(__kmp_global_tdgs[tdg_index].recordMap);
   free(__kmp_global_tdgs[tdg_index].rootTasks);
   free(__kmp_global_tdgs[tdg_index].taskIdent);
   free(__kmp_global_tdgs[tdg_index].colorMap);
}

void copyPreallocData(int gtid, kmp_int32 ThisMapSize, void *args,
                      kmp_tdg_info *tdg) {

   kmp_info_t *thread = __kmp_threads[gtid];
   kmp_taskdata_t *parent_task = thread->th.th_current_task;
   kmp_node_info *thisRecordMap = tdg->recordMap;
   for (kmp_int32 i = 0; i < ThisMapSize; i++) {
    thisRecordMap[i].parent_task = parent_task;
    int pragma = thisRecordMap[i].pragma_id;
    int *sharedPositions = tdg->task_static_table[pragma].sharedDataPositions;
    int numShareds =
        tdg->task_static_table[pragma].sizeOfShareds / sizeof(void *);
    for (int j = 0; j < numShareds; j++) {
      memcpy((char *)thisRecordMap[i].shared_data + j * sizeof(void *),
             (char *)args + sharedPositions[j] * sizeof(void *),
             sizeof(void *));
    }

    int *firstPrivatePositions =
        tdg->task_static_table[pragma].firstPrivateDataPositions;
    int *firstPrivateOffsets =
        tdg->task_static_table[pragma].firstPrivateDataOffsets;
    int *firstPrivateSizes =
        tdg->task_static_table[pragma].firstPrivateDataSizes;
    int numPrivates = tdg->task_static_table[pragma].numFirstPrivates;
    int currentPosition;
    int currentOffset;
    int currentSize;
    for (int j = 0; j < numPrivates; j++) {
      currentPosition = firstPrivatePositions[j];
      currentOffset = firstPrivateOffsets[j];
      currentSize = firstPrivateSizes[j];
      char ***adressValue =
          (char ***)((char *)args + currentPosition * sizeof(void *));
      memcpy((char *)thisRecordMap[i].private_data + currentOffset,
             *adressValue, currentSize);
    }

    int numGlobals = tdg->task_static_table[pragma].numGlobals;
    GlobalVarInfo *globals = tdg->task_static_table[pragma].GlobalVars;
    for (int j = 0; j < numGlobals; j++) {
      int currentOffset = globals[j].offset;
      int currentSize = globals[j].size;
      void *globalVar = globals[j].var;
      void **globalVarPointer = &globalVar;
      bool isPointer = globals[j].isPointer;
      memcpy((char *)thisRecordMap[i].private_data + currentOffset,
             (isPointer) ? globalVarPointer : globalVar, currentSize);
    }
   }
}

void __kmpc_taskgraph(ident_t *loc_ref, kmp_int32 gtid, kmp_uint32 tdg_id,
                      void (*entry)(void *), void *args, kmp_int32 if_cond,
                      kmp_int32 tdg_flags) {
  int tdg_index = -1;
  kmp_tdg_flags_t *input_flags = (kmp_tdg_flags_t *)&tdg_flags;
  int tdg_type = input_flags->static_tdg;
  int nowait = input_flags->nowait;
  int isSingleBasicBlock = input_flags->single_bb;
  // use input_flags->target_graph to build taskgraph with target tasks

  for (int i = 0; i < __kmp_ntdgs; i++) {
    if (__kmp_global_tdgs[i].tdgId == tdg_id) {
      tdg_index = i;
      if (strcmp(__kmp_global_tdgs[i].loc, "static") != 0) {
        // Check if the same tdg is already running, in this case we wait
        kmp_int32 remainingTasks =
            KMP_ATOMIC_LD_RLX(&__kmp_global_tdgs[i].remainingTasks);
        // Wait if this tdg is already running

        if (__kmp_global_tdgs[tdg_index].tdgStatus == KMP_TDG_PREALLOC) {
          copyPreallocData(gtid, __kmp_global_tdgs[tdg_index].mapSize, args, &__kmp_global_tdgs[tdg_index]);
        }

        if (remainingTasks > 0) {
          __kmpc_omp_taskwait(loc_ref, gtid);
        }

        // the TDG is ready to execute if:
        //   1. no re-record is needed
        //   2. the tdg is static and "entry" is a single basic block
        if(!if_cond || (tdg_type == KMP_STATIC_TDG && isSingleBasicBlock)) {
          if (!nowait)
            __kmpc_taskgroup(loc_ref, gtid);
            
          __kmpc_execute_tdg(loc_ref, gtid, tdg_index, nowait);

          if (!nowait) {
            kmp_info_t *thread = __kmp_threads[gtid];
            kmp_task_team_t *task_team = thread->th.th_task_team;
            kmp_int32 nthreads = 1;
            if(task_team)
              nthreads = task_team->tt.tt_nproc;
            __kmpc_end_taskgroup(loc_ref, gtid);
            wait_all_threads_scheduled(gtid, nthreads);
          }

          return;
        }
      }
    }
  }
  // we need to record the tdg if:
  //  1. the tdg is dynamic
  //  2. entry is not a single basic block
  if (tdg_type == KMP_DYNAMIC_TDG || !isSingleBasicBlock) {
    // printf("Recording! \n");
    if(tdg_index!=-1 && if_cond)
      freeTDG(tdg_index, gtid);
      
    __kmpc_record(loc_ref, gtid, entry, args, tdg_id, if_cond, nowait);
  } else if (tdg_type == KMP_STATIC_TDG) {
    if (!nowait)
      __kmpc_taskgroup(loc_ref, gtid);
    char *my_env_var = getenv("OMP_TASK_SCHEDULE");
    if (my_env_var && strcmp(my_env_var, "static") == 0) {
      __kmp_tdg_static_schedule = true;
    }

    // Update loc data
    __kmp_global_tdgs[tdg_index].loc = loc_ref->psource;
    kmp_int32 thisMapSize = __kmp_global_tdgs[tdg_index].mapSize;

    if (__kmp_global_tdgs[tdg_index].tdgStatus == KMP_TDG_PREALLOC) {
        copyPreallocData(gtid, thisMapSize, args, &__kmp_global_tdgs[tdg_index]);
    } else {
      __kmpc_fill_data(loc_ref, gtid, entry, args);
      // From KMP_TDG_FILL_DATA to KMP_TDG_NONE
      __kmp_global_tdgs[tdg_index].tdgStatus = KMP_TDG_NONE;
    }
    if (!nowait && !__kmp_tdg_static_schedule && enable_tdg_scheduling)
      distribute_tasks(gtid, tdg_index);

    __kmpc_execute_tdg(loc_ref, gtid, tdg_index, nowait);
    if (!nowait)
      __kmpc_end_taskgroup(loc_ref, gtid);
  } else {
    printf("internal OpenMP error: tdg_type not recognized\n");
  }
}

/*************      KMP WAITING TDG        ***********/
// This is a linked list to manage tasks that are waiting to a free task
// structure to be executed. Contains functions to insert and obtain tasks.

// Initializes the list
void init_waiting_tdg(int index) {
  __kmp_global_tdgs[index].waiting_tdg_to_execute.size = 0;
  __kmp_global_tdgs[index].waiting_tdg_to_execute.head = NULL;
  __kmp_global_tdgs[index].waiting_tdg_to_execute.tail = NULL;
  __kmp_init_futex_lock(&__kmp_global_tdgs[index].waiting_tdg_to_execute.waiting_tdg_lock);
}

// Returns the number of tasks waiting
int check_waiting_tdg(kmp_tdg_info *tdg) {
  __kmp_acquire_futex_lock(&tdg->waiting_tdg_to_execute.waiting_tdg_lock, 0);
  int size = tdg->waiting_tdg_to_execute.size;
  __kmp_release_futex_lock(&tdg->waiting_tdg_to_execute.waiting_tdg_lock, 0);
  return size;
}

// Insert a task at the begining of the list. Currently not used.
void insert_first_to_waiting_tdg(struct kmp_node_info *tdg_to_insert, kmp_tdg_info *tdg) {

  __kmp_acquire_futex_lock(&tdg->waiting_tdg_to_execute.waiting_tdg_lock, 0);
  tdg_to_insert->next_waiting_tdg = tdg->waiting_tdg_to_execute.head;
  tdg->waiting_tdg_to_execute.head = tdg_to_insert;
  tdg->waiting_tdg_to_execute.size += 1;
  __kmp_release_futex_lock(&tdg->waiting_tdg_to_execute.waiting_tdg_lock, 0);
}

// Insert a task at the end of the list. Used at launching points when no free
// task structure is available, at: 1) TDG execution of the roots 2) when a
// tasks finishes and queues their dependants
void insert_to_waiting_tdg(struct kmp_node_info *tdg_to_insert, kmp_tdg_info *tdg) {
  __kmp_acquire_futex_lock(&tdg->waiting_tdg_to_execute.waiting_tdg_lock, 0);
  if (!tdg->waiting_tdg_to_execute.size) {
    tdg->waiting_tdg_to_execute.head = tdg_to_insert;
    tdg->waiting_tdg_to_execute.tail = tdg_to_insert;
  } else {
    tdg->waiting_tdg_to_execute.tail->next_waiting_tdg = tdg_to_insert;
    tdg->waiting_tdg_to_execute.tail = tdg->waiting_tdg_to_execute.tail->next_waiting_tdg;
  }
  tdg->waiting_tdg_to_execute.size += 1;
  tdg->waiting_tdg_to_execute.tail->next_waiting_tdg = NULL;
  __kmp_release_futex_lock(&tdg->waiting_tdg_to_execute.waiting_tdg_lock, 0);
}

// Obtains a free task structure, returns NULL in case no task structure is
// available. Used when creating tasks: 1) When filling data for the first time
// 2) lazy task creation
struct kmp_node_info *get_from_waiting_tdg(kmp_tdg_info *tdg) {
  __kmp_acquire_futex_lock(&tdg->waiting_tdg_to_execute.waiting_tdg_lock, 0);
  struct kmp_node_info *temp;
  temp = tdg->waiting_tdg_to_execute.head;

  if (!tdg->waiting_tdg_to_execute.size) {
    __kmp_release_futex_lock(&tdg->waiting_tdg_to_execute.waiting_tdg_lock, 0);
    return NULL;
  } else {
    tdg->waiting_tdg_to_execute.head = tdg->waiting_tdg_to_execute.head->next_waiting_tdg;
    tdg->waiting_tdg_to_execute.size -= 1;
  }

  __kmp_release_futex_lock(&tdg->waiting_tdg_to_execute.waiting_tdg_lock, 0);
  return temp;
}

/************* 		FREE SPACE INDEXER		***********/
// This is a linked list to store the free task structures that are available at
// a moment, contains functions to insert and obtain tasks structures.

// Initializes the list
void init_free_space_indexer(int index) {
  __kmp_global_tdgs[index].free_space_indexer.head = NULL;
  __kmp_global_tdgs[index].free_space_indexer.tail = NULL;
  __kmp_global_tdgs[index]. free_space_indexer.n_free_tasks = 0;
  __kmp_global_tdgs[index].free_space_indexer.max_free_tasks = 0;
  __kmp_init_futex_lock(&__kmp_global_tdgs[index].free_space_indexer.lock);
}

// Returns the number of free task structures
int get_num_free_task(kmp_tdg_info *tdg) { return tdg->free_space_indexer.n_free_tasks; }

// Obtains a free node, otherwise returns NULL. Used at
// kmp_get_free_task_from_indexer function.
struct kmp_space_indexer_node *get_free_indexer_node(kmp_tdg_info *tdg) {
  __kmp_acquire_futex_lock(&tdg->free_space_indexer.lock, 0);

  if (!get_num_free_task(tdg)) {
    __kmp_release_futex_lock(&tdg->free_space_indexer.lock, 0);
    return NULL;
  }

  struct kmp_space_indexer_node *node;

  node = tdg->free_space_indexer.head;

  tdg->free_space_indexer.head = tdg->free_space_indexer.head->next;
  tdg->free_space_indexer.n_free_tasks -= 1;

  __kmp_release_futex_lock(&tdg->free_space_indexer.lock, 0);
  return node;
}

void insert_indexer_node(struct kmp_space_indexer_node *node, kmp_tdg_info *tdg) {
  __kmp_acquire_futex_lock(&tdg->free_space_indexer.lock, 0);
  node->next = NULL;
  if (!get_num_free_task(tdg)) {
    tdg->free_space_indexer.head = node;
    tdg->free_space_indexer.tail = node;
  } else {
    tdg->free_space_indexer.tail->next = node;
    tdg->free_space_indexer.tail =  tdg->free_space_indexer.tail->next;
  }
  tdg->free_space_indexer.n_free_tasks += 1;
  __kmp_release_futex_lock(&tdg->free_space_indexer.lock, 0);
}

kmp_task_t *kmp_get_free_task_from_indexer(kmp_tdg_info *tdg) {
  struct kmp_space_indexer_node *node = get_free_indexer_node(tdg);

  if (node == NULL) {
    return NULL;
  }

  kmp_task_t *task = node->task;
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  taskdata->indexer_node = node;
  node->next = NULL;
  return task;
}

void clean_free_task(kmp_task_t *task) {
  // kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  // TODO: Limpiar task + privates tambien, no solo el taskdata
  // memset(&taskdata, 0, sizeof(kmp_taskdata_t));
}

void kmp_insert_task_in_indexer(kmp_task_t *task) {
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);

  struct kmp_space_indexer_node *node = taskdata->indexer_node;
  clean_free_task(task);
  taskdata->indexer_node = node;
  insert_indexer_node(node, taskdata->tdg);
}

void __kmpc_prealloc_tasks(kmp_task_alloc_info *task_static_data,
                           char *preallocated_tasks,
                           kmp_space_indexer_node *preallocated_nodes,
                           unsigned int n_task_constructs,
                           unsigned int max_concurrent_tasks,
                           unsigned int task_size,
                           kmp_uint32 tdg_id) {

  int index = -1;
  //If the TDG is encountered preallocation has already been performed
  for (int i = 0; i < __kmp_ntdgs; i++) {
    if (__kmp_global_tdgs[i].tdgId == tdg_id){
        index = i;
        if(__kmp_global_tdgs[i].tdgStatus== KMP_TDG_PREALLOC){
          return;
        }
    }
  }

  __kmp_global_tdgs[index].tdgStatus = KMP_TDG_PREALLOC;

  if (max_concurrent_tasks == 0 || max_concurrent_tasks > n_task_constructs) {
    max_concurrent_tasks = n_task_constructs;
  }

  __kmp_global_tdgs[index].task_static_table = task_static_data;

  init_free_space_indexer(index);
  init_waiting_tdg(index);

  __kmp_global_tdgs[index].free_space_indexer.max_free_tasks = max_concurrent_tasks;

  // printf("Runtime es %d \n", max_arg_size);
  for (int i = 0; i < (int)max_concurrent_tasks; i++) {
    struct kmp_space_indexer_node *node = preallocated_nodes;
    preallocated_nodes++;

    kmp_taskdata_t *taskdata = (kmp_taskdata_t *)(preallocated_tasks);
    preallocated_tasks += task_size;

    kmp_task_t *task = KMP_TASKDATA_TO_TASK(taskdata);
    node->task = task;
    insert_indexer_node(node, &__kmp_global_tdgs[index]);
  }
}

kmp_task_t *kmp_init_lazy_task(int static_id,
                               kmp_int32 gtid, kmp_node_info *thisRecordMap, kmp_tdg_info *globalTdg) {

  kmp_task_t *new_task = kmp_get_free_task_from_indexer(globalTdg);
  if (new_task == NULL)
    return nullptr;
  
  kmp_uint32 tdg_id = globalTdg->tdgId;
  kmp_taskdata_t *new_taskdata = KMP_TASK_TO_TASKDATA(new_task);
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_node_info *tdg = &thisRecordMap[static_id];
  kmp_taskdata_t *parent_task = tdg->parent_task;
  kmp_team_t *team = thread->th.th_team;
  kmp_task_alloc_info tdg_static_data = globalTdg->task_static_table[tdg->pragma_id];
  kmp_tasking_flags_t *flags = (kmp_tasking_flags_t *)&tdg_static_data.flags;

  int tdg_index = -1;
  for (int i = 0; i < __kmp_ntdgs; i++) {
    if (__kmp_global_tdgs[i].tdgId == tdg_id) {
      tdg_index=i;
    }
  }

  if (tdg_static_data.sizeOfShareds >= 4) {

    int shareds_offset = sizeof(kmp_taskdata_t) + tdg_static_data.sizeOfTask;
    shareds_offset = __kmp_round_up_to_val(shareds_offset, sizeof(void *));
    // Avoid double allocation here by combining shareds with taskdata
    new_task->shareds = &((char *)new_taskdata)[shareds_offset];
    // Make sure shareds struct is aligned to pointer size
    KMP_DEBUG_ASSERT(
        (((kmp_uintptr_t)new_task->shareds) & (sizeof(void *) - 1)) == 0);
  } else {
    new_task->shareds = NULL;
  }

  new_task->routine = tdg_static_data.taskEntry;

  new_taskdata->is_taskgraph = 1;
  new_taskdata->tdg = &__kmp_global_tdgs[tdg_index];
  new_taskdata->groupID = 0;
  new_taskdata->td_task_id = tdg->static_id;
  new_taskdata->td_team = thread->th.th_team;
  new_taskdata->td_alloc_thread = thread;
  new_taskdata->td_parent = parent_task;
  new_taskdata->td_level = parent_task->td_level + 1; // increment nesting level
  KMP_ATOMIC_ST_RLX(&new_taskdata->td_untied_count, 0);

  new_taskdata->td_ident = NULL;
  new_taskdata->td_taskwait_ident = NULL;
  new_taskdata->td_taskwait_counter = 0;
  new_taskdata->td_taskwait_thread = 0;
  KMP_DEBUG_ASSERT(new_taskdata->td_parent != NULL);
  // avoid copying icvs for proxy tasks
  if (flags->proxy == TASK_FULL)
    copy_icvs(&new_taskdata->td_icvs, &new_taskdata->td_parent->td_icvs);

  new_taskdata->td_flags.tiedness = flags->tiedness;
  new_taskdata->td_flags.final = flags->final;
  new_taskdata->td_flags.merged_if0 = flags->merged_if0;
  new_taskdata->td_flags.destructors_thunk = flags->destructors_thunk;
  new_taskdata->td_flags.proxy = flags->proxy;
  new_taskdata->td_flags.detachable = flags->detachable;
  new_taskdata->td_flags.hidden_helper = flags->hidden_helper;
  new_taskdata->td_task_team = thread->th.th_task_team;
  new_taskdata->td_size_alloc = tdg_static_data.sizeOfTask +
                                tdg_static_data.sizeOfShareds +
                                sizeof(kmp_taskdata_t);
  new_taskdata->td_flags.tasktype = TASK_EXPLICIT;

  new_taskdata->td_flags.tasking_ser =
      (__kmp_tasking_mode == tskm_immediate_exec);

  // GEH - TODO: fix this to copy parent task's value of team_serial flag
  new_taskdata->td_flags.team_serial = (team->t.t_serialized) ? 1 : 0;

  // GEH - Note we serialize the task if the team is serialized to make sure
  // implicit parallel region tasks are not left until program termination to
  // execute. Also, it helps locality to execute immediately.

  new_taskdata->td_flags.task_serial =
      (parent_task->td_flags.final || new_taskdata->td_flags.team_serial ||
       new_taskdata->td_flags.tasking_ser || flags->merged_if0);

  new_taskdata->td_flags.started = 0;
  new_taskdata->td_flags.executing = 0;
  new_taskdata->td_flags.complete = 0;
  new_taskdata->td_flags.freed = 0;

  new_taskdata->td_flags.native = flags->native;

  KMP_ATOMIC_ST_RLX(&new_taskdata->td_incomplete_child_tasks, 0);
  // start at one because counts current task and children
  KMP_ATOMIC_ST_RLX(&new_taskdata->td_allocated_child_tasks, 1);
  new_taskdata->td_taskgroup =
      parent_task->td_taskgroup; // task inherits taskgroup from the parent task
  new_taskdata->td_dephash = NULL;
  new_taskdata->td_depnode = NULL;

  if (flags->tiedness == TASK_UNTIED)
    new_taskdata->td_last_tied = NULL; // will be set when the task is scheduled
  else
    new_taskdata->td_last_tied = new_taskdata;

  // Initialize counter to 1 event, the task finish event
  new_taskdata->td_allow_completion_event.pending_events_count = 1;
  new_taskdata->td_allow_completion_event.ed.task = new_task;

  if((tdg_static_data.sizeOfTask - sizeof(kmp_task_t)) > 0)
    memcpy(new_task + 1, tdg->private_data,
         tdg_static_data.sizeOfTask - sizeof(kmp_task_t));
  if(tdg_static_data.sizeOfShareds>=4)
    memcpy(new_task->shareds, tdg->shared_data, tdg_static_data.sizeOfShareds);

  tdg->task = new_task;
  return new_task;
}
#endif

/*!
@ingroup TASKING
@param loc_ref location of the original task directive
@param gtid Global Thread ID of encountering thread
@param ndeps Number of depend items with possible aliasing
@param dep_list List of depend items with possible aliasing
@param ndeps_noalias Number of depend items with no aliasing
@param noalias_dep_list List of depend items with no aliasing

Blocks the current task until all specifies dependences have been fulfilled.
*/
void __kmpc_omp_wait_deps(ident_t *loc_ref, kmp_int32 gtid, kmp_int32 ndeps,
                          kmp_depend_info_t *dep_list, kmp_int32 ndeps_noalias,
                          kmp_depend_info_t *noalias_dep_list) {
  __kmpc_omp_taskwait_deps_51(loc_ref, gtid, ndeps, dep_list, ndeps_noalias,
                              noalias_dep_list, false);
}

/* __kmpc_omp_taskwait_deps_51 : Function for OpenMP 5.1 nowait clause.
                                 Placeholder for taskwait with nowait clause.
                                 Earlier code of __kmpc_omp_wait_deps() is now
                                 in this function.
*/
void __kmpc_omp_taskwait_deps_51(ident_t *loc_ref, kmp_int32 gtid,
                                 kmp_int32 ndeps, kmp_depend_info_t *dep_list,
                                 kmp_int32 ndeps_noalias,
                                 kmp_depend_info_t *noalias_dep_list,
                                 kmp_int32 has_no_wait) {
  KA_TRACE(10, ("__kmpc_omp_taskwait_deps(enter): T#%d loc=%p nowait#%d\n",
                gtid, loc_ref, has_no_wait));
  if (ndeps == 0 && ndeps_noalias == 0) {
    KA_TRACE(10, ("__kmpc_omp_taskwait_deps(exit): T#%d has no dependences to "
                  "wait upon : loc=%p\n",
                  gtid, loc_ref));
    return;
  }
  __kmp_assert_valid_gtid(gtid);
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *current_task = thread->th.th_current_task;

#if OMPT_SUPPORT
  // this function represents a taskwait construct with depend clause
  // We signal 4 events:
  //  - creation of the taskwait task
  //  - dependences of the taskwait task
  //  - schedule and finish of the taskwait task
  ompt_data_t *taskwait_task_data = &thread->th.ompt_thread_info.task_data;
  KMP_ASSERT(taskwait_task_data->ptr == NULL);
  if (ompt_enabled.enabled) {
    if (!current_task->ompt_task_info.frame.enter_frame.ptr)
      current_task->ompt_task_info.frame.enter_frame.ptr =
          OMPT_GET_FRAME_ADDRESS(0);
    if (ompt_enabled.ompt_callback_task_create) {
      ompt_callbacks.ompt_callback(ompt_callback_task_create)(
          &(current_task->ompt_task_info.task_data),
          &(current_task->ompt_task_info.frame), taskwait_task_data,
          ompt_task_taskwait | ompt_task_undeferred | ompt_task_mergeable, 1,
          OMPT_LOAD_OR_GET_RETURN_ADDRESS(gtid));
    }
  }

#if OMPT_OPTIONAL
  /* OMPT grab all dependences if requested by the tool */
  if (ndeps + ndeps_noalias > 0 && ompt_enabled.ompt_callback_dependences) {
    kmp_int32 i;

    int ompt_ndeps = ndeps + ndeps_noalias;
    ompt_dependence_t *ompt_deps = (ompt_dependence_t *)KMP_OMPT_DEPS_ALLOC(
        thread, (ndeps + ndeps_noalias) * sizeof(ompt_dependence_t));

    KMP_ASSERT(ompt_deps != NULL);

    for (i = 0; i < ndeps; i++) {
      ompt_deps[i].variable.ptr = (void *)dep_list[i].base_addr;
      if (dep_list[i].flags.in && dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_inout;
      else if (dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_out;
      else if (dep_list[i].flags.in)
        ompt_deps[i].dependence_type = ompt_dependence_type_in;
      else if (dep_list[i].flags.mtx)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_mutexinoutset;
      else if (dep_list[i].flags.set)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inoutset;
    }
    for (i = 0; i < ndeps_noalias; i++) {
      ompt_deps[ndeps + i].variable.ptr = (void *)noalias_dep_list[i].base_addr;
      if (noalias_dep_list[i].flags.in && noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inout;
      else if (noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_out;
      else if (noalias_dep_list[i].flags.in)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_in;
      else if (noalias_dep_list[i].flags.mtx)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_mutexinoutset;
      else if (noalias_dep_list[i].flags.set)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inoutset;
    }
    ompt_callbacks.ompt_callback(ompt_callback_dependences)(
        taskwait_task_data, ompt_deps, ompt_ndeps);
    /* We can now free the allocated memory for the dependences */
    /* For OMPD we might want to delay the free until end of this function */
    KMP_OMPT_DEPS_FREE(thread, ompt_deps);
    ompt_deps = NULL;
  }
#endif /* OMPT_OPTIONAL */
#endif /* OMPT_SUPPORT */

  // We can return immediately as:
  // - dependences are not computed in serial teams (except with proxy tasks)
  // - if the dephash is not yet created it means we have nothing to wait for
  bool ignore = current_task->td_flags.team_serial ||
                current_task->td_flags.tasking_ser ||
                current_task->td_flags.final;
  ignore =
      ignore && thread->th.th_task_team != NULL &&
      thread->th.th_task_team->tt.tt_found_proxy_tasks == FALSE &&
      thread->th.th_task_team->tt.tt_hidden_helper_task_encountered == FALSE;
  ignore = ignore || current_task->td_dephash == NULL;

  if (ignore) {
    KA_TRACE(10, ("__kmpc_omp_taskwait_deps(exit): T#%d has no blocking "
                  "dependences : loc=%p\n",
                  gtid, loc_ref));
#if OMPT_SUPPORT
    __ompt_taskwait_dep_finish(current_task, taskwait_task_data);
#endif /* OMPT_SUPPORT */
    return;
  }

  kmp_depnode_t node = {0};
  __kmp_init_node(&node);

  if (!__kmp_check_deps(gtid, &node, NULL, &current_task->td_dephash,
                        DEP_BARRIER, ndeps, dep_list, ndeps_noalias,
                        noalias_dep_list)) {
    KA_TRACE(10, ("__kmpc_omp_taskwait_deps(exit): T#%d has no blocking "
                  "dependences : loc=%p\n",
                  gtid, loc_ref));
#if OMPT_SUPPORT
    __ompt_taskwait_dep_finish(current_task, taskwait_task_data);
#endif /* OMPT_SUPPORT */
    return;
  }

  int thread_finished = FALSE;
  kmp_flag_32<false, false> flag(
      (std::atomic<kmp_uint32> *)&node.dn.npredecessors, 0U);
  while (node.dn.npredecessors > 0) {
    flag.execute_tasks(thread, gtid, FALSE,
                       &thread_finished USE_ITT_BUILD_ARG(NULL),
                       __kmp_task_stealing_constraint);
  }

#if OMPT_SUPPORT
  __ompt_taskwait_dep_finish(current_task, taskwait_task_data);
#endif /* OMPT_SUPPORT */
  KA_TRACE(10, ("__kmpc_omp_taskwait_deps(exit): T#%d finished waiting : loc=%p\
                \n",
                gtid, loc_ref));
}
