//! ## Problem brief
//!
//! We need to insert halt checks often enough that execution cannot run
//! forever without giving the runtime a chance to stop it.
//!
//! The basic invariant is counted in safepoints, not in blocks or
//! instructions. If `halt_check_every = N`, then after every `N` safepoints
//! on any executable path, the next safepoint must be followed by a halt
//! check.
//!
//! There are two separate problems:
//!
//! 1. **Acyclic path problem**: on a DAG, make sure long paths with many
//!    safepoints get periodic halt checks.
//! 2. **Cycle problem**: on a general directed CFG, make sure no directed
//!    cycle can run forever without executing a halt check.
//!
//! The DAG problem can be solved with a forward countdown state.
//!
//! The cycle problem needs an additional structural invariant: every directed
//! cycle must contain at least one safepoint. Once that is true, we can force
//! halt checks inside cyclic SCCs.
//!
//! ---
//!
//! ## Invariants
//!
//! `halt_check_in = r` means:
//!
//! - after seeing `r` more safepoints, the next safepoint must be followed by
//!   a halt check.
//!
//! Equivalently, if `N = halt_check_every`, then this path has seen
//! `N - r` safepoints since the previous halt check.
//!
//! The state carried across edges is:
//!
//! ```nocompile_test
//! struct HaltState {
//!     remaining: NonZero<usize>,
//!
//!     // Safepoints since the previous halt check on this edge/path.
//!     //
//!     // This must be edge/path-local. It may contain safepoints from multiple
//!     // blocks, not only the immediate predecessor block.
//!     suffix_safepoints: Queue<SafepointLoc>,
//! }
//! ````
//!
//! When processing a safepoint:
//!
//! ```nocompile_test
//! suffix_safepoints.push(loc);
//! remaining -= 1;
//!
//! if remaining == 0 {
//!     insert_halt_check_after(loc);
//!     suffix_safepoints.clear();
//!     remaining = N;
//! }
//! ```
//!
//! At a branch, clone the state to each successor.
//!
//! ---
//!
//! ## DAG solution
//!
//! For a DAG, process blocks in topological order.
//!
//! For merges, the simplest correct rule is `min`.
//!
//! ```nocompile_test
//! halt_check_in_at_merge = predecessors.iter().map(|predecessor| {
//!     predecessor.halt_check_out
//! }).min().unwrap();
//! ```
//!
//! Why?
//!
//! If one predecessor arrives with:
//!
//! ```text
//! block1 out = 8
//! block2 out = 3
//! ```
//!
//! then choosing `3` means the continuation may insert a halt check earlier
//! on the `block1` path than strictly necessary, but it will never insert one
//! too late.
//!
//! That gives a clean first implementation:
//!
//! ```nocompile_test
//! entry halt_check_in = N
//!
//! for block in topo_order {
//!     let halt_check_in = if block == ENTRYPOINT {
//!         N
//!     } else {
//!         min(pred.halt_check_out for pred in preds(block))
//!     };
//!
//!     let halt_check_out = process_block(block, halt_check_in);
//!
//!     for succ in succs(block) {
//!         edge_state[block -> succ] = halt_check_out;
//!     }
//! }
//! ```
//!
//! This is conservative, deterministic, and easy to prove correct.
//!
//! ---
//!
//! ## Merge optimization for large countdown differences
//!
//! The `min` rule is always safe, but can be conservative.
//!
//! Suppose a merge has two incoming countdowns:
//!
//! ```text
//! r_hi = max incoming remaining
//! r_lo = min incoming remaining
//! ```
//!
//! where:
//!
//! ```text
//! r_hi > r_lo
//! ```
//!
//! If we want to normalize the `r_lo` edge up to `r_hi`, then we insert a
//! halt check on the `r_lo` path before the merge.
//!
//! The correct location is:
//!
//! ```text
//! after the safepoint that has exactly N - r_hi safepoints after it before the merge
//! ```
//!
//! Equivalently:
//!
//! ```text
//! after the (r_hi - r_lo)-th safepoint since the previous halt check
//! ```
//!
//! So if we keep a suffix list of safepoints since the last halt check, the
//! insertion point is:
//!
//! ```nocompile_test
//! let target = r_hi;
//!
//! let safepoints_after_insert = N - target;
//!
//! let insertion_safepoint = suffix_safepoints
//!     .iter()
//!     .nth_back(safepoints_after_insert);
//! ```
//!
//! This optimization is only valid when the edge has a complete path-local
//! unchecked suffix. If suffix precision was lost at a previous merge, or if
//! the needed safepoint is not present in the suffix, fall back to `min`.
//!
//! ---
//!
//! ## Why the insertion point may not be in the immediate predecessor
//!
//! Let:
//!
//! ```text
//! N = 10
//! ```
//!
//! CFG:
//!
//! ```text
//! entry:
//!     br cond A B1
//!
//! A:
//!     safepoint
//!     safepoint
//!     safepoint
//!     br M
//!
//! B1:
//!     safepoint
//!     safepoint
//!     safepoint
//!     safepoint
//!     safepoint
//!     safepoint
//!     safepoint
//!     br B2
//!
//! B2:
//!     safepoint
//!     safepoint
//!     br M
//!
//! M:
//!     ...
//! ```
//!
//! Assume both branches start with:
//!
//! ```text
//! halt_check_in = 10
//! ```
//!
//! Then:
//!
//! ```text
//! A has 3 safepoints => remaining = 7
//! B has 9 safepoints => remaining = 1
//! ```
//!
//! At merge `M`:
//!
//! ```text
//! r_hi = 7
//! r_lo = 1
//! diff = 6
//! ```
//!
//! If we normalize the `B` path to `7`, we need to insert a halt check so
//! that there are:
//!
//! ```text
//! N - r_hi = 10 - 7 = 3
//! ```
//!
//! safepoints after the inserted check before the merge.
//!
//! But `B2` contains only 2 safepoints.
//!
//! So the correct insertion point is in `B1`, not in the immediate
//! predecessor `B2`.
//!
//! Specifically, the `B` suffix has 9 safepoints total:
//!
//! ```text
//! B1: s1 s2 s3 s4 s5 s6 s7
//! B2: s8 s9
//! ```
//!
//! We need 3 safepoints after the inserted check, so we insert after `s6`:
//!
//! ```text
//! B1: s1 s2 s3 s4 s5 s6 HALT_CHECK s7
//! B2: s8 s9
//! ```
//!
//! Then after the inserted halt check, the path sees 3 safepoints before the
//! merge, so it arrives with:
//!
//! ```text
//! 10 - 3 = 7
//! ```
//!
//! matching the `A` path.
//!
//! Therefore the immediate predecessor block is not enough. The optimization
//! needs path-local suffix history.
//!
//! ---
//!
//! ## General directed graph solution
//!
//! A general CFG may contain cycles, so ordinary block topological order is
//! not enough.
//!
//! The solution is to work over strongly connected components.
//!
//! ### Step 1: Validate safepoint coverage of cycles
//!
//! Every directed cycle must contain at least one safepoint.
//!
//! Equivalently:
//!
//! ```text
//! the subgraph induced by blocks with no safepoints must be acyclic
//! ```
//!
//! Why this equivalence holds:
//!
//! * If there is a directed cycle with no safepoint, every block on that cycle
//!   is safepoint-free, so the safepoint-free subgraph contains a cycle.
//! * If the safepoint-free subgraph contains a cycle, then the original graph
//!   has a directed cycle with no safepoint.
//!
//! So the pass should compute SCCs of the safepoint-free subgraph. If any SCC
//! is cyclic, the IR invariant is violated. Either an earlier pass must insert
//! a safepoint into that cycle, or this pass must fail loudly.
//!
//! ### Step 2: Condense SCCs into a DAG
//!
//! Compute SCCs of the full CFG.
//!
//! The SCC condensation graph is always a DAG, even if the original CFG has
//! cycles.
//!
//! Process SCCs in topological order.
//!
//! ### Step 3: Process acyclic SCCs with the DAG countdown logic
//!
//! An acyclic SCC is a single block with no self-loop.
//!
//! For these components, use the ordinary `HaltState` transfer:
//!
//! ```nocompile_test
//! halt_state_in = merge predecessor states
//! halt_state_out = process_block(block, halt_state_in)
//! ```
//!
//! This preserves the existing DAG behavior.
//!
//! ### Step 4: Process cyclic SCCs by forcing checks after safepoints
//!
//! For a cyclic SCC, we already know every cycle contains at least one
//! safepoint.
//!
//! Therefore a simple safe rule is:
//!
//! ```text
//! insert a halt check after every safepoint in the cyclic SCC
//! ```
//!
//! This guarantees every directed cycle in the SCC contains at least one halt
//! check.
//!
//! This may insert more halt checks than the theoretical minimum, but it is
//! simple, local, and sound. Computing a smaller set of safepoints that hits
//! every cycle is a feedback-vertex-style optimization and should not be the
//! first implementation.
//!
//! ### Step 5: Conservatively summarize cyclic SCC exits
//!
//! After forcing halt checks inside a cyclic SCC:
//!
//! * paths that hit a safepoint reset their countdown to `N`;
//! * paths that do not hit a safepoint preserve the incoming countdown.
//!
//! Since every `remaining` value is `<= N`, the safe summary for every exit is
//! the incoming merged countdown with suffix history discarded:
//!
//! ```nocompile_test
//! halt_state_out = HaltState {
//!     remaining: halt_state_in.remaining,
//!     suffix_safepoints: Queue::new(),
//! };
//! ```
//!
//! Discarding suffix precision is important because inside a cyclic SCC there
//! may be many possible paths. Keeping one concrete suffix would be unsound.
//!
//! ---
//!
//! ## Final strategy
//!
//! For arbitrary directed graphs:
//!
//! ```nocompile_test
//! assert every cycle has at least one safepoint
//!
//! compute SCCs
//! compute SCC condensation DAG
//!
//! for component in scc_topological_order {
//!     let halt_state_in = merge external predecessor states
//!
//!     if component is acyclic {
//!         process the single block with HaltState
//!         propagate halt_state_out to successors
//!     } else {
//!         insert a halt check after every safepoint in the component
//!
//!         // Conservative SCC summary.
//!         halt_state_out = HaltState::from_remaining(halt_state_in.remaining)
//!
//!         propagate halt_state_out to successors outside the component
//!     }
//! }
//! ```
//!
//! This gives:
//!
//! 1. the precise DAG behavior for acyclic code;
//! 2. safe periodic halt checks along long acyclic paths;
//! 3. guaranteed halt checks inside every cycle;
//! 4. no fixed-point countdown analysis inside loops;
//! 5. no unsound use of suffix history across cyclic control flow.

use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZero;
use std::ops::DerefMut;
use smallvec::SmallVec;
use crate::ir::{Block, ExecIrBuilder, Stmt, StmtKind};
use crate::ir::arena::{handle_impl_helper, make_handle, Arena, ArenaMap, ArenaSet, Storable};

/// represents the state at the end of one path into the merge.
/// After seeing `r` more safepoints, the next safepoint forces a halt check.
/// Equivalently, if:
///
/// N = halt_check_every
///
/// then this path has already seen:
///
/// N - r
///
/// safepoints since the previous halt check.
#[derive(Clone)]
struct HaltState {
    remaining: NonZero<usize>,
    suffix_safepoints: rpds::Queue<Stmt>
}

struct HaltStateMap {
    outgoing_edges: ArenaMap<Block, HashMap<Block, HaltState>>,
    incoming_edges: ArenaMap<Block, HashSet<Block>>,
}

impl HaltStateMap {
    pub fn new(ir: &ExecIrBuilder) -> Self {
        let edge_capacity = ir.blocks.len().div_ceil(2).saturating_mul(3);
        let forward_edges = ArenaMap::with_capacity(edge_capacity);
        let backward_edges = ArenaMap::with_capacity(edge_capacity);

        Self {
            outgoing_edges: forward_edges,
            incoming_edges: backward_edges,
        }
    }

    pub fn add_edge(&mut self, from: Block, to: Block, state: HaltState) -> Option<HaltState> {
        let edges = {
            self.outgoing_edges.get_or_insert_with(from, || HashMap::with_capacity(1))
        };

        let backward_edges = {
            self.incoming_edges.get_or_insert_with(to, || HashSet::with_capacity(1))
        };

        backward_edges.insert(from);
        edges.insert(to, state)
    }

    pub fn drain_incoming(
        &mut self,
        towards: Block
    ) -> impl Iterator<Item=(Block, HaltState)> + use<'_> {
        // its .map() -> .flatten()
        // because flat_map() complains about the borrow checker
        self
            .incoming_edges
            .get_mut(towards)
            .map(|incoming_edges| {
                DrainHaltState {
                    to: towards,
                    outgoing_edges: &mut self.outgoing_edges,
                    drain: incoming_edges.drain(),
                }
            })
            .into_iter()
            .flatten()
    }
}

struct DrainHaltState<'a> {
    to: Block,
    outgoing_edges: &'a mut ArenaMap<Block, HashMap<Block, HaltState>>,
    drain: std::collections::hash_set::Drain<'a, Block>,
}

impl Iterator for DrainHaltState<'_> {
    type Item = (Block, HaltState);

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next().map(|from| {
            let to = self.to;
            let edge_must_exist = "edge must exist in forward map if it exists in backward map";
            let state = self.outgoing_edges
                .get_mut(from)
                .and_then(|map| map.remove(&to))
                .expect(edge_must_exist);

            (from, state)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.drain.size_hint()
    }
}

impl Drop for DrainHaltState<'_> {
    fn drop(&mut self) {
        for _ in self {
            // run next to completion
        }
    }
}


struct HaltCheckInserter<'a> {
    ir: &'a mut ExecIrBuilder,
    // note: this is deliberately a hashmap
    //       because there aren't that many
    //       safepoints compared to other types of stmts
    safepoint_stmt_to_block_and_index: HashMap<Stmt, (Block, usize)>,
    map: HaltStateMap,

    // Safepoints after which this pass inserted a compensating halt check.
    // note this is delibaretly a hashset because a small portion of statements are safepoints
    // and an even smaller portion affects resets
    // this makes safepoint edits lazy,
    // and that means we can skip repeated expensive global scans to the HaltState
    semantic_reset_after: HashSet<Stmt>,
}

impl<'a> HaltCheckInserter<'a> {
    pub fn new(halt_check_every: NonZero<usize>, ir: &'a mut ExecIrBuilder) -> Self {
        let safepoint_stmt_est = ir.stmts.len().div_ceil(128);
        let mut safepoints = HashMap::with_capacity(safepoint_stmt_est);

        for (block, data) in ir.blocks.iter() {
            for (i, stmt) in data.stmts.iter().copied().enumerate() {
                if let StmtKind::Safepoint = ir.stmts[stmt].rvalue {
                    let old_pos = safepoints.insert(stmt, (block, i));
                    assert!(old_pos.is_none());
                }
            }
        }

        let map = HaltStateMap::new(ir);

        Self {
            ir,
            safepoint_stmt_to_block_and_index: safepoints,
            map,
            semantic_reset_after: HashSet::with_capacity(
                (safepoint_stmt_est / halt_check_every).div_ceil(2)
            )
        }
    }

    pub fn ir(&mut self) -> &mut ExecIrBuilder {
        self.ir
    }

    pub fn insert_halt_check_after_safepoint_indexed(
        &mut self,
        block: Block,
        stmt_index: usize,
    ) -> Block {
        let safepoints = &mut self.safepoint_stmt_to_block_and_index;

        let continuation = self.ir.insert_halt_check_at(block, stmt_index.strict_add(1));
        for (i, &stmt) in self.ir.blocks[continuation].stmts.iter().enumerate() {
            if let StmtKind::Safepoint = self.ir.stmts[stmt].rvalue {
                let old = safepoints.insert(stmt, (continuation, i));
                assert!(old.is_some_and(|(old_block, old_idx)| {
                    old_block == block && old_idx > stmt_index
                }))
            }
        }

        // since the edges are (from -> to) where HaltState is the state of things after
        // from runs to completion; when splitting a node, to is unaffected.
        // since it will always exist, and the edge from -> to always means
        // the end of `from` jumps towards `to`
        // and so we need to remap anything `from` maps to and place it in `continuation`
        if let Some(edges) = self.map.outgoing_edges.remove(block) {
            let insert_continuation_err =
                "continuation is a fresh block, and can't have any existing edges";

            // remap incoming
            for &to in edges.keys() {
                let set = self.map.incoming_edges.get_mut(to).unwrap();
                let removed = set.remove(&block);
                assert!(
                    removed,
                    "if there is a forward edge from block -> to; then to -> block must exist"
                );
                let inserted = set.insert(continuation);
                assert!(inserted, "{insert_continuation_err}");
            }

            // remap outgoing
            let old_edges = self.map.outgoing_edges.insert(continuation, edges);
            assert!(
                old_edges.is_none(),
                "{insert_continuation_err}"
            );
        }


        continuation
    }

    pub fn insert_compensating_halt_check_after_safepoint(
        &mut self,
        at: Stmt,
    ) -> Block {
        let inserted = self.semantic_reset_after.insert(at);

        assert!(
            inserted,
            "attempted to insert two compensating halt checks after the same safepoint"
        );

        let (block, stmt_index) = self.safepoint_stmt_to_block_and_index[&at];

        self.insert_halt_check_after_safepoint_indexed(block, stmt_index)
    }
}


trait BlockHalter {
    fn process_safepoint(&mut self, halt_check_every: NonZero<usize>, safepoint: Stmt) -> bool;

    fn take_current_state(&mut self) -> HaltState;

    fn process_block(
        &mut self,
        inserter: &mut HaltCheckInserter,
        halt_check_every: NonZero<usize>,
        mut block: Block
    ) -> (Block, HaltState) {
        'break_down_loop: loop {
            let ir = inserter.ir();

            let mut split = None;
            for (i, &stmt) in ir.blocks[block].stmts.iter().enumerate() {
                if let StmtKind::Safepoint = ir.stmts[stmt].rvalue {
                    let should_halt = self.process_safepoint(halt_check_every, stmt);
                    if should_halt {
                        split = Some(i);
                        break
                    }
                }
            }

            let Some(pos) = split else {
                break 'break_down_loop
            };

            block = inserter.insert_halt_check_after_safepoint_indexed(block, pos);
        }

        let state_after_block = self.take_current_state();

        (block, state_after_block)
    }
}

struct CyclicBlockHalter {
    halt_check_every: Option<NonZero<usize>>,
    state_in_remaining: NonZero<usize>,
}

impl CyclicBlockHalter {
    pub fn new(state_in: HaltState) -> Self {
        Self {
            halt_check_every: None,
            state_in_remaining: state_in.remaining
        }
    }
}

impl BlockHalter for CyclicBlockHalter {
    #[inline]
    fn process_safepoint(&mut self, halt_check_every: NonZero<usize>, _: Stmt) -> bool {
        self.halt_check_every = Some(halt_check_every);
        true
    }

    fn take_current_state(&mut self) -> HaltState {
        let remaining = self
            .halt_check_every
            .take()
            .unwrap_or(self.state_in_remaining);

        HaltState::from_remaining(remaining)
    }
}

struct ACyclicBlockHalter(HaltState);

impl ACyclicBlockHalter {
    fn new(state_in: HaltState) -> Self {
        Self(state_in)
    }
}

impl BlockHalter for ACyclicBlockHalter {
    fn process_safepoint(
        &mut self,
        halt_check_every: NonZero<usize>,
        safepoint: Stmt
    ) -> bool {
        let state = &mut self.0;

        let new_remaining = NonZero::new(state.remaining.get().strict_sub(1));
        state.remaining = new_remaining.unwrap_or(halt_check_every);

        let should_insert_halt_check = new_remaining.is_none();

        match should_insert_halt_check {
            // There is never a need to keep suffix history before this point,
            // because after inserting a halt check the countdown resets to N.
            true => state.suffix_safepoints = rpds::Queue::new(),
            false => {
                state.suffix_safepoints.enqueue_mut(safepoint);
                if state.suffix_safepoints.len() > halt_check_every.get() {
                    assert!(state.suffix_safepoints.dequeue_mut());
                }
            }
        }

        should_insert_halt_check
    }

    fn take_current_state(&mut self) -> HaltState {
        self.0.clone()
    }
}

impl HaltState {
    fn from_remaining(remaining: NonZero<usize>) -> Self {
        Self {
            remaining,
            suffix_safepoints: rpds::Queue::new(),
        }
    }

    fn has_complete_suffix(&self, halt_check_every: NonZero<usize>) -> bool {
        self.suffix_safepoints.len()
            == halt_check_every.get().strict_sub(self.remaining.get())
    }

    // canonicalization can only increase remaining, never decrease it.
    fn canonicalize_against_resets(
        &mut self,
        halt_check_every: NonZero<usize>,
        reset_after: &HashSet<Stmt>,
    ) {
        if self.suffix_safepoints.is_empty() || reset_after.is_empty() {
            return;
        }

        let mut found_reset = false;
        let mut new_suffix = rpds::Queue::new();

        for stmt in self.suffix_safepoints.iter().copied() {
            if reset_after.contains(&stmt) {
                found_reset = true;

                // This reset becomes the latest known reset so far.
                // Anything before it, including the old tentative suffix after an
                // earlier reset, is no longer relevant.
                new_suffix = rpds::Queue::new();
            } else if found_reset {
                new_suffix.enqueue_mut(stmt);
            }
        }

        if !found_reset {
            return;
        }

        let after_count = new_suffix.len();

        debug_assert!(
            after_count < halt_check_every.get(),
            "unchecked suffix after canonicalization should be shorter than halt_check_every"
        );

        debug_assert!(after_count < self.suffix_safepoints.len());

        let remaining = halt_check_every.get().strict_sub(after_count);
        let new_remaining = NonZero::new(remaining)
            .expect("after_count must be strictly less than halt_check_every");

        self.remaining = new_remaining;
        self.suffix_safepoints = new_suffix;
    }

    fn has_insertion_point(
        &self,
        halt_check_every: NonZero<usize>,
        target: NonZero<usize>,
    ) -> bool {
        if self.remaining >= target {
            return false;
        }

        if !self.has_complete_suffix(halt_check_every) {
            return false;
        }

        let safepoints_after_insert = halt_check_every.get().strict_sub(target.get());

        if self.suffix_safepoints.len() <= safepoints_after_insert {
            return false;
        }

        true
    }

    fn compensating_insertion_safepoint(
        &self,
        halt_check_every: NonZero<usize>,
        target: NonZero<usize>,
    ) -> Stmt {
        assert!(self.has_insertion_point(halt_check_every, target));

        let safepoints_after_insert =
            halt_check_every.get().strict_sub(target.get());

        let skip_from_front = self
            .suffix_safepoints
            .len()
            .strict_sub(safepoints_after_insert)
            .strict_sub(1);

        self.suffix_safepoints
            .iter()
            .copied()
            .nth(skip_from_front)
            .unwrap()
    }

    // Precondition: self has already been canonicalized against
    // inserter.semantic_reset_after.
    fn insert_compensating_check(
        &mut self,
        inserter: &mut HaltCheckInserter,
        halt_check_every: NonZero<usize>,
        target: NonZero<usize>,
    ) {
        if self.remaining >= target {
            return;
        }

        let insertion_safepoint =
            self.compensating_insertion_safepoint(halt_check_every, target);

        inserter.insert_compensating_halt_check_after_safepoint(
            insertion_safepoint,
        );

        self.canonicalize_against_resets(
            halt_check_every,
            &inserter.semantic_reset_after,
        );

        debug_assert!(
            self.remaining >= target,
            "compensating insertion failed to normalize state"
        );
    }

    fn merge_halt_states_inner(
        inserter: &mut HaltCheckInserter,
        halt_check_every: NonZero<usize>,
        incoming: &mut [HaltState],
    ) -> HaltState {
        for state in incoming.iter_mut() {
            state.canonicalize_against_resets(
                halt_check_every,
                &inserter.semantic_reset_after
            )
        }

        match *incoming {
            // No incoming edges; start fresh
            [] => HaltState::from_remaining(halt_check_every),

            // Single predecessor: preserve precise path-local suffix history.
            [ref state] => state.clone(),

            _ => {
                let r_lo = incoming
                    .iter()
                    .map(|state| state.remaining)
                    .min()
                    .unwrap();

                let r_hi = incoming
                    .iter()
                    .map(|state| state.remaining)
                    .max()
                    .unwrap();

                let diff = r_hi.get().strict_sub(r_lo.get());
                let threshold = halt_check_every
                    .div_ceil(const { NonZero::new(2).unwrap() });

                if diff > threshold.get() {
                    let target = r_hi;

                    let can_normalize = incoming.iter().all(|state| {
                        state.remaining >= target
                            || state.has_insertion_point(halt_check_every, target)
                    });

                    if can_normalize {
                        for (i, state) in incoming.iter_mut().enumerate() {
                            if i != 0 {
                                state.canonicalize_against_resets(
                                    halt_check_every,
                                    &inserter.semantic_reset_after,
                                );
                            }

                            state.insert_compensating_check(
                                inserter,
                                halt_check_every,
                                target
                            )
                        }

                        debug_assert!(
                            incoming.iter().all(|state| state.remaining >= target),
                            "merge normalization failed to bring every incoming state up to target"
                        );
                        return HaltState::from_remaining(r_hi);
                    }
                }

                HaltState::from_remaining(r_lo)
            }
        }
    }

    fn merge_halt_states(
        inserter: &mut HaltCheckInserter,
        halt_check_every: NonZero<usize>,
        mut incoming: impl DerefMut<Target=[HaltState]>,
    ) -> HaltState {
        Self::merge_halt_states_inner(
            inserter,
            halt_check_every,
            &mut incoming
        )
    }
}


make_handle!(SccComponent);

handle_impl_helper! {
    impl usize like for SccComponent;
}

struct OwnedComponent(Vec<Block>);

impl Storable for OwnedComponent {
    type Handle = SccComponent;
}

struct Tarjan<'a> {
    ir: &'a ExecIrBuilder,
    allowed: Option<&'a ArenaSet<Block>>,

    next_index: usize,
    index: ArenaMap<Block, usize>,
    lowlink: ArenaMap<Block, usize>,

    stack: Vec<Block>,
    on_stack: ArenaSet<Block>,

    components: Arena<OwnedComponent>,
}

impl<'a> Tarjan<'a> {
    pub fn strong_connect(&mut self, block: Block) {
        stacker::maybe_grow(
            4 * 1024,
            2 * 1024 * 1024,
            move || self.strong_connect_inner(block)
        )
    }

    #[inline(always)]
    fn block_is_allowed(&self, block: Block) -> bool {
        self.allowed.is_none_or(|allowed| allowed.contains(block))
    }


    fn strong_connect_inner(&mut self, block: Block) {
        self.index.insert(block, self.next_index);
        self.lowlink.insert(block, self.next_index);
        self.next_index = self.next_index.strict_add(1);

        self.stack.push(block);
        self.on_stack.insert(block);

        for succ in self.ir.successors(block) {
            if !self.block_is_allowed(succ) {
                continue
            }

            if self.index.get(succ).is_none() {
                self.strong_connect(succ);

                let block_lowlink = self.lowlink[block];
                let succ_lowlink = self.lowlink[succ];

                self.lowlink.insert(block, block_lowlink.min(succ_lowlink));
            } else if self.on_stack.contains(succ) {
                let block_lowlink = self.lowlink[block];
                let succ_index = self.index[succ];

                self.lowlink.insert(block, block_lowlink.min(succ_index));
            }
        }

        if self.lowlink[block] == self.index[block] {
            let mut component = Vec::new();

            loop {
                let member = self.stack.pop().unwrap();
                self.on_stack.remove(member);
                component.push(member);

                if member == block {
                    break
                }
            }

            self.components.store(OwnedComponent(component));
        }
    }

    fn run(mut self) -> Arena<OwnedComponent> {
        for block in self.ir.blocks.keys() {
            if !self.block_is_allowed(block) {
                continue
            }

            if self.index.get(block).is_none() {
                self.strong_connect(block);
            }
        }

        self.components
    }
}

fn strongly_connected_components(
    ir: &ExecIrBuilder,
    allowed: Option<ArenaSet<Block>>
) -> Arena<OwnedComponent> {
    let tarjan = Tarjan {
        ir,
        allowed: allowed.as_ref(),

        next_index: 0,
        index: ArenaMap::new(),
        lowlink: ArenaMap::new(),

        stack: Vec::new(),
        on_stack: ArenaSet::new(),

        components: Arena::new(),
    };

    tarjan.run()
}

fn component_is_cyclic(ir: &ExecIrBuilder, component: &[Block]) -> bool {
    match *component {
        // No blocks: never cyclic.
        [] => false,

        // One block: cyclic only if it has a self-loop.
        [one_block] => ir.successors(one_block).any(|succ| succ == one_block),

        // A multi-block SCC is always cyclic.
        //
        // Pick any two distinct blocks A and B. Since this is an SCC,
        // A reaches B and B reaches A; concatenating those paths gives
        // a directed cycle.
        [_, _, ..] => true
    }
}

fn block_has_safepoint(ir: &ExecIrBuilder, block: Block) -> bool {
    ir.blocks[block]
        .stmts
        .iter()
        .any(|&stmt| matches!(&ir.stmts[stmt].rvalue, StmtKind::Safepoint))
}

// prevents the bad case where a cyclic SCC merely contains a safepoint somewhere
// but also contains a separate safepoint-free cycle.
//
// example of why checking simply for if an SCC contains a safepoint is wrong
// block A:
//    _v1 = true;
//    br_nz _v1 A B
// block B:
//    safepoint
//    br A
fn assert_cycles_have_safepoints(ir: &ExecIrBuilder) {
    let allowed = ir
        .blocks
        .keys()
        .filter(|&block| !block_has_safepoint(ir, block))
        .collect::<ArenaSet<_>>();

    let components = strongly_connected_components(ir, Some(allowed));

    for (_, component) in components.iter() {
        if component_is_cyclic(ir, &component.0) {
            panic!(
                "IR invariant violated: found a directed cycle with no safepoint"
            );
        }
    }
}

struct SccGraph {
    components: Arena<OwnedComponent>,
    component_of: ArenaMap<Block, SccComponent>,
    topo_order: Vec<SccComponent>,
    is_cyclic: ArenaSet<SccComponent>,
}

impl SccGraph {
    pub fn new(ir: &ExecIrBuilder) -> Self {
        assert_cycles_have_safepoints(ir);

        let components = strongly_connected_components(ir, None);
        let mut component_of = ArenaMap::<Block, SccComponent>::with_capacity(
            ir.blocks.len()
        );

        for (component_id, component) in components.iter() {
            for &block in &component.0 {
                component_of.insert(block, component_id);
            }
        }

        let component_of = component_of;

        let mut edges = vec![ArenaSet::<SccComponent>::new(); components.len()];
        let mut indegree = vec![0_usize; components.len()];

        for (block, &from_component) in component_of.iter() {
            for succ in ir.successors(block) {
                let to_component = component_of[succ];

                if from_component == to_component {
                    continue;
                }

                if edges[from_component.get()].insert(to_component) {
                    indegree[to_component.get()] = indegree[to_component.get()].strict_add(1);
                }
            }
        }

        let mut ready = indegree
            .iter()
            .enumerate()
            .filter(|&(_, &degree)| degree == 0)
            .map(|(component_id, _)| SccComponent::new(component_id))
            .collect::<VecDeque<SccComponent>>();

        let mut topo_order = Vec::with_capacity(components.len());

        while let Some(component_id) = ready.pop_front() {
            topo_order.push(component_id);

            for succ_component in edges[component_id.get()].iter() {
                indegree[succ_component.get()] = indegree[succ_component.get()].strict_sub(1);

                if indegree[succ_component.get()] == 0 {
                    ready.push_back(succ_component);
                }
            }
        }

        assert_eq!(topo_order.len(), components.len());

        let is_cyclic = components
            .iter()
            .filter(|(_, component)| component_is_cyclic(ir, &component.0))
            .map(|(component, _)| component)
            .collect::<ArenaSet<_>>();

        SccGraph {
            components,
            component_of,
            topo_order,
            is_cyclic,
        }
    }

    pub fn component_is_cyclic(&self, component: SccComponent) -> bool {
        self.is_cyclic.contains(component)
    }
}


#[allow(dead_code)]
pub fn insert_halt_checks(ir: &mut ExecIrBuilder) {
    let halt_check_every: NonZero<u32> = ir.halt_check_every;
    let halt_check_every: NonZero<usize> = halt_check_every
        .try_into()
        .unwrap_or(NonZero::<usize>::MAX);

    let scc_graph = SccGraph::new(ir);

    let mut inserter = HaltCheckInserter::new(halt_check_every, ir);

    const INCOMING_STACK_BUFFER: usize = 64;


    for component_id in scc_graph.topo_order.iter().copied() {
        let component = scc_graph.components[component_id].0.as_slice();


        let mut incoming = SmallVec::<[HaltState; 64]>::new();

        for &block in component {
            let iter = inserter
                .map
                .drain_incoming(block)
                .map(|(_precessor, state)| state);

            incoming.extend(iter);
        }


        let state_in = HaltState::merge_halt_states(
            &mut inserter,
            halt_check_every,
            incoming,
        );

        let is_cyclic = scc_graph.component_is_cyclic(component_id);

        let halter: &mut dyn BlockHalter = match is_cyclic {
            true => &mut CyclicBlockHalter::new(state_in),
            false => &mut ACyclicBlockHalter::new(state_in),
        };

        if cfg!(debug_assertions) && !is_cyclic {
            assert_eq!(component.len(), 1);
        }

        for &block in component {
            let (last_block, state_out) = halter.process_block(
                &mut inserter,
                halt_check_every,
                block,
            );

            for successor in inserter.ir().successors(last_block) {
                if is_cyclic && scc_graph.component_of[successor] == component_id {
                    continue;
                }

                let old = inserter.map.add_edge(
                    last_block,
                    successor,
                    state_out.clone()
                );
                assert!(old.is_none());
            }
        }
    }
}