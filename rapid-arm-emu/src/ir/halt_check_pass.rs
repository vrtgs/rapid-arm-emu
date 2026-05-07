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

use std::collections::{HashMap, VecDeque};
use std::num::NonZero;
use crate::ir::{Block, ExecIrBuilder, StmtKind};
use crate::ir::arena::{handle_impl_helper, make_handle, Arena, ArenaMap, ArenaSet, Storable};

#[derive(Copy, Clone)]
struct SafepointLoc {
    stmt_index: usize,
    block: Block,
}


#[derive(Copy, Clone)]
struct ShouldHalt(bool);

#[derive(Clone)]
struct HaltState {
    remaining: NonZero<usize>,
    suffix_safepoints: rpds::Queue<SafepointLoc>
}

impl HaltState {
    fn from_remaining(remaining: NonZero<usize>) -> Self {
        Self {
            remaining,
            suffix_safepoints: rpds::Queue::new(),
        }
    }

    /// This is important:
    ///
    /// `suffix_safepoints` is only valid for compensating insertion if it
    /// represents the whole unchecked suffix since the previous halt check.
    ///
    /// If we passed through a merge where histories disagreed, we keep the
    /// countdown but intentionally discard suffix precision.
    fn has_complete_suffix(&self, halt_check_every: NonZero<usize>) -> bool {
        self.suffix_safepoints.len()
            == halt_check_every.get().strict_sub(self.remaining.get())
    }

    fn can_normalize_to(
        &self,
        halt_check_every: NonZero<usize>,
        target: NonZero<usize>,
    ) -> bool {
        if self.remaining >= target {
            return true;
        }

        if !self.has_complete_suffix(halt_check_every) {
            return false;
        }

        let safepoints_after_insert = halt_check_every.get().strict_sub(target.get());

        // We need the safepoint after which exactly
        // `safepoints_after_insert` safepoints remain before the merge.
        self.suffix_safepoints.len() > safepoints_after_insert
    }

    pub fn normalize_to(
        &mut self,
        ir: &mut ExecIrBuilder,
        halt_check_every: NonZero<usize>,
        target: NonZero<usize>,
    ) {
        assert!(self.can_normalize_to(halt_check_every, target));

        if self.remaining >= target {
            return;
        }

        let safepoints_after_insert = halt_check_every.get().strict_sub(target.get());

        // Deliberately discard suffix precision after compensating insertion.
        //
        // Keeping it would require remapping SafepointLocs if the insertion
        // split a block. The countdown remains correct, and losing suffix
        // precision only disables later normalization opportunities.
        let mut suffixes = std::mem::take(&mut self.suffix_safepoints);

        let skip_from_front = suffixes
            .len()
            .strict_sub(safepoints_after_insert)
            .strict_sub(1);

        for _ in 0..skip_from_front {
            assert!(suffixes.dequeue_mut());
        }

        // `can_normalize_to` ensured this safepoint exists.
        let insertion_safepoint = suffixes.peek().unwrap();

        ir.insert_halt_check_at(
            insertion_safepoint.block,
            insertion_safepoint.stmt_index.strict_add(1),
        );

        self.remaining = target;
        // `self.suffix_safepoints` was taken above and intentionally remains empty.
        // This discards suffix precision after the compensating insertion.
    }

    pub fn push_safepoint(
        &mut self,
        halt_check_every: NonZero<usize>,
        safepoint: SafepointLoc,
    ) -> ShouldHalt {
        let new_remaining = NonZero::new(self.remaining.get().strict_sub(1));
        self.remaining = new_remaining.unwrap_or(halt_check_every);

        let halt = ShouldHalt(new_remaining.is_none());

        match halt {
            // There is never a need to keep suffix history before this point,
            // because after inserting a halt check the countdown resets to N.
            ShouldHalt(true) => self.suffix_safepoints = rpds::Queue::new(),
            ShouldHalt(false) => {
                self.suffix_safepoints.enqueue_mut(safepoint);
                if self.suffix_safepoints.len() > halt_check_every.get() {
                    self.suffix_safepoints.dequeue_mut();
                }
            }
        }

        halt
    }

    #[inline]
    fn break_down_block_inner(
        mut this: Option<&mut Self>,
        ir: &mut ExecIrBuilder,
        halt_check_every: NonZero<usize>,
        mut block: Block
    ) -> Block {
        'break_down_loop: loop {
            let mut split = None;
            for (i, stmt) in ir.blocks[block].stmts.iter().enumerate() {
                if let StmtKind::Safepoint = stmt.rvalue {
                    let should_halt = this.as_mut().map_or(
                        ShouldHalt(true),
                        |this| this.push_safepoint(
                            halt_check_every,
                            SafepointLoc {
                                stmt_index: i,
                                block
                            }
                        )
                    );

                    if should_halt.0 {
                        // insert directly after safepoint
                        split = Some(i.strict_add(1));
                        break
                    }
                }
            }

            let Some(pos) = split else {
                break 'break_down_loop
            };

            block = ir.insert_halt_check_at(block, pos);
        }

        block
    }

    pub fn break_down_block(
        mut self,
        ir: &mut ExecIrBuilder,
        halt_check_every: NonZero<usize>,
        block: Block
    ) -> (Block, Self) {
        let last_block = Self::break_down_block_inner(
            Some(&mut self),
            ir,
            halt_check_every,
            block
        );

        (last_block, self)
    }

    /// For cyclic SCCs, countdown analysis inside the SCC is not necessary for
    /// termination safety. Once we have proven every cycle contains at least one
    /// safepoint, inserting a halt check after every safepoint in the cyclic SCC
    /// guarantees every cycle contains at least one halt check.
    pub fn force_halt_checks_after_safepoints(
        ir: &mut ExecIrBuilder,
        block: Block,
    ) -> Block {
        Self::break_down_block_inner(
            None,
            ir,
            const { NonZero::new(1).unwrap() },
            block
        )
    }

    fn merge_halt_states(
        ir: &mut ExecIrBuilder,
        halt_check_every: NonZero<usize>,
        incoming: &mut [(Block, HaltState)],
    ) -> HaltState {
        match incoming {
            [] => HaltState::from_remaining(halt_check_every),

            // Single predecessor: preserve precise path-local suffix history.
            [(_, state)] => state.clone(),

            _ => {
                let r_lo = incoming
                    .iter()
                    .map(|(_, state)| state.remaining)
                    .min()
                    .unwrap();

                let r_hi = incoming
                    .iter()
                    .map(|(_, state)| state.remaining)
                    .max()
                    .unwrap();

                let diff = r_hi.get().strict_sub(r_lo.get());
                let threshold = halt_check_every
                    .div_ceil(const { NonZero::new(2).unwrap() });

                if diff > threshold.get() {
                    let can_normalize = incoming
                        .iter()
                        .all(|(_, s)| s.can_normalize_to(halt_check_every, r_hi));

                    if can_normalize {
                        for (_, state) in incoming.iter_mut() {
                            state.normalize_to(ir, halt_check_every, r_hi);
                        }

                        return HaltState::from_remaining(r_hi);
                    }
                }

                HaltState::from_remaining(r_lo)
            }
        }
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


fn block_has_safepoint(ir: &ExecIrBuilder, block: Block) -> bool {
    ir.blocks[block]
        .stmts
        .iter()
        .any(|stmt| matches!(&stmt.rvalue, StmtKind::Safepoint))
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


struct SccGraph {
    components: Arena<OwnedComponent>,
    component_of: ArenaMap<Block, SccComponent>,
    topo_order: Vec<SccComponent>,
    is_cyclic: ArenaSet<SccComponent>,
}

fn assert_cyclce_have_safepoint(ir: &ExecIrBuilder) {
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

impl SccGraph {
    pub fn new(ir: &ExecIrBuilder) -> Self {
        assert_cyclce_have_safepoint(ir);

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

        debug_assert_eq!(topo_order.len(), components.len());

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

pub fn insert_halt_checks(ir: &mut ExecIrBuilder) {
    let halt_check_every = NonZero::<usize>::try_from(ir.halt_check_every)
        .unwrap_or(NonZero::<usize>::MAX);

    let scc_graph = SccGraph::new(ir);

    let mut edge_state = HashMap::<(Block, Block), HaltState>::new();

    for component_id in scc_graph.topo_order.iter().copied() {
        let component = &scc_graph.components[component_id];

        let mut incoming = Vec::<(Block, HaltState)>::new();

        for &block in &component.0 {
            if block == Block::ENTRYPOINT {
                incoming.push((
                    Block::ENTRYPOINT,
                    HaltState::from_remaining(halt_check_every)
                ));
                debug_assert!(ir.predecessors(block).is_empty());
                continue
            }

            for &pred in ir.predecessors(block).iter() {
                if scc_graph.component_of.get(pred).copied() == Some(component_id) {
                    continue;
                }

                if let Some(state) = edge_state.remove(&(pred, block)) {
                    incoming.push((pred, state));
                }
            }
        }

        let halt_state_in =
            HaltState::merge_halt_states(ir, halt_check_every, &mut incoming);

        if scc_graph.component_is_cyclic(component_id) {
            // TODO more complex analysis to be able to have fewer halt checks
            //      this currently works though, and this is low priority for
            //      now; note if this changes, please update the docs for this
            //      pass

            // Cyclic SCC rule:
            //
            // We have already proven every cycle in this SCC contains at least
            // one safepoint. By forcing a halt check after every safepoint in
            // this SCC, every cycle now contains at least one halt check.
            //
            // For outgoing countdown state, use conservative identity:
            //
            // - paths that hit a safepoint reset to N because of the forced check
            // - paths that do not hit a safepoint preserve the incoming countdown
            //
            // Since incoming.remaining <= N, using incoming.remaining for all
            // exits is safe.
            let halt_state_out = HaltState::from_remaining(halt_state_in.remaining);

            for &block in &component.0 {
                let tail_block =
                    HaltState::force_halt_checks_after_safepoints(ir, block);

                for succ in ir.successors(tail_block) {
                    if scc_graph.component_of.get(succ).copied() == Some(component_id) {
                        continue;
                    }

                    edge_state.insert((tail_block, succ), halt_state_out.clone());
                }
            }
        } else {
            let &[block] = component.0.as_slice() else {
                panic!("empty SCC component")
            };

            let (tail_block, halt_state_out) =
                halt_state_in.break_down_block(ir, halt_check_every, block);

            for succ in ir.successors(tail_block) {
                edge_state.insert((tail_block, succ), halt_state_out.clone());
            }
        }
    }
}