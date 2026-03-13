use crate::{CellMark, DecisionTree, PlacementInfo, TreasureHuntProblem};
use rustc_hash::FxHashMap;
use smallvec::{SmallVec, smallvec};
use std::time::{Duration, Instant};


/// A single valid placement of a treasure piece on the grid.
#[derive(Clone, Copy, Debug)]
struct Placement {
    /// Metadata describing the treasure type, top-left corner, and orientation.
    info: PlacementInfo,
    /// Bitmask of grid cells occupied by this placement, with bit `r*cols + c` set
    /// for each cell `(r, c)` covered by the piece.
    cells: u64,
}

/// The solver's view of the current search state, used as a memoization key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct State {
    /// Bitmask of cells that are definitively not covered by any remaining
    /// treasure. Bit `r*cols + c` is set if cell `(r, c)` has been ruled out —
    /// either because it was probed and returned "empty", or because a
    /// previously revealed treasure occupies it.
    forbidden: u64,
    /// Number of pieces of each treasure type that still need to be placed in
    /// the configuration space. `remaining[t]` counts how many type-`t`
    /// treasures have not yet been located.
    remaining: [u8; 3],
}

/// Precomputed, problem-specific data used throughout the search.
#[derive(Debug, Clone)]
struct SolverData {
    /// Number of columns in the grid (equals the const generic `A`).
    cols: usize,
    /// All valid placements for each of the three treasure types.
    /// `type_placements[t]` lists every axis-aligned rectangle of the
    /// correct dimensions (both orientations) that fits within the grid,
    /// deduplicated by cell bitmask.
    type_placements: [SmallVec<[Placement; 2]>; 3],
    /// Inverted placement index: for each cell `ci`, the list of placements that
    /// overlap it, as `(treasure_type, placement_idx)` pairs. Used to enumerate
    /// the hit branches when a probe lands on `ci`.
    cell_coverage: Vec<SmallVec<[(usize, usize); 3]>>,
    /// Total number of cells in the grid (`A * B`).
    total_cells: usize,
}

/// Memoization tables shared across the search.
#[derive(Debug, Clone)]
struct Caches {
    /// Maps a `State` to the exact number of valid complete configurations
    /// (ways to place all remaining treasures in non-forbidden cells).
    count: FxHashMap<State, u64>,
    /// Maps a `State` to a proven lower bound on the minimax probe depth
    /// needed to identify the configuration. If `depth_lb[s] = k`, no
    /// strategy can solve state `s` in fewer than `k` probes.
    depth_lb: FxHashMap<State, u8>,
}

// ─── Placement Enumeration ───────────────────────────────────────────────────

/// Returns every distinct placement of a treasure piece with nominal dimensions
/// `w×h` inside a `cols×rows` grid.
///
/// Both axis-aligned orientations (`w×h` and `h×w`) are tried; if the piece is
/// square they collapse to one. The list is sorted and deduplicated by cell
/// bitmask so that rotating a square piece never yields a duplicate entry.
fn enumerate_placements(
    treasure_type: u8,
    w: u8,
    h: u8,
    cols: usize,
    rows: usize,
) -> SmallVec<[Placement; 2]> {
    let orientations: SmallVec<[(u8, u8); 2]> = if w == h {
        smallvec![(w, h)]
    } else {
        smallvec![(w, h), (h, w)]
    };

    let mut placements: SmallVec<[Placement; 2]> = orientations
        .into_iter()
        .flat_map(|(pw, ph)| {
            let max_c = cols as u8;
            let max_r = rows as u8;
            (0..=max_r.saturating_sub(ph)).flat_map(move |r| {
                (0..=max_c.saturating_sub(pw)).map(move |c| {
                    let cells = (0..ph)
                        .flat_map(|dr| {
                            (0..pw).map(move |dc| {
                                1u64 << ((r + dr) as usize * cols + (c + dc) as usize)
                            })
                        })
                        .fold(0u64, |a, b| a | b);
                    Placement {
                        info: PlacementInfo {
                            treasure_type,
                            row: r,
                            col: c,
                            placed_w: pw,
                            placed_h: ph,
                        },
                        cells,
                    }
                })
            })
        })
        .collect();

    placements.sort_by_key(|p| p.cells);
    placements.dedup_by_key(|p| p.cells);
    placements
}

/// Precomputes all placement lists and the cell-coverage index for the given
/// problem, producing the `SolverData` that is shared across the entire search.
///
/// Panics if the grid has more than 64 cells, since the algorithm represents
/// cell sets as `u64` bitmasks.
fn build_solver_data<const A: usize, const B: usize>(
    problem: &TreasureHuntProblem<A, B>,
) -> SolverData {
    assert!(A * B <= 64, "grid too large for u64 bitmask");

    let type_placements: [SmallVec<_>; 3] = std::array::from_fn(|t| {
        enumerate_placements(
            t as u8,
            problem.treasures[t].width,
            problem.treasures[t].length,
            A,
            B,
        )
    });

    let total_cells = A * B;
    let cell_coverage: Vec<SmallVec<[(usize, usize); 3]>> = (0..total_cells)
        .map(|ci| {
            let mask = 1u64 << ci;
            (0..3)
                .flat_map(|t| {
                    type_placements[t]
                        .iter()
                        .enumerate()
                        .filter(move |(_, p)| p.cells & mask != 0)
                        .map(move |(pi, _)| (t, pi))
                })
                .collect()
        })
        .collect();

    SolverData {
        cols: A,
        type_placements,
        cell_coverage,
        total_cells,
    }
}

// ─── Configuration Counting ──────────────────────────────────────────────────

/// Returns the number of valid complete configurations reachable from `state`:
/// the number of ways to place every remaining treasure in non-forbidden cells
/// without overlap. Results are memoized in `caches.count`.
fn count_configs(data: &SolverData, caches: &mut Caches, state: State) -> u64 {
    if let Some(&c) = caches.count.get(&state) {
        return c;
    }
    let result = count_inner(data, state.forbidden, state.remaining, 0, 0);
    caches.count.insert(state, result);
    result
}

/// Recursive inner loop for `count_configs`.
///
/// Places each treasure type in turn, iterating over placements in index order
/// to avoid counting the same multi-piece assignment twice. For a given
/// `type_idx`, only placements at index `>= min_pi` are considered so that
/// pieces of the same type are placed in a canonical (non-decreasing index)
/// order.
fn count_inner(
    data: &SolverData,
    forbidden: u64,
    remaining: [u8; 3],
    type_idx: usize,
    min_pi: usize,
) -> u64 {
    if type_idx >= 3 {
        return 1;
    }
    if remaining[type_idx] == 0 {
        return count_inner(data, forbidden, remaining, type_idx + 1, 0);
    }
    let pls = &data.type_placements[type_idx];
    let mut total = 0u64;
    for pi in min_pi..pls.len() {
        let p = &pls[pi];
        if p.cells & forbidden == 0 {
            let mut rem = remaining;
            rem[type_idx] -= 1;
            total += count_inner(data, forbidden | p.cells, rem, type_idx, pi + 1);
        }
    }
    total
}

/// Returns ⌈log₂(n)⌉, clamped to 0 for `n ≤ 1`.
///
/// Used as the information-theoretic lower bound on the number of probes
/// needed to distinguish among `n` configurations: any binary decision tree
/// with `n` leaves has height at least ⌈log₂(n)⌉.
fn ceil_log2(n: u64) -> u8 {
    if n <= 1 {
        0
    } else {
        (64 - (n - 1).leading_zeros()) as u8
    }
}

// ─── Cell Partition Scoring ──────────────────────────────────────────────────
//
// For a candidate probe, compute how the configuration space splits:
//   miss branch = configs where this cell is empty
//   hit branches = one per placement covering this cell
// Returns (worst_branch_size, branch_details).

/// One possible "hit" outcome when a probe lands on a cell that is covered by
/// a known placement.
#[derive(Debug, Clone, Copy)]
struct HitBranch {
    /// Index into `SolverData::type_placements` identifying which treasure type
    /// was hit.
    treasure_type: usize,
    /// Index of the specific placement within `type_placements[treasure_type]`
    /// that was revealed by this hit.
    placement_idx: usize,
    /// Number of valid configurations consistent with the current state after
    /// this placement is confirmed and removed from the remaining set.
    config_count: u64,
}

/// The result of analysing a single candidate probe cell: how it partitions the
/// current configuration space into branches.
#[derive(Debug, Clone)]
struct CellPartition {
    /// Size of the largest branch that can result from probing this cell.
    /// This is `max(miss_count, hit_0_count, hit_1_count, …)` and is the
    /// quantity minimised by the greedy and exact strategies.
    pub worst_branch_size: u64,
    /// All hit branches — one per valid placement that covers this cell and is
    /// consistent with the current state.
    pub hits: Vec<HitBranch>,
}

/// Partitions the `total` configurations reachable from `state` according to
/// the outcome of probing cell `ci`.
///
/// A probe on `ci` yields one of the following outcomes:
/// - **Miss**: the cell is empty; `total - Σ(hit counts)` configurations remain.
/// - **Hit** (one per valid placement covering `ci`): the placement is fully
///   revealed; the state transitions by forbidding those cells and decrementing
///   the corresponding treasure count.
///
/// Returns the `CellPartition` containing all hit branches and the worst-case
/// (largest) branch size across the miss and all hit branches.
fn cell_partition(
    data: &SolverData,
    caches: &mut Caches,
    state: &State,
    ci: usize,
    total: u64,
) -> CellPartition {
    let mut hits: Vec<HitBranch> = Vec::new();
    let mut hit_sum = 0u64;

    for &(t, pi) in &data.cell_coverage[ci] {
        if state.remaining[t] == 0 {
            continue;
        }
        let p = &data.type_placements[t][pi];
        if p.cells & state.forbidden != 0 {
            continue;
        }
        let mut rem = state.remaining;
        rem[t] -= 1;
        let sub = State {
            forbidden: state.forbidden | p.cells,
            remaining: rem,
        };
        let cnt = count_configs(data, caches, sub);
        if cnt > 0 {
            hits.push(HitBranch {
                treasure_type: t,
                placement_idx: pi,
                config_count: cnt,
            });
            hit_sum += cnt;
        }
    }

    let miss = total.saturating_sub(hit_sum);
    let worst = hits.iter().map(|h| h.config_count).fold(miss, u64::max);
    CellPartition {
        worst_branch_size: worst,
        hits,
    }
}

// ─── Greedy Solver ───────────────────────────────────────────────────────────

/// Computes the worst-case depth of the greedy decision tree rooted at `state`
/// without materialising the tree itself.
///
/// At each step the cell that minimises the largest branch size is chosen
/// (see `pick_greedy_cell`), then the function recurses into every branch
/// (miss and all hits) and returns `1 + max(recursive depths)`.
fn greedy_depth(data: &SolverData, caches: &mut Caches, state: State) -> u8 {
    let total = count_configs(data, caches, state);
    if total <= 1 {
        return 0;
    }
    let (best_ci, _) = pick_greedy_cell(data, caches, &state, total);

    let mut worst: u8 = 0;

    // Empty branch
    let es = State {
        forbidden: state.forbidden | (1u64 << best_ci),
        remaining: state.remaining,
    };
    worst = worst.max(greedy_depth(data, caches, es));

    // Hit branches
    let CellPartition { hits, .. } = cell_partition(data, caches, &state, best_ci, total);
    for &HitBranch {
        treasure_type: t,
        placement_idx: pi,
        ..
    } in &hits
    {
        let p = &data.type_placements[t][pi];
        let mut rem = state.remaining;
        rem[t] -= 1;
        let hs = State {
            forbidden: state.forbidden | p.cells,
            remaining: rem,
        };
        worst = worst.max(greedy_depth(data, caches, hs));
    }
    1 + worst
}

/// Builds and returns the full greedy `DecisionTree` rooted at `state`.
///
/// Uses the same greedy cell-selection strategy as `greedy_depth` but
/// materialises the tree, recursing into every branch to construct the
/// complete subtrees.
fn greedy_tree(data: &SolverData, caches: &mut Caches, state: State) -> DecisionTree {
    let total = count_configs(data, caches, state);
    if total <= 1 {
        return DecisionTree::Leaf;
    }
    let (best_ci, _) = pick_greedy_cell(data, caches, &state, total);
    let row = (best_ci / data.cols) as u8;
    let col = (best_ci % data.cols) as u8;

    let es = State {
        forbidden: state.forbidden | (1u64 << best_ci),
        remaining: state.remaining,
    };
    let on_empty = Box::new(greedy_tree(data, caches, es));

    let CellPartition { hits, .. } = cell_partition(data, caches, &state, best_ci, total);
    let on_hit: Vec<(PlacementInfo, Box<DecisionTree>)> = hits
        .iter()
        .map(
            |&HitBranch {
                 treasure_type: t,
                 placement_idx: pi,
                 ..
             }| {
                let p = &data.type_placements[t][pi];
                let mut rem = state.remaining;
                rem[t] -= 1;
                let hs = State {
                    forbidden: state.forbidden | p.cells,
                    remaining: rem,
                };
                (p.info, Box::new(greedy_tree(data, caches, hs)))
            },
        )
        .collect();

    DecisionTree::Probe {
        row,
        col,
        on_empty,
        on_hit,
    }
}

/// Scans all non-forbidden cells and returns the index of the one that
/// minimises the worst-case branch configuration count (i.e., the greedy
/// minimax cell), along with that worst-case count.
fn pick_greedy_cell(
    data: &SolverData,
    caches: &mut Caches,
    state: &State,
    total: u64,
) -> (usize, u64) {
    let mut worst = u64::MAX;
    let mut best = 0usize;
    for ci in 0..data.total_cells {
        if state.forbidden & (1u64 << ci) != 0 {
            continue;
        }
        let CellPartition {
            worst_branch_size: score,
            ..
        } = cell_partition(data, caches, state, ci, total);
        if score < worst {
            (best, worst) = (ci, score);
        }
    }
    (best, worst)
}

// ─── Exact Minimax Search ────────────────────────────────────────────────────

/// Attempts to find a probing strategy for `state` whose worst-case depth does
/// not exceed `budget`, returning `Some(optimal_depth)` on success or `None`
/// if no such strategy exists within the budget (or if the deadline expires).
///
/// The search is a branch-and-bound minimax:
/// - Candidate probe cells are sorted by their greedy partition score so the
///   most promising cells are tried first.
/// - The information-theoretic lower bound `⌈log₂(configs)⌉` prunes hopeless
///   states immediately.
/// - `caches.depth_lb` records proven lower bounds so that previously-failed
///   budget levels are not retried.
/// - When a `deadline` is provided the search returns `None` as soon as the
///   clock passes it, allowing a time-limited iterative-deepening loop in the
///   caller.
fn search_depth(
    data: &SolverData,
    caches: &mut Caches,
    state: State,
    budget: u8,
    deadline: Option<Instant>,
) -> Option<u8> {
    if let Some(dl) = deadline {
        if Instant::now() >= dl {
            return None;
        }
    }
    let total = count_configs(data, caches, state);
    if total <= 1 {
        return Some(0);
    }
    if budget == 0 {
        return None;
    }
    let lb = ceil_log2(total);
    if lb > budget {
        return None;
    }
    if let Some(&prev) = caches.depth_lb.get(&state) {
        if prev > budget {
            return None;
        }
    }

    // Candidates sorted by partition score (ascending = best first)
    let mut scored: Vec<(usize, u64)> = (0..data.total_cells)
        .filter(|&ci| state.forbidden & (1u64 << ci) == 0)
        .map(|ci| {
            let CellPartition {
                worst_branch_size: s,
                ..
            } = cell_partition(data, caches, &state, ci, total);
            (ci, s)
        })
        .collect();
    scored.sort_unstable_by_key(|x| x.1);

    let mut global_best: Option<u8> = None;

    for &(ci, _) in &scored {
        let cur_budget = match global_best {
            Some(gb) => gb - 1,
            None => budget - 1,
        };

        let es = State {
            forbidden: state.forbidden | (1u64 << ci),
            remaining: state.remaining,
        };
        let ed = match search_depth(data, caches, es, cur_budget, deadline) {
            Some(d) => d,
            None => continue,
        };

        let covering: Vec<(usize, usize)> = data.cell_coverage[ci]
            .iter()
            .filter(|&&(t, pi)| {
                state.remaining[t] > 0 && data.type_placements[t][pi].cells & state.forbidden == 0
            })
            .copied()
            .collect();

        let mut worst = ed;
        let mut ok = true;
        for &(t, pi) in &covering {
            let p = &data.type_placements[t][pi];
            let mut rem = state.remaining;
            rem[t] -= 1;
            let hs = State {
                forbidden: state.forbidden | p.cells,
                remaining: rem,
            };
            let sub_b = match global_best {
                Some(gb) => (gb - 1).min(budget - 1),
                None => budget - 1,
            };
            match search_depth(data, caches, hs, sub_b, deadline) {
                Some(d) => worst = worst.max(d),
                None => {
                    ok = false;
                    break;
                }
            }
            if worst >= budget || global_best.map_or(false, |gb| worst >= gb) {
                ok = false;
                break;
            }
        }

        if ok {
            let depth = 1 + worst;
            match &mut global_best {
                Some(gb) if depth < *gb => *gb = depth,
                None => global_best = Some(depth),
                _ => {}
            }
            if depth <= lb {
                break;
            }
        }
    }

    match global_best {
        Some(d) => {
            caches.depth_lb.insert(state, d);
            Some(d)
        }
        None => {
            caches.depth_lb.insert(state, budget + 1);
            None
        }
    }
}

/// Reconstructs the optimal `DecisionTree` for `state` given that
/// `search_depth` has already confirmed a solution exists within `budget`
/// probes.
///
/// The function re-runs the same candidate enumeration as `search_depth` to
/// find the first probe cell whose subtrees all fit within `budget - 1`, then
/// recursively builds each subtree. Panics if no valid probe is found, which
/// would indicate a logic error (the caller passed an incorrect budget).
fn build_tree(data: &SolverData, caches: &mut Caches, state: State, budget: u8) -> DecisionTree {
    let total = count_configs(data, caches, state);
    if total <= 1 {
        return DecisionTree::Leaf;
    }

    let mut scored: Vec<(usize, u64)> = (0..data.total_cells)
        .filter(|&ci| state.forbidden & (1u64 << ci) == 0)
        .map(|ci| {
            let CellPartition {
                worst_branch_size: s,
                ..
            } = cell_partition(data, caches, &state, ci, total);
            (ci, s)
        })
        .collect();
    scored.sort_unstable_by_key(|x| x.1);

    for &(ci, _) in &scored {
        let row = (ci / data.cols) as u8;
        let col = (ci % data.cols) as u8;

        let es = State {
            forbidden: state.forbidden | (1u64 << ci),
            remaining: state.remaining,
        };
        let Some(_ed) = search_depth(data, caches, es, budget - 1, None) else {
            continue;
        };

        let covering: Vec<(usize, usize)> = data.cell_coverage[ci]
            .iter()
            .filter(|&&(t, pi)| {
                state.remaining[t] > 0 && data.type_placements[t][pi].cells & state.forbidden == 0
            })
            .copied()
            .collect();

        let mut worst = _ed;
        let mut ok = true;
        for &(t, pi) in &covering {
            let p = &data.type_placements[t][pi];
            let mut rem = state.remaining;
            rem[t] -= 1;
            let hs = State {
                forbidden: state.forbidden | p.cells,
                remaining: rem,
            };
            match search_depth(data, caches, hs, budget - 1, None) {
                Some(d) => worst = worst.max(d),
                None => {
                    ok = false;
                    break;
                }
            }
            if 1 + worst > budget {
                ok = false;
                break;
            }
        }

        if ok && 1 + worst <= budget {
            let on_empty = Box::new(build_tree(data, caches, es, budget - 1));
            let on_hit = covering
                .iter()
                .map(|&(t, pi)| {
                    let p = &data.type_placements[t][pi];
                    let mut rem = state.remaining;
                    rem[t] -= 1;
                    let hs = State {
                        forbidden: state.forbidden | p.cells,
                        remaining: rem,
                    };
                    (p.info, Box::new(build_tree(data, caches, hs, budget - 1)))
                })
                .collect();
            return DecisionTree::Probe {
                row,
                col,
                on_empty,
                on_hit,
            };
        }
    }
    unreachable!("build_tree: no probe at budget {budget}");
}

// ─── Tree Metrics ────────────────────────────────────────────────────────────

/// Returns the height of the decision tree: the length of the longest path from
/// the root to a `Leaf`, which equals the worst-case number of probes.
pub fn tree_height(t: &DecisionTree) -> u8 {
    match t {
        DecisionTree::Leaf => 0,
        DecisionTree::Probe {
            on_empty, on_hit, ..
        } => {
            let h = on_hit
                .iter()
                .map(|(_, s)| tree_height(s))
                .fold(tree_height(on_empty), u8::max);
            1 + h
        }
    }
}

/// Returns the total number of nodes in the decision tree (both `Probe` and
/// `Leaf` nodes), which gives a rough measure of tree complexity.
pub fn tree_node_count(t: &DecisionTree) -> usize {
    match t {
        DecisionTree::Leaf => 1,
        DecisionTree::Probe {
            on_empty, on_hit, ..
        } => {
            1 + tree_node_count(on_empty)
                + on_hit
                    .iter()
                    .map(|(_, s)| tree_node_count(s))
                    .sum::<usize>()
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// The public solver, holding precomputed problem data and memoization caches
/// that persist across multiple `solve` calls for the same grid geometry.
pub struct Solver {
    /// Precomputed placements and cell-coverage index for the problem.
    data: SolverData,
    /// Memoization tables for configuration counts and depth lower bounds.
    caches: Caches,
}

/// The result of a solve: a complete decision tree together with summary
/// statistics.
pub struct Solution {
    /// Worst-case number of probes required by the decision tree.
    pub depth: u8,
    /// The decision tree itself.
    pub tree: DecisionTree,
    /// Total number of valid starting configurations (ways to place all
    /// treasures in the initial state).
    pub total_configs: u64,
    /// `true` if `depth` is provably optimal (equals the minimax optimum);
    /// `false` if the tree was produced by the greedy heuristic and may be
    /// sub-optimal.
    pub is_optimal: bool,
}

impl Solver {
    /// Creates a new `Solver` for the given problem, precomputing all placements
    /// and the cell-coverage index. The solver can be reused across multiple
    /// calls to `solve` or `solve_greedy` for the same grid dimensions.
    pub fn new<const A: usize, const B: usize>(problem: &TreasureHuntProblem<A, B>) -> Self {
        let data = build_solver_data(problem);
        let caches = Caches {
            count: FxHashMap::default(),
            depth_lb: FxHashMap::default(),
        };
        Solver { data, caches }
    }

    /// Constructs the initial `State` for the given problem by marking every
    /// cell that is pre-labeled `CellMark::Empty` as forbidden, and setting
    /// `remaining[t]` to the count of each treasure type.
    fn initial_state<const A: usize, const B: usize>(
        &self,
        problem: &TreasureHuntProblem<A, B>,
    ) -> State {
        let mut forbidden = 0u64;
        for r in 0..B {
            for c in 0..A {
                if problem.cells[r][c] == CellMark::Empty {
                    forbidden |= 1u64 << (r * A + c);
                }
            }
        }
        State {
            forbidden,
            remaining: std::array::from_fn(|i| problem.treasures[i].amount),
        }
    }

    /// Solves the problem using the greedy heuristic only.
    ///
    /// At each decision node the cell that minimises the worst-case remaining
    /// configuration count is chosen. This runs quickly but the resulting tree
    /// depth may be larger than the true minimax optimum. The returned
    /// `Solution` always has `is_optimal = false`.
    pub fn solve_greedy<const A: usize, const B: usize>(
        &mut self,
        problem: &TreasureHuntProblem<A, B>,
    ) -> Solution {
        let init = self.initial_state(problem);
        let total = count_configs(&self.data, &mut self.caches, init);
        let tree = greedy_tree(&self.data, &mut self.caches, init);
        let depth = tree_height(&tree);
        Solution {
            depth,
            tree,
            total_configs: total,
            is_optimal: false,
        }
    }

    /// Attempts an exact minimax solve within `time_limit`, falling back to the
    /// greedy solution if time runs out.
    ///
    /// Strategy:
    /// 1. Compute the total number of configurations and the greedy upper bound.
    /// 2. If the greedy depth already equals the information-theoretic lower
    ///    bound `⌈log₂(configs)⌉`, it is provably optimal and is returned
    ///    immediately.
    /// 3. Otherwise, run iterative-deepening exact search from the lower bound
    ///    up to (but not including) the greedy depth, stopping as soon as a
    ///    feasible depth is found or the deadline is reached.
    /// 4. If the deadline expires before finding a better solution, the greedy
    ///    tree is returned with `is_optimal = false`.
    pub fn solve<const A: usize, const B: usize>(
        &mut self,
        problem: &TreasureHuntProblem<A, B>,
        time_limit: Duration,
    ) -> Solution {
        let t0 = Instant::now();
        let init = self.initial_state(problem);
        let total = count_configs(&self.data, &mut self.caches, init);

        if total <= 1 {
            return Solution {
                depth: 0,
                tree: DecisionTree::Leaf,
                total_configs: total,
                is_optimal: true,
            };
        }

        let lb = ceil_log2(total);
        let greedy_d = greedy_depth(&self.data, &mut self.caches, init);
        eprintln!("  configs={total}, info-LB={lb}, greedy-UB={greedy_d}");

        if greedy_d <= lb {
            eprintln!("  greedy = LB → optimal");
            let tree = greedy_tree(&self.data, &mut self.caches, init);
            return Solution {
                depth: greedy_d,
                tree,
                total_configs: total,
                is_optimal: true,
            };
        }

        let deadline = t0 + time_limit;
        for d in lb..greedy_d {
            if Instant::now() >= deadline {
                eprintln!("  timeout → greedy (depth {greedy_d})");
                break;
            }
            eprintln!("  exact: trying depth {d}…");
            self.caches.depth_lb.clear();
            if search_depth(&self.data, &mut self.caches, init, d, Some(deadline)).is_some() {
                eprintln!("  ✓ optimal depth = {d}");
                let tree = build_tree(&self.data, &mut self.caches, init, d);
                return Solution {
                    depth: d,
                    tree,
                    total_configs: total,
                    is_optimal: true,
                };
            }
        }

        let tree = greedy_tree(&self.data, &mut self.caches, init);
        Solution {
            depth: greedy_d,
            tree,
            total_configs: total,
            is_optimal: false,
        }
    }

    /// Returns the number of distinct valid placements for each of the three
    /// treasure types, in type order. Useful for diagnostics and logging.
    pub fn placement_counts(&self) -> [usize; 3] {
        std::array::from_fn(|i| self.data.type_placements[i].len())
    }
}

// ─── Display ─────────────────────────────────────────────────────────────────

impl core::fmt::Display for DecisionTree {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fn show(
            f: &mut core::fmt::Formatter<'_>,
            t: &DecisionTree,
            ind: usize,
        ) -> core::fmt::Result {
            let pad = " ".repeat(ind);
            match t {
                DecisionTree::Leaf => writeln!(f, "{pad}→ determined"),
                DecisionTree::Probe {
                    row,
                    col,
                    on_empty,
                    on_hit,
                } => {
                    writeln!(f, "{pad}probe ({row}, {col})")?;
                    writeln!(f, "{pad}  if empty:")?;
                    show(f, on_empty, ind + 4)?;
                    for (info, sub) in on_hit {
                        writeln!(
                            f,
                            "{pad}  if hit T{} @({},{}) {}×{}:",
                            info.treasure_type, info.row, info.col, info.placed_w, info.placed_h
                        )?;
                        show(f, sub, ind + 4)?;
                    }
                    Ok(())
                }
            }
        }
        show(f, self, 0)
    }
}

impl core::fmt::Display for Solution {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let tag = if self.is_optimal {
            "optimal"
        } else {
            "greedy upper bound"
        };
        writeln!(f, "Configurations : {}", self.total_configs)?;
        writeln!(f, "Worst-case depth: {} ({tag})", self.depth)?;
        writeln!(f, "Tree nodes      : {}", tree_node_count(&self.tree))?;
        writeln!(f, "\nDecision tree:")?;
        write!(f, "{}", self.tree)
    }
}
