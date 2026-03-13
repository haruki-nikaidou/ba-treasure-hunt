use crate::{CellMark, DecisionTree, PlacementInfo, TreasureHuntProblem};
use rustc_hash::FxHashMap;
use smallvec::{SmallVec, smallvec};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
struct Placement {
    info: PlacementInfo,
    cells: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct State {
    forbidden: u64,
    remaining: [u8; 3],
}

#[derive(Debug, Clone)]
struct SolverData {
    cols: usize,
    type_placements: [SmallVec<[Placement; 2]>; 3],
    cell_coverage: Vec<SmallVec<[(usize, usize); 3]>>,
    total_cells: usize,
}

#[derive(Debug, Clone)]
struct Caches {
    count: FxHashMap<State, u64>,
    depth_lb: FxHashMap<State, u8>,
}

// ─── Placement Enumeration ───────────────────────────────────────────────────

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

fn count_configs(data: &SolverData, caches: &mut Caches, state: State) -> u64 {
    if let Some(&c) = caches.count.get(&state) {
        return c;
    }
    let result = count_inner(data, state.forbidden, state.remaining, 0, 0);
    caches.count.insert(state, result);
    result
}

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

#[derive(Debug, Clone)]
struct CellPartition {
    pub worst_branch_size: u64,
    pub hits: Vec<(usize, usize, u64)>,
}

fn cell_partition(
    data: &SolverData,
    caches: &mut Caches,
    state: &State,
    ci: usize,
    total: u64,
) -> CellPartition {
    let mut hits: Vec<(usize, usize, u64)> = Vec::new();
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
            hits.push((t, pi, cnt));
            hit_sum += cnt;
        }
    }

    let miss = total.saturating_sub(hit_sum);
    let worst = hits.iter().map(|h| h.2).fold(miss, u64::max);
    CellPartition {
        worst_branch_size: worst,
        hits,
    }
}

// ─── Greedy Solver ───────────────────────────────────────────────────────────

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
    for (t, pi, _) in &hits {
        let p = &data.type_placements[*t][*pi];
        let mut rem = state.remaining;
        rem[*t] -= 1;
        let hs = State {
            forbidden: state.forbidden | p.cells,
            remaining: rem,
        };
        worst = worst.max(greedy_depth(data, caches, hs));
    }
    1 + worst
}

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
        .map(|&(t, pi, _)| {
            let p = &data.type_placements[t][pi];
            let mut rem = state.remaining;
            rem[t] -= 1;
            let hs = State {
                forbidden: state.forbidden | p.cells,
                remaining: rem,
            };
            (p.info, Box::new(greedy_tree(data, caches, hs)))
        })
        .collect();

    DecisionTree::Probe {
        row,
        col,
        on_empty,
        on_hit,
    }
}

/// Pick the cell that minimises the worst-case branch config count.
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

pub struct Solver {
    data: SolverData,
    caches: Caches,
}

pub struct Solution {
    pub depth: u8,
    pub tree: DecisionTree,
    pub total_configs: u64,
    pub is_optimal: bool,
}

impl Solver {
    pub fn new<const A: usize, const B: usize>(problem: &TreasureHuntProblem<A, B>) -> Self {
        let data = build_solver_data(problem);
        let caches = Caches {
            count: FxHashMap::default(),
            depth_lb: FxHashMap::default(),
        };
        Solver { data, caches }
    }

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

    /// Fast greedy solve.
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

    /// Exact solve with time limit; falls back to greedy.
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
