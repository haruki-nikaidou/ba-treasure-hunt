//! # Treasure Hunt
//!
//! In a map of size `a*b`, there are three types of treasures, with `n_1`, `n_2`, and `n_3` of each type, respectively.
//! The dimensions of each treasure are `(a_1, b_1)`, `(a_2, b_2)`, and `(a_3, b_3)`.
//! Treasures can be placed horizontally or vertically.
//! In a single search, one cell can be revealed;
//! if a cell contains a treasure, regardless of how the treasure is oriented,
//! all cells belonging to that treasure are revealed. Find the decision tree with the minimum height.
//!
//! Reference values: `a=9`, `b=5`. `1<a_i<b_i<4`. `3<n_1+n_2+n_3<15`.

pub mod algorithm;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellMark {
    Unknown,
    Empty
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Treasure {
    pub width: u8,
    pub length: u8,
    pub amount: u8
}

pub struct TreasureHuntProblem<const A: usize, const B: usize> {
    pub cells: [[CellMark; A]; B],
    pub treasures: [Treasure; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlacementInfo {
    pub treasure_type: u8,
    pub row: u8,
    pub col: u8,
    pub placed_w: u8,
    pub placed_h: u8,
}

/// The decision tree.
#[derive(Clone, Debug)]
pub enum DecisionTree {
    Leaf,
    Probe {
        row: u8,
        col: u8,
        on_empty: Box<DecisionTree>,
        on_hit: Vec<(PlacementInfo, Box<DecisionTree>)>,
    },
}
