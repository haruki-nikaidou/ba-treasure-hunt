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
