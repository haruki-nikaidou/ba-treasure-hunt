//! # Treasure Hunt
//!
//! In a map of size `a*b`, there are three types of treasures, with `n_1`, `n_2`, and `n_3` of each type, respectively.
//! The dimensions of each treasure are `(a_1, b_1)`, `(a_2, b_2)`, and `(a_3, b_3)`.
//! Treasures can be placed horizontally or vertically.
//! In a single search, one cell can be revealed;
//! if a cell contains a treasure, regardless of how the treasure is oriented,
//! all cells belonging to that treasure are revealed. Find the decision tree with the minimum height.
//!
//! Reference values: `a=9`, `b=5`. `1<=a_i<b_i<=4`. `3<=n_1+n_2+n_3<=15`.

pub mod algorithm;

/// Bitmask of grid cells occupied by this placement, with bit `y * A + x` set
/// for each cell `(x, y)` covered by the piece.
///
/// Use `0` to represent an initial cell, `1` to represent a revealed cell
#[derive(Debug, Clone, Copy)]
pub struct Board(pub u64);

impl Board {
    const A: u8 = 9;
    const B: u8 = 5;
    const MAX_RANGE: Coordinate = Coordinate {
        x: Self::A,
        y: Self::B,
    };
    pub fn coord_to_index(c: Coordinate) -> u8 {
        if !c.is_in(Self::MAX_RANGE) {
            panic!("Invalid Coordinate");
        }
        c.y * Self::A + c.x
    }
    pub fn index_to_coord(index: u8) -> Coordinate {
        let y = index / Self::A;
        let x = index % Self::A;
        Coordinate { x, y }
    }
    pub fn area_to_bits(a: Area) -> u64 {
        let Area { start, size } = a;
        if !start.is_in(Self::MAX_RANGE) {
            panic!("Invalid Coordinate");
        }
        let end = start + size;
        if !end.is_in(Self::MAX_RANGE) {
            panic!("Invalid Coordinate");
        }
        let row_musk = u64::MAX >> (64 - size.x);
        let col_musk = (0..size.y).fold(row_musk, |acc, index| acc | (row_musk << index * Self::A));
        col_musk
    }
    pub fn access(self, coordinate: Coordinate) -> bool {
        let musk = 1u64 << (Self::coord_to_index(coordinate));
        self.0 & musk != 0
    }
    pub fn set(&mut self, coordinate: Coordinate, value: bool) {
        if value {
            let musk = 1u64 << (Self::coord_to_index(coordinate));
            self.0 |= musk;
        } else {
            let musk = !(1u64 << (Self::coord_to_index(coordinate)));
            self.0 &= musk;
        }
    }
    pub fn set_area(&mut self, area: Area, value: bool) {
        let musk = Self::area_to_bits(area);
        if value {
            self.0 |= musk;
        } else {
            let col_musk = !musk;
            self.0 &= col_musk;
        }
    }
    pub fn check_area_empty(&self, area: Area) -> bool {
        let musk = Self::area_to_bits(area);
        musk & self.0 == 0
    }
    pub fn search_empty_area(&self, size: Coordinate) -> Option<Coordinate> {
        let bits = Self::area_to_bits(Area {
            start: Coordinate { x: 0, y: 0 },
            size,
        });
        let area_end = bits.trailing_zeros() as u8;
        let board_end = Self::A * Self::B - 1;
        let match_ranges = board_end - area_end;
        (0..match_ranges)
            .find(|&shift| self.0 & (bits << shift) != 0)
            .map(Self::index_to_coord)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coordinate {
    pub x: u8,
    pub y: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Area {
    pub start: Coordinate,
    pub size: Coordinate,
}

impl Coordinate {
    pub fn is_in(self, other: Coordinate) -> bool {
        other.x >= self.x && other.y >= self.y
    }
    pub fn get_orientation(self) -> Orientation {
        match self.x.cmp(&self.y) {
            std::cmp::Ordering::Equal => Orientation::Equal,
            std::cmp::Ordering::Less => Orientation::Vertical,
            std::cmp::Ordering::Greater => Orientation::Horizontal,
        }
    }
    pub fn transpose(self) -> Coordinate {
        Self {
            x: self.y,
            y: self.x,
        }
    }
}

impl std::ops::Add for Coordinate {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Treasure {
    pub size: Coordinate,
    pub amount: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct TreasureHuntProblem {
    pub treasures: [Treasure; 3],
}

impl TreasureHuntProblem {
    /// Let all treasure be horizontal orientation or equal orientation
    pub fn normalization(self) -> Self {
        let mapped = self
            .treasures
            .map(|treasures| match treasures.size.get_orientation() {
                Orientation::Equal => treasures,
                Orientation::Vertical => Treasure {
                    amount: treasures.amount,
                    size: treasures.size.transpose(),
                },
                Orientation::Horizontal => treasures,
            });
        Self { treasures: mapped }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Orientation {
    /// ```text
    /// o x x
    /// x x x
    /// ```
    Horizontal,

    /// ```text
    /// o x
    /// x x
    /// x x
    /// ```
    Vertical,

    /// ```text
    /// o x x
    /// x x x
    /// x x x
    /// ```
    Equal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlacementInfo {
    pub treasure_type: u8,
    pub position: Coordinate,
    pub orientation: Orientation,
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
