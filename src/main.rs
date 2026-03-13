use ba_treasure_hunt::*;

fn main() {
    let problem = TreasureHuntProblem {
        cells: [[CellMark::Unknown; 5]; 9],
        treasures: [
            Treasure {
                width: 1,
                length: 1,
                amount: 3,
            },
            Treasure {
                width: 1,
                length: 3,
                amount: 2,
            },
            Treasure {
                width: 2,
                length: 2,
                amount: 1,
            },
        ],
    };
    let mut solver = algorithm::Solver::new(&problem);
    let result = solver.solve(&problem, std::time::Duration::from_mins(2));

    println!(
        "depth: {}, total_configs: {}",
        result.depth, result.total_configs
    );
}
