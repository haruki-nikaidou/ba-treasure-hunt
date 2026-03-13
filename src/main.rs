use ba_treasure_hunt::algorithm::{Solution, Solver};
use ba_treasure_hunt::{DecisionTree, Treasure, TreasureHuntProblem};
use crossterm::cursor;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{self, ClearType};
use crossterm::{execute, queue};
use std::io::{self, Write};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

const WIDTH: usize = 9;
const HEIGHT: usize = 5;

#[derive(Clone, Copy, Debug)]
enum Orientation {
    Horizontal,
    Vertical,
}

impl Orientation {
    fn toggle(self) -> Self {
        match self {
            Orientation::Horizontal => Orientation::Vertical,
            Orientation::Vertical => Orientation::Horizontal,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Orientation::Horizontal => "H",
            Orientation::Vertical => "V",
        }
    }
}

fn main() -> io::Result<()> {
    let treasures = configure_treasures()?;
    run_tui(treasures)
}

fn configure_treasures() -> io::Result<[Treasure; 3]> {
    println!("Treasure Hunt setup (grid fixed at {}x{}).", WIDTH, HEIGHT);
    println!("For each treasure type enter: width length amount");
    println!("Example: 1 3 4");

    let mut treasures = [Treasure {
        width: 1,
        length: 2,
        amount: 1,
    }; 3];

    for (i, t) in treasures.iter_mut().enumerate() {
        loop {
            print!("T{} > ", i + 1);
            io::stdout().flush()?;
            let mut line = String::new();
            io::stdin().read_line(&mut line)?;
            let parts: Vec<_> = line.split_whitespace().collect();
            if parts.len() != 3 {
                println!("Need exactly 3 values.");
                continue;
            }
            let Ok(width) = parts[0].parse::<u8>() else {
                println!("Invalid width.");
                continue;
            };
            let Ok(length) = parts[1].parse::<u8>() else {
                println!("Invalid length.");
                continue;
            };
            let Ok(amount) = parts[2].parse::<u8>() else {
                println!("Invalid amount.");
                continue;
            };
            *t = Treasure {
                width,
                length,
                amount,
            };
            break;
        }
    }

    Ok(treasures)
}

fn run_tui(treasures: [Treasure; 3]) -> io::Result<()> {
    let mut stdout = io::stdout();
    terminal::enable_raw_mode()?;
    execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide)?;

    let result = run_loop(treasures, &mut stdout);

    execute!(stdout, cursor::Show, terminal::LeaveAlternateScreen)?;
    terminal::disable_raw_mode()?;
    result
}

fn run_loop(treasures: [Treasure; 3], stdout: &mut io::Stdout) -> io::Result<()> {
    let mut current_tree: Option<DecisionTree> = None;
    let mut compute_rx: Option<Receiver<Solution>> = None;
    let mut started_at: Option<Instant> = None;
    let mut spinner_idx: usize = 0;

    let mut selected_row = 0usize;
    let mut selected_col = 0usize;
    let mut selected_result: u8 = 0;
    let mut selected_orientation = Orientation::Horizontal;
    let mut message = String::from("Press 'o' to compute.");

    loop {
        if let Some(rx) = &compute_rx {
            if let Ok(solution) = rx.try_recv() {
                current_tree = Some(solution.tree);
                compute_rx = None;
                started_at = None;
                message = "Computed. Pick a cell with arrows and Enter.".to_string();
            }
        }

        draw(
            stdout,
            current_tree.as_ref(),
            compute_rx.is_some(),
            started_at,
            &mut spinner_idx,
            selected_row,
            selected_col,
            selected_result,
            selected_orientation,
            &message,
        )?;

        if !event::poll(Duration::from_millis(80))? {
            continue;
        }

        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        if matches!(key.code, KeyCode::Char('q')) {
            break;
        }

        if compute_rx.is_some() {
            continue;
        }

        match key.code {
            KeyCode::Char('o') => {
                let (tx, rx) = mpsc::channel();
                compute_rx = Some(rx);
                started_at = Some(Instant::now());
                message = "Computing...".to_string();
                thread::spawn(move || {
                    let mut solver = Solver::new(&TreasureHuntProblem::<WIDTH, HEIGHT> {
                        cells: [[ba_treasure_hunt::CellMark::Unknown; WIDTH]; HEIGHT],
                        treasures,
                    });
                    let solution = solver.solve(
                        &TreasureHuntProblem::<WIDTH, HEIGHT> {
                            cells: [[ba_treasure_hunt::CellMark::Unknown; WIDTH]; HEIGHT],
                            treasures,
                        },
                        Duration::from_secs(60),
                    );
                    let _ = tx.send(solution);
                });
            }
            KeyCode::Left => selected_col = selected_col.saturating_sub(1),
            KeyCode::Right => {
                if selected_col + 1 < WIDTH {
                    selected_col += 1;
                }
            }
            KeyCode::Up => selected_row = selected_row.saturating_sub(1),
            KeyCode::Down => {
                if selected_row + 1 < HEIGHT {
                    selected_row += 1;
                }
            }
            KeyCode::Char('0') => selected_result = 0,
            KeyCode::Char('1') => selected_result = 1,
            KeyCode::Char('2') => selected_result = 2,
            KeyCode::Char('3') => selected_result = 3,
            KeyCode::Char('j') | KeyCode::Char('k') => {
                selected_orientation = selected_orientation.toggle()
            }
            KeyCode::Enter => {
                message = format!(
                    "Selected cell ({}, {}). Set result with 0-3, orientation with j/k, confirm with Y.",
                    selected_row, selected_col
                );
            }
            KeyCode::Char('n') => {
                message = "Exit requested with 'n'. Press q to quit program.".to_string();
            }
            KeyCode::Char('Y') => {
                let Some(tree) = current_tree.as_ref() else {
                    message = "Compute first with 'o'.".to_string();
                    continue;
                };
                match apply_result(
                    tree,
                    selected_row,
                    selected_col,
                    selected_result,
                    selected_orientation,
                ) {
                    Ok(next) => {
                        current_tree = Some(next);
                        message = "Result applied. Continue from updated state.".to_string();
                    }
                    Err(()) => {
                        message = "Invalid result for current node; state unchanged.".to_string();
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn apply_result(
    tree: &DecisionTree,
    row: usize,
    col: usize,
    result: u8,
    orientation: Orientation,
) -> Result<DecisionTree, ()> {
    match tree {
        DecisionTree::Leaf => Ok(DecisionTree::Leaf),
        DecisionTree::Probe {
            row: pr,
            col: pc,
            on_empty,
            on_hit,
        } => {
            if *pr as usize != row || *pc as usize != col {
                return Err(());
            }
            if result == 0 {
                return Ok((**on_empty).clone());
            }
            let treasure_type = result - 1;
            let mut candidates = on_hit.iter().filter(|(info, _)| {
                if info.treasure_type != treasure_type {
                    return false;
                }
                let is_horizontal = info.placed_w >= info.placed_h;
                match orientation {
                    Orientation::Horizontal => is_horizontal,
                    Orientation::Vertical => !is_horizontal,
                }
            });
            if let Some((_, sub)) = candidates.next() {
                Ok((**sub).clone())
            } else {
                Err(())
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw(
    stdout: &mut io::Stdout,
    tree: Option<&DecisionTree>,
    computing: bool,
    started_at: Option<Instant>,
    spinner_idx: &mut usize,
    selected_row: usize,
    selected_col: usize,
    selected_result: u8,
    selected_orientation: Orientation,
    message: &str,
) -> io::Result<()> {
    let spinner = ["|", "/", "-", "\\"];
    *spinner_idx = (*spinner_idx + 1) % spinner.len();
    queue!(
        stdout,
        cursor::MoveTo(0, 0),
        terminal::Clear(ClearType::All),
        cursor::MoveTo(0, 0)
    )?;

    writeln!(stdout, "Treasure Hunt TUI ({}x{})", WIDTH, HEIGHT)?;
    if computing {
        let elapsed = started_at.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
        writeln!(
            stdout,
            "Status: computing {} ({elapsed:.1}s)",
            spinner[*spinner_idx]
        )?;
        writeln!(stdout, "Only 'q' works while computing.")?;
    } else {
        writeln!(stdout, "Status: idle")?;
    }

    let depth_map = depth_map(tree);

    for (r, row) in depth_map.iter().enumerate() {
        for (c, depth) in row.iter().enumerate() {
            let text = match depth {
                Some(d) => d.to_string(),
                None => "?".to_string(),
            };
            if r == selected_row && c == selected_col {
                write!(stdout, "[{text:^3}]")?;
            } else {
                write!(stdout, " {text:^3} ")?;
            }
        }
        writeln!(stdout)?;
    }

    writeln!(stdout, "")?;
    writeln!(stdout, "Selected result: {selected_result}")?;
    writeln!(
        stdout,
        "Orientation: {} (toggle with j/k)",
        selected_orientation.as_str()
    )?;
    writeln!(stdout, "{message}")?;
    writeln!(
        stdout,
        "Keys: o=compute, arrows=move, Enter=select cell, 0/1/2/3=result, Y=confirm, n=exit step, q=quit"
    )?;
    stdout.flush()
}

fn depth_map(tree: Option<&DecisionTree>) -> [[Option<u8>; WIDTH]; HEIGHT] {
    let mut map = [[None; WIDTH]; HEIGHT];
    if let Some(t) = tree {
        fill_depths(t, 1, &mut map);
    }
    map
}

fn fill_depths(tree: &DecisionTree, depth: u8, map: &mut [[Option<u8>; WIDTH]; HEIGHT]) {
    match tree {
        DecisionTree::Leaf => {}
        DecisionTree::Probe {
            row,
            col,
            on_empty,
            on_hit,
        } => {
            let slot = &mut map[*row as usize][*col as usize];
            *slot = Some(slot.map_or(depth, |d| d.min(depth)));
            fill_depths(on_empty, depth + 1, map);
            for (_, sub) in on_hit {
                fill_depths(sub, depth + 1, map);
            }
        }
    }
}
