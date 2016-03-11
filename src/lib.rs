extern crate rand;
extern crate rayon;
extern crate time;

pub mod selection;
pub mod mo;
pub mod domination;
pub mod crowding_distance;
pub mod population;
pub mod driver;

#[test]
fn test_abc() {
    use mo::MultiObjective2;
    use domination::{fast_non_dominated_sort, DominationHelper};
    use selection::select_solutions;
    use crowding_distance::crowding_distance_assignment;

    let mut solutions: Vec<MultiObjective2<f32>> = Vec::new();
    solutions.push(MultiObjective2::from((1.0, 0.1)));
    solutions.push(MultiObjective2::from((0.1, 0.1)));
    solutions.push(MultiObjective2::from((0.1, 1.0)));
    solutions.push(MultiObjective2::from((0.5, 0.5)));
    solutions.push(MultiObjective2::from((0.5, 0.5)));

    println!("solutions: {:?}", solutions);
    let selection = select_solutions(&solutions[..], 5, 2, &mut DominationHelper);
    println!("selection: {:?}", selection);

    let fronts = fast_non_dominated_sort(&solutions[..], 10, &mut DominationHelper);
    println!("solutions: {:?}", solutions);
    println!("fronts: {:?}", fronts);

    for (rank, front) in fronts.iter().enumerate() {
        let distances = crowding_distance_assignment(&solutions[..], rank as u32, &front[..], 2);
        println!("front: {:?}", front);
        println!("distances: {:?}", distances);
    }
}
