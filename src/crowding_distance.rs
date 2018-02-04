use multi_objective::MultiObjective;
use non_dominated_sort::Front;
use std::f64::INFINITY;

pub struct AssignedCrowdingDistance<'a, S>
where
    S: 'a,
{
    index: usize,
    solution: &'a S,
    rank: usize,
    crowding_distance: f64,
}

pub struct ObjectiveStat {
    spread: f64,
}

/// Assigns a crowding distance to each solution in `front`.
pub fn assign_crowding_distance<'a, S>(
    front: &Front<'a, S>,
    multi_objective: &MultiObjective<S, f64>,
) -> (Vec<AssignedCrowdingDistance<'a, S>>, Vec<ObjectiveStat>) {
    let mut a: Vec<_> = front
        .solutions
        .iter()
        .map(|i| AssignedCrowdingDistance {
            index: i.index,
            solution: i.solution,
            rank: front.rank,
            crowding_distance: 0.0,
        })
        .collect();

    let objective_stat: Vec<_> = multi_objective
        .objectives
        .iter()
        .map(|objective| {
            // First, sort according to objective
            a.sort_by(|a, b| objective.total_order(a.solution, b.solution));

            // Assign infinite crowding distance to the extremes
            {
                a.first_mut().unwrap().crowding_distance = INFINITY;
                a.last_mut().unwrap().crowding_distance = INFINITY;
            }

            // The distance between the "best" and "worst" solution
            // according to "objective".
            let spread = objective
                .distance(a.first().unwrap().solution, a.last().unwrap().solution)
                .abs();
            debug_assert!(spread >= 0.0);

            if spread > 0.0 {
                let norm = 1.0 / (spread * (multi_objective.objectives.len() as f64));
                debug_assert!(norm > 0.0);

                for i in 1..a.len() - 1 {
                    debug_assert!(i >= 1 && i + 1 < a.len());

                    let distance = objective
                        .distance(a[i + 1].solution, a[i - 1].solution)
                        .abs();
                    debug_assert!(distance >= 0.0);
                    a[i].crowding_distance += distance * norm;
                }
            }

            ObjectiveStat { spread }
        })
        .collect();

    return (a, objective_stat);
}

#[test]
fn test_crowding_distance() {
    use test_helper_objective::{Objective1, Objective2, Tuple};
    use non_dominated_sort::NonDominatedSort;

    // construct a multi objective over a Tuple
    let mo = MultiObjective::<Tuple, f64>::new(&[&Objective1, &Objective2]);

    let a = Tuple(1, 3);
    let b = Tuple(3, 1);
    let c = Tuple(3, 3);
    let d = Tuple(2, 2);

    let solutions = vec![a, b, c, d];

    let fronts = NonDominatedSort::new(&solutions, &mo).pareto_fronts();
    assert_eq!(2, fronts.len());
    let f0 = &fronts[0];
    let f1 = &fronts[1];

    assert_eq!(3, f0.solutions.len());
    assert_eq!(&a, f0.solutions[0].solution);
    assert_eq!(&b, f0.solutions[1].solution);
    assert_eq!(&d, f0.solutions[2].solution);
    assert_eq!(1, f1.solutions.len());
    assert_eq!(&c, f1.solutions[0].solution);

    let (crowding, stat) = assign_crowding_distance(f0, &mo);

    assert_eq!(2, stat.len());
    assert_eq!(2.0, stat[0].spread);
    assert_eq!(2.0, stat[1].spread);

    // Same number as solutions in front
    assert_eq!(3, crowding.len());
    // All have rank 0
    assert_eq!(0, crowding[0].rank);
    assert_eq!(0, crowding[1].rank);
    assert_eq!(0, crowding[2].rank);

    let ca = crowding.iter().find(|i| i.solution.eq(&a)).unwrap();
    let cb = crowding.iter().find(|i| i.solution.eq(&b)).unwrap();
    let cd = crowding.iter().find(|i| i.solution.eq(&d)).unwrap();

    assert_eq!(INFINITY, ca.crowding_distance);
    assert_eq!(INFINITY, cb.crowding_distance);

    // only cd is in the middle. spread is in both dimensions the same
    // (2.0). norm is 1.0 / (spread * #objectives) = 1.0 / 4.0. As we
    // add two times 0.5, the crowding distance should be 1.0.
    assert_eq!(1.0, cd.crowding_distance);
}
