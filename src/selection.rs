use multi_objective::MultiObjective;
use non_dominated_sort::NonDominatedSort;
use crowding_distance::{assign_crowding_distance, AssignedCrowdingDistance};
use std::cmp::PartialOrd;

/// Select `n` solutions using the approach taken by NSGA. We first sort
/// the solutions into their corresponding pareto fronts. Then, we put
/// as many "whole" fronts into the result as possible, until we cannot
/// fit a whole front into the result (otherwise we would have more than
/// `n` solutions). For this last front, sort it's solutions according
/// to their crowding distance (higher crowding distance is better!),
/// and chose those solutions with the higher crowding distance.

pub fn selection_nsga<'a, S>(
    solutions: &'a [S],
    n: usize,
    multi_objective: &MultiObjective<S, f64>,
) -> Vec<AssignedCrowdingDistance<'a, S>>
where
    S: 'a,
{
    // Cannot select more solutions than we actually have
    let n = solutions.len().min(n);
    debug_assert!(n <= solutions.len());

    let mut result = Vec::with_capacity(n);
    let mut missing_solutions = n;

    for front in NonDominatedSort::new(solutions, multi_objective) {
        let (mut assigned_crowding, _) = assign_crowding_distance(&front, multi_objective);

        if assigned_crowding.len() > missing_solutions {
            // the front does not fit in total. sort it's solutions
            // according to the crowding distance and take the best
            // solutions until we have "n" solutions in the result.

            assigned_crowding.sort_by(|a, b| {
                debug_assert_eq!(a.rank, b.rank);
                a.crowding_distance
                    .partial_cmp(&b.crowding_distance)
                    .unwrap()
                    .reverse()
            });
        }

        // Take no more than `missing_solutions`
        let take = assigned_crowding.len().min(missing_solutions);

        result.extend(assigned_crowding.into_iter().take(take));

        missing_solutions -= take;
        if missing_solutions == 0 {
            break;
        }
    }

    debug_assert_eq!(n, result.len());

    return result;
}
