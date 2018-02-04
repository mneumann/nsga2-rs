use multi_objective::MultiObjective;
use non_dominated_sort::{NonDominatedSort, SolutionWithIndex};
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

    for front in NonDominatedSort::new(solutions, multi_objective) {
        if result.len() + front.solutions.len() <= n {
            // the whole front fits into the result

            // NOTE: I think in the original paper, the authors would
            // not defined the crowding distance. But in order to select
            // parents to reproduce, having the crowding distance
            // readily available is desired.
            let (mut assigned_crowding, _) = assign_crowding_distance(&front, multi_objective);

            for s in assigned_crowding {
                result.push(s);
                debug_assert!(result.len() <= n);
            }
        } else {
            // the front does not fit in total.
            // we will sort it's solutions according to the crowding distance and take the best
            // until we have "n" solutions in the result

            let (mut assigned_crowding, _) = assign_crowding_distance(&front, multi_objective);

            assigned_crowding.sort_by(|a, b| {
                debug_assert_eq!(a.rank, b.rank);
                a.crowding_distance
                    .partial_cmp(&b.crowding_distance)
                    .unwrap()
                    .reverse()
            });

            debug_assert!(n >= result.len());

            for s in assigned_crowding.into_iter().take(n - result.len()) {
                result.push(s);
                debug_assert!(result.len() <= n);
            }

            debug_assert_eq!(n, result.len());
            break;
        }

        if result.len() >= n {
            break;
        }
    }

    debug_assert_eq!(n, result.len());

    return result;
}
