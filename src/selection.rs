use rand::Rng;
use multi_objective::MultiObjective;
use domination::Domination;
use sort::FastNonDominatedSorter;
use crowding_distance::{SolutionRankDist, crowding_distance_assignment};
use std::cmp;

/// Select `n` out of the `solutions`, assigning rank and distance using the first `num_objectives`
/// objectives.

pub fn select_solutions<T, D>(solutions: &[T],
                              n: usize,
                              num_objectives: usize,
                              domination: &mut D)
                              -> Vec<SolutionRankDist>
    where T: MultiObjective,
          D: Domination<T>
{
    let mut selection = Vec::with_capacity(cmp::min(solutions.len(), n));

    let pareto_fronts = FastNonDominatedSorter::new(solutions, domination);

    for (rank, front) in pareto_fronts.enumerate() {
        if selection.len() >= n {
            break;
        }
        let missing = n - selection.len();

        let mut solution_rank_dist = crowding_distance_assignment(solutions,
                                                                  rank as u32,
                                                                  &front,
                                                                  num_objectives);
        if solution_rank_dist.len() <= missing {
            // whole front fits into result.
            // XXX: should we sort?
            selection.extend(solution_rank_dist);
            assert!(selection.len() <= n);
        } else {
            // choose only best from this front, according to the crowding distance.
            solution_rank_dist.sort_by(|a, b| {
                debug_assert!(a.rank == b.rank);
                a.partial_cmp(b).unwrap()
            });
            selection.extend(solution_rank_dist.into_iter().take(missing));
            assert!(selection.len() == n);
            break;
        }
    }

    assert!(selection.len() == cmp::min(solutions.len(), n));
    return selection;
}



/// Select the best individual out of `k` randomly choosen.
/// This gives individuals with better fitness a higher chance to reproduce.
/// `n` is the total number of individuals.
///
/// NOTE: We are not using `sample(rng, 0..n, k)` as it is *very* expensive.
/// Instead we call `rng.gen_range()` k-times. The drawn items could be the same,
/// but the probability is very low if `n` is high compared to `k`.
#[inline]
pub fn tournament_selection_fast<R: Rng, F>(rng: &mut R,
                                            better_than: F,
                                            n: usize,
                                            k: usize)
                                            -> usize
    where F: Fn(usize, usize) -> bool
{
    assert!(n > 0);
    assert!(k > 0);
    assert!(n >= k);

    let mut best: usize = rng.gen_range(0, n);

    for _ in 1..k {
        let i = rng.gen_range(0, n);
        if better_than(i, best) {
            best = i;
        }
    }

    return best;
}
