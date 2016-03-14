use rand::Rng;
use multi_objective::MultiObjective;
use domination::Domination;
use sort::FastNonDominatedSorter;
use crowding_distance::{crowding_distance_assignment, CrowdingDistanceAssignment};
use std::cmp::{self, Ordering};

/// Select `n` out of the `solutions`, assigning rank and distance using the first `num_objectives`
/// objectives.

pub fn select_solutions<T, F, D>(solutions: &mut [T],
                                 select_n: usize,
                                 num_objectives: usize,
                                 domination: &mut D)
    where T: CrowdingDistanceAssignment<F>,
          F: MultiObjective,
          D: Domination<F>
{
    // We can select at most `select_n` solutions.
    let mut missing = cmp::min(solutions.len(), select_n);

    // make sure all solutions are unselected
    for s in solutions.iter_mut() {
        s.unselect();
    }

    let pareto_fronts = FastNonDominatedSorter::new(solutions, &|s| s.fitness(), domination);

    for (rank, mut front) in pareto_fronts.enumerate() {
        assert!(missing > 0);

        // assign rank and crowding distance of those solutions in `front`.
        crowding_distance_assignment(solutions,
                                     &mut front,
                                     rank as u32,
                                     num_objectives);

        if front.len() <= missing {
            // whole front fits into result. XXX: should we sort?

            for i in front.into_iter() {
                solutions[i].select();
                assert!(missing > 0);
                missing -= 1;
            }

            if missing == 0 {
                break;
            }
        } else {
            // choose only best from this front, according to the crowding distance.
            front.sort_by(|&a, &b| {
                debug_assert!(solutions[a].rank() == solutions[b].rank());
                if solutions[a].has_better_rank_and_crowding(&solutions[b]) {
                    Ordering::Less
                } else if solutions[b].has_better_rank_and_crowding(&solutions[a]) {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            });

            for i in front.into_iter().take(missing) {
                assert!(missing > 0);
                solutions[i].select();
                missing -= 1;
            }

            assert!(missing == 0);
            break;
        }
    }
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
