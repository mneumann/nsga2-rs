use rand::Rng;
use multi_objective::MultiObjective;
use domination::Domination;
use sort::FastNonDominatedSorter;
use crowding_distance::{crowding_distance_assignment, CrowdingDistanceAssignment};
use std::cmp::{self, Ordering};

pub trait SelectSolutions<T, F>: Sized
where
    T: CrowdingDistanceAssignment<F>,
    F: MultiObjective + Domination,
{
    /// Select `n` out of the `solutions`, assigning rank and distance using `objectives`.
    /// `rng` might not be used.
    fn select_solutions<R>(
        &self,
        solutions: &mut [T],
        select_n: usize,
        objectives: &[usize],
        rng: &mut R,
    ) where
        R: Rng;
}


/// Uses the approach NSGA takes.
pub struct SelectNSGA;

impl<T, F> SelectSolutions<T, F> for SelectNSGA
where
    T: CrowdingDistanceAssignment<F>,
    F: MultiObjective + Domination,
{
    fn select_solutions<R>(
        &self,
        solutions: &mut [T],
        select_n: usize,
        objectives: &[usize],
        _rng: &mut R,
    ) where
        R: Rng,
    {
        assert!(objectives.len() > 0);

        // We can select at most `select_n` solutions.
        let mut missing = cmp::min(solutions.len(), select_n);

        // make sure all solutions are unselected
        for s in solutions.iter_mut() {
            s.unselect();
        }

        let pareto_fronts = FastNonDominatedSorter::new(solutions, &|s| s.fitness(), objectives);

        for (rank, mut front) in pareto_fronts.enumerate() {
            assert!(missing > 0);

            // assign rank and crowding distance of those solutions in `front`.
            let _ = crowding_distance_assignment(solutions, &mut front, rank as u32, objectives);

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
}

/// Selection using NSGP as described in [1].
///
/// [1]: Multi-objective genetic programming with redundancy-regulations for automatic construction
/// of image feature extractors.  Watchareeruetai, Ukrit and Matsumoto, Tetsuya and Takeuchi,
/// Yoshinori and Hiroaki, KUDO and Ohnishi, Noboru.  2010
pub struct SelectNSGP {
    pub objective_eps: f64,
}

impl<T, F> SelectSolutions<T, F> for SelectNSGP
where
    T: CrowdingDistanceAssignment<F>,
    F: MultiObjective + Domination,
{
    fn select_solutions<R>(
        &self,
        solutions: &mut [T],
        select_n: usize,
        objectives: &[usize],
        rng: &mut R,
    ) where
        R: Rng,
    {
        assert!(objectives.len() > 0);

        // We can select at most `select_n` solutions.
        let mut missing = cmp::min(solutions.len(), select_n);

        // make sure all solutions are unselected
        for s in solutions.iter_mut() {
            s.unselect();
        }

        let pareto_fronts = FastNonDominatedSorter::new(solutions, &|s| s.fitness(), objectives);

        let mut fronts_grouped: Vec<Vec<Vec<_>>> = Vec::new();
        for (rank, mut front) in pareto_fronts.enumerate() {

            // first assign rank and crowding distance of those solutions in `front`.
            let min_max_distances =
                crowding_distance_assignment(solutions, &mut front, rank as u32, objectives);
            debug_assert!(min_max_distances.len() == objectives.len());

            // then group this front into individuals with similar (or equal) fitness.
            let mut groups: Vec<Vec<usize>> = Vec::new();

            'group: for idx in front {
                // insert `idx` into a group.
                // test fitness of `idx` (solutions[idx]) against a member of each group.
                // if none fits, create a new group.

                let fitness = solutions[idx].fitness();
                for grp in groups.iter_mut() {
                    let cmp_idx = grp[0];

                    if fitness.similar_to(
                        solutions[cmp_idx].fitness(),
                        &objectives,
                        &min_max_distances,
                        self.objective_eps,
                    )
                    {
                        grp.push(idx);
                        continue 'group;
                    }
                }

                // create a new group
                groups.push(vec![idx]);
            }

            // update the crowding_distance within each group. simply take the average.
            // also record the number of points in the group within each individual's fitness. this
            // is required for sexual selection.
            for grp in groups.iter_mut() {
                // use same average crowding distance for each point in the group
                assert!(grp.len() > 0);
                let avg: f64 = grp.iter().map(|&idx| solutions[idx].dist()).fold(
                    0.0,
                    |acc, x| acc + x,
                ) / grp.len() as f64;
                for &idx in grp.iter() {
                    solutions[idx].set_dist(avg);
                    solutions[idx].set_crowd(grp.len());
                }

                // sort grp according to primary fitness
                grp.sort_by(|&a, &b| {
                    solutions[a].fitness().cmp_objective(
                        solutions[b].fitness(),
                        objectives[0],
                    )
                });
            }

            // group fronts, so that the first group is the one which contains
            // the best individual of fitness objective #0. this makes
            // sure that we always ever include the elite
            groups.sort_by(|a, b| {
                solutions[a[0]].fitness().cmp_objective(
                    solutions[b[0]].fitness(),
                    objectives[0],
                )
            });

            fronts_grouped.push(groups);
        }

        // Each pareto_front has now been grouped into individuals of same (or similar) fitness.
        // Now go through each front, and in each front, choose exactly one individual
        // from each group. Then go to the next front etc. once we are through all
        // fronts, continue with the first and so on, until enough individuals are selected.

        let mut round = 0;
        'outer: while missing > 0 {
            for (front_i, current_front) in fronts_grouped.iter_mut().enumerate() {
                // select up to `missing` solutions from current_front, but from each group only one
                // (which is choosen randomly).
                for (grp_i, grp) in current_front.iter_mut().enumerate() {
                    if missing > 0 {
                        let i = if round == 0 && front_i == 0 && grp_i == 0 {
                            // first round, first group and non-dominated front.
                            // Use always the best according to 0th objective.
                            0
                        } else {
                            // Select randomly
                            rng.gen_range(0, grp.len())
                        };
                        // remove `i`th element from grp returning the solution index
                        let idx = grp.swap_remove(i);
                        assert!(solutions[idx].is_selected() == false);
                        solutions[idx].select();
                        missing -= 1;
                    } else {
                        break 'outer;
                    }
                }

                // only retain groups which are non-empty
                current_front.retain(|grp| grp.len() > 0);
            }
            round += 1;
        }
    }
}


/// Selection using a modified version of NSGP [1].
pub struct SelectNSGPMod {
    pub objective_eps: f64,
}

impl<T, F> SelectSolutions<T, F> for SelectNSGPMod
where
    T: CrowdingDistanceAssignment<F>,
    F: MultiObjective + Domination,
{
    fn select_solutions<R>(
        &self,
        solutions: &mut [T],
        select_n: usize,
        objectives: &[usize],
        rng: &mut R,
    ) where
        R: Rng,
    {
        assert!(objectives.len() > 0);

        // We can select at most `select_n` solutions.
        let mut missing = cmp::min(solutions.len(), select_n);

        // make sure all solutions are unselected
        for s in solutions.iter_mut() {
            s.unselect();
        }

        let pareto_fronts = FastNonDominatedSorter::new(solutions, &|s| s.fitness(), objectives);

        let mut fronts_grouped: Vec<Vec<Vec<Vec<_>>>> = Vec::new();
        for (rank, mut front) in pareto_fronts.enumerate() {
            // first assign rank and crowding distance of
            // those solutions in `front`.
            let min_max_distances =
                crowding_distance_assignment(solutions, &mut front, rank as u32, objectives);
            debug_assert!(min_max_distances.len() == objectives.len());

            // then group this front into individuals with similar (or equal) fitness.
            // we have one group per objective
            let mut objective_groups = Vec::new();

            for (&obj, &min_max) in objectives.iter().zip(min_max_distances.iter()) {
                let mut groups: Vec<Vec<usize>> = Vec::new();

                'group: for &idx in front.iter() {
                    // insert `idx` into a group.
                    // test fitness of `idx` (solutions[idx]) against a member of each group.
                    // if none fits, create a new group.

                    let fitness = solutions[idx].fitness();
                    for grp in groups.iter_mut() {
                        let cmp_idx = grp[0];

                        if fitness.similar_to(
                            solutions[cmp_idx].fitness(),
                            &[obj],
                            &[min_max],
                            self.objective_eps,
                        )
                        {
                            grp.push(idx);
                            continue 'group;
                        }
                    }

                    // create a new group
                    groups.push(vec![idx]);
                }

                // update the crowding_distance within each group. simply take the average.
                // also record the number of points in the group within each individual's fitness. this
                // is required for sexual selection.
                for grp in groups.iter_mut() {
                    // use same average crowding distance for each point in the group
                    assert!(grp.len() > 0);
                    let avg: f64 = grp.iter().map(|&idx| solutions[idx].dist()).fold(
                        0.0,
                        |acc, x| acc + x,
                    ) / grp.len() as f64;
                    for &idx in grp.iter() {
                        solutions[idx].set_dist(avg);
                        solutions[idx].set_crowd(grp.len()); // XXX
                    }

                    // sort grp according to objective
                    grp.sort_by(|&a, &b| {
                        solutions[a].fitness().cmp_objective(
                            solutions[b].fitness(),
                            obj,
                        )
                    });
                }

                // group fronts, so that the first group is the one which contains
                // the best individual of fitness objective #0. this makes
                // sure that we always ever include the elite
                groups.sort_by(|a, b| {
                    solutions[a[0]].fitness().cmp_objective(
                        solutions[b[0]].fitness(),
                        obj,
                    )
                });

                objective_groups.push(groups);
            }

            fronts_grouped.push(objective_groups);
        }

        // Each pareto_front has now been grouped into individuals of same (or similar) fitness.
        // Now go through each front, and in each front, choose exactly one individual
        // from each group. Then go to the next front etc. once we are through all
        // fronts, continue with the first and so on, until enough individuals are selected.
        // XXX

        let mut round = 0;
        'outer: while missing > 0 {
            for (front_i, current_front) in fronts_grouped.iter_mut().enumerate() {
                // select up to `missing` solutions from current_front, but from each group only one
                // (which is choosen randomly).
                for objective_grp in current_front.iter_mut() {
                    'next_group: for (grp_i, grp) in objective_grp.iter_mut().enumerate() {
                        if missing > 0 {
                            loop {
                                if grp.is_empty() {
                                    break;
                                }
                                //let i = if round == 0 && front_i == 0 && grp_i == 0 { // XXX?
                                // first round, first group and non-dominated front. Use always the best according
                                // to 0th objective.
                                let i = rng.gen_range(0, grp.len());
                                //} else {
                                // Select randomly
                                //rng.gen_range(0, grp.len())
                                //};
                                let idx = grp.swap_remove(i); // remove `i`th element from grp returning the solution index
                                if !solutions[idx].is_selected() {
                                    solutions[idx].select();
                                    missing -= 1;
                                    break;
                                }
                            }
                        } else {
                            break 'outer;
                        }
                    }
                }

                // only retain groups which are non-empty
                current_front.retain(|grp| grp.len() > 0);
            }
            round += 1;
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
pub fn tournament_selection_fast<R: Rng, F>(
    rng: &mut R,
    better_than: F,
    n: usize,
    k: usize,
) -> usize
where
    F: Fn(usize, usize) -> bool,
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
