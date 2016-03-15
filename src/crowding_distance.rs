use multi_objective::MultiObjective;
use std::f64::INFINITY;

pub trait CrowdingDistanceAssignment<F> where F: MultiObjective {
    fn fitness(&self) -> &F;
    fn rank_mut(&mut self) -> &mut u32;
    fn dist_mut(&mut self) -> &mut f64;
    fn rank(&self) -> u32;
    fn dist(&self) -> f64;

    fn set_rank(&mut self, rank: u32) {
        *self.rank_mut() = rank;
    }

    fn set_dist(&mut self, dist: f64) {
        *self.dist_mut() = dist;
    }

    fn select(&mut self); 
    fn unselect(&mut self); 
    fn is_selected(&self) -> bool; 

    #[inline]
    fn has_better_rank_and_crowding(&self, other: &Self) -> bool {
        (self.rank() < other.rank()) || 
        ((self.rank() == other.rank()) && self.dist() > other.dist())
    }
}

pub fn crowding_distance_assignment<T, F>(solutions: &mut [T],
                                          front_indices: &mut Vec<usize>,
                                          common_rank: u32,
                                          num_objectives: usize) where T: CrowdingDistanceAssignment<F>,
                                                                       F: MultiObjective {
    assert!(num_objectives > 0);
    assert!(num_objectives <= F::NUM_OBJECTIVES);

    // Initialize all solutions of that pareto front 
    for &i in front_indices.iter() {
        solutions[i].set_rank(common_rank);
        solutions[i].set_dist(0.0);
        solutions[i].unselect();
    }

    let l = front_indices.len();

    for m in 0..num_objectives {
        // sort front_indices according to objective `m`
        front_indices.sort_by(|&a, &b| solutions[a].fitness().cmp_objective(solutions[b].fitness(), m));

        let min_idx = front_indices[0];
        let max_idx = front_indices[l - 1];

        // assign infinite crowding distance to edges.
        *solutions[min_idx].dist_mut() = INFINITY;
        *solutions[max_idx].dist_mut() = INFINITY;

        let dist_max_min = solutions[max_idx].fitness().dist_objective(solutions[min_idx].fitness(), m).abs();
        debug_assert!(dist_max_min >= 0.0);
        if dist_max_min != 0.0 {
            let norm = num_objectives as f64 * dist_max_min;
            debug_assert!(norm != 0.0);
            for i in 1..(l - 1) {
                let prev_idx = front_indices[i - 1];
                let curr_idx = front_indices[i];
                let next_idx = front_indices[i + 1];

                let dist = solutions[next_idx].fitness().dist_objective(solutions[prev_idx].fitness(), m).abs();
                debug_assert!(dist >= 0.0);
                *solutions[curr_idx].dist_mut() += dist / norm;
            }
        }
    }
}
