use std::cmp::Ordering;
use multi_objective::MultiObjective;
use std::f32;

#[derive(Debug)]
pub struct SolutionRankDist {
    pub rank: u32,
    pub dist: f32,
    pub idx: usize,
}

impl PartialEq for SolutionRankDist {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.dist == other.dist
    }
}

// Implement the crowding-distance comparison operator.
impl PartialOrd for SolutionRankDist {
    #[inline]
    // compare on rank first (ASC), then on dist (DESC)
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.rank.partial_cmp(&other.rank) {
            Some(Ordering::Equal) => {
                // first criterion equal, second criterion decides
                // reverse ordering
                self.dist.partial_cmp(&other.dist).map(|i| i.reverse())
            }
            other => other,
        }
    }
}

pub fn crowding_distance_assignment<P: MultiObjective>(solutions: &[P],
                                                       common_rank: u32,
                                                       individuals_idx: &[usize],
                                                       num_objectives: usize)
                                                       -> Vec<SolutionRankDist> {
    assert!(num_objectives > 0);

    let mut s: Vec<_> = individuals_idx.iter()
                                       .map(|&i| {
                                           SolutionRankDist {
                                               rank: common_rank,
                                               dist: 0.0,
                                               idx: i,
                                           }
                                       })
                                       .collect();
    let l = s.len();

    for m in 0..num_objectives {
        // sort using objective `m`
        s.sort_by(|a, b| solutions[a.idx].cmp_objective(&solutions[b.idx], m));

        // assign infinite crowding distance to edges.
        s[0].dist = f32::INFINITY;
        s[l - 1].dist = f32::INFINITY;

        let min_idx = s[0].idx;
        let max_idx = s[l - 1].idx;

        let dist_max_min = solutions[max_idx].dist_objective(&solutions[min_idx], m).abs();
        if dist_max_min != 0.0 {
            let norm = num_objectives as f32 * dist_max_min;
            debug_assert!(norm != 0.0);
            for i in 1..(l - 1) {
                let next_idx = s[i + 1].idx;
                let prev_idx = s[i - 1].idx;
                let dist = solutions[next_idx].dist_objective(&solutions[prev_idx], m).abs();
                debug_assert!(dist >= 0.0);
                s[i].dist += dist / norm;
            }
        }
    }

    return s;
}
