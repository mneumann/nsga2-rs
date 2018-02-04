use multi_objective::MultiObjective;
use crowding_distance::AssignedCrowdingDistance;

pub trait SelectAndRank {
    fn select_and_rank<'a, S: 'a>(
        &self,
        solutions: &'a [S],
        n: usize,
        multi_objective: &MultiObjective<S, f64>,
    ) -> Vec<AssignedCrowdingDistance<'a, S>>;
}
