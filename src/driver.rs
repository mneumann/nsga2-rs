use rand::Rng;
use domination::{Dominate, Domination};
use multi_objective::MultiObjective;
use population::{UnratedPopulation, RatedPopulation, SelectedPopulation};
use time;

pub struct DriverConfig {
    pub mu: usize,
    pub lambda: usize,
    pub k: usize,
    pub ngen: usize,
    pub num_objectives: usize,
    pub parallel_weight: f64,
}

pub trait Driver: Sync
{
    type IND: Clone + Sync;
    type FIT: Dominate + MultiObjective + Clone + Send;

    fn random_individual<R>(&self, rng: &mut R) -> Self::IND where R: Rng;
    fn initial_population<R>(&self, rng: &mut R, n: usize) -> UnratedPopulation<Self::IND> where R: Rng {
        (0..n).map(|_| self.random_individual(rng)).collect()
    }
    fn fitness(&self, ind: &Self::IND) -> Self::FIT;
    fn mate<R>(&self, rng: &mut R, parent1: &Self::IND, parent2: &Self::IND) -> Self::IND where R: Rng;
    fn is_solution(&self, _ind: &Self::IND, _fit: &Self::FIT) -> bool {
        false
    }

    /// This can be used to update certain objectives in relation to the whole population.
    fn population_metric(&self, _population: &mut RatedPopulation<Self::IND, Self::FIT>) {
    }

    fn run<R, D, L>(&self,
                    rng: &mut R,
                    config: &DriverConfig,
                    domination: &mut D,
                    logger: &L)
                    -> SelectedPopulation<Self::IND, Self::FIT>
        where R: Rng,
              D: Domination<Self::FIT>,
              L: Fn(usize, u64, usize, &SelectedPopulation<Self::IND, Self::FIT>)
    {
        // this is generation 0. it's empty
        let mut time_last = time::precise_time_ns();
        let mut parents = SelectedPopulation::<Self::IND, Self::FIT>::new();
        let mut offspring = self.initial_population(rng, config.mu);
        assert!(offspring.len() == config.mu);
        let mut gen: usize = 0;

        loop {
            let rated_offspring = offspring.rate_in_parallel(&|ind| self.fitness(ind), config.parallel_weight);
            let mut next_generation = parents.merge(rated_offspring);
            // apply a population metric on the whole population
            self.population_metric(&mut next_generation);
            parents = next_generation.select(config.mu, config.num_objectives, domination);

            let mut found_solutions = 0;
            parents.all(&mut |ind, fit| {
                if self.is_solution(ind, fit) {
                    found_solutions += 1;
                }
            });

            let now = time::precise_time_ns();
            let duration = now - time_last;
            time_last = now;

            logger(gen, duration, found_solutions, &parents);

            if found_solutions > 0 {
                break;
            }

            gen += 1;
            if gen > config.ngen {
                break;
            }

            offspring = parents.reproduce(rng,
                                          config.lambda,
                                          config.k,
                                          &|rng, p1, p2| self.mate(rng, p1, p2));
        }

        return parents;
    }
}
