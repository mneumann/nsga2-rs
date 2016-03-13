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
}

pub trait Driver<I, F>: Sync
where I: Clone + Sync,
      F: Dominate + MultiObjective + Clone + Send
{
    fn random_individual<R: Rng>(&self, rng: &mut R) -> I;
    fn random_population<R: Rng>(&self, rng: &mut R, n: usize) -> UnratedPopulation<I> {
        (0..n).map(|_| self.random_individual(rng)).collect()
    }
    fn fitness(&self, ind: &I) -> F;
    fn mate<R: Rng>(&self, rng: &mut R, parent1: &I, parent2: &I) -> I;
    fn is_solution(&self, _ind: &I, _fit: &F) -> bool {
        false
    }

    /// This can be used to update certain objectives in relation to the whole population.
    fn population_metric(&self, _population: &mut RatedPopulation<I, F>) {
    }

    fn run<R, D, L>(&self,
                    rng: &mut R,
                    config: &DriverConfig,
                    weight: f64,
                    domination: &mut D,
                    logger: &L)
                    -> SelectedPopulation<I, F>
        where R: Rng,
              D: Domination<F>,
              L: Fn(usize, u64, usize, &SelectedPopulation<I, F>)
    {
        // this is generation 0. it's empty
        let mut time_last = time::precise_time_ns();
        let mut parents = SelectedPopulation::<I, F>::new();
        let mut offspring = self.random_population(rng, config.mu);
        let mut gen: usize = 0;

        loop {
            let rated_offspring = offspring.rate_in_parallel(&|ind| self.fitness(ind), weight);
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
