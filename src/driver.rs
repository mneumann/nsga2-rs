use rand::Rng;
use domination::Domination;
use multi_objective::MultiObjective;
use population::{Individual, UnratedPopulation, RatedPopulation, RankedPopulation};
use selection::SelectSolutions;
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
    type GENOME: Clone + Sync + Send; // XXX: clone?
    type FIT: MultiObjective + Domination + Clone + Send; // XXX: clone?
    type SELECTION: SelectSolutions<Individual<Self::GENOME, Self::FIT>, Self::FIT>;

    fn random_genome<R>(&self, rng: &mut R) -> Self::GENOME where R: Rng;
    fn initial_population<R>(&self, rng: &mut R, n: usize) -> UnratedPopulation<Self::GENOME, Self::FIT> where R: Rng {
        let mut pop = UnratedPopulation::new();
        for _ in 0..n {
            pop.push(self.random_genome(rng));
        }
        assert!(pop.len() == n);
        return pop;
    }
    fn fitness(&self, ind: &Self::GENOME) -> Self::FIT;
    // XXX: Make use of Fitness!
    fn mate<R>(&self, rng: &mut R, parent1: &Self::GENOME, parent2: &Self::GENOME) -> Self::GENOME where R: Rng;
    fn is_solution(&self, _ind: &Self::GENOME, _fit: &Self::FIT) -> bool {
        false
    }

    /// This can be used to update certain objectives in relation to the whole population.
    fn population_metric(&self, _population: &mut RatedPopulation<Self::GENOME, Self::FIT>) {
    }

    fn run<R, L>(&self,
                 rng: &mut R,
                 config: &DriverConfig,
                 logger: &L)
                 -> RankedPopulation<Self::GENOME, Self::FIT>
        where R: Rng,
              L: Fn(usize, u64, usize, &RankedPopulation<Self::GENOME, Self::FIT>)
    {
        // this is generation 0. it's empty
        let mut time_last = time::precise_time_ns();
        let mut parents = RankedPopulation::<Self::GENOME, Self::FIT>::new();
        let mut offspring = self.initial_population(rng, config.mu);
        assert!(offspring.len() == config.mu);
        let mut gen: usize = 0;

        loop {
            let rated_offspring = offspring.rate_in_parallel(&|ind| self.fitness(ind), config.parallel_weight);
            let mut next_generation = parents.merge(rated_offspring);
            // apply a population metric on the whole population
            self.population_metric(&mut next_generation);
            let mut selection = Self::SELECTION::new(rng); 
            parents = next_generation.select(config.mu, config.num_objectives, &mut selection);

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
