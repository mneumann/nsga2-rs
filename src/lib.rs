extern crate rand;
extern crate rayon;
extern crate time;

use rand::Rng;
use mo::MultiObjective;
pub use self::domination::{Domination, Dominate, DominationHelper};
pub use population::{UnratedPopulation, RatedPopulation, SelectedPopulation};

pub mod selection;
pub mod mo;
pub mod domination;
pub mod crowding_distance;
pub mod population;

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
            let next_generation = parents.merge(rated_offspring);
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

#[test]
fn test_dominates() {
    use mo::MultiObjective2;

    let a = MultiObjective2::from((1.0f32, 0.1));
    let b = MultiObjective2::from((0.1f32, 0.1));
    let c = MultiObjective2::from((0.1f32, 1.0));

    assert_eq!(false, a.dominates(&a));
    assert_eq!(false, a.dominates(&b));
    assert_eq!(false, a.dominates(&c));

    assert_eq!(true, b.dominates(&a));
    assert_eq!(false, b.dominates(&b));
    assert_eq!(true, b.dominates(&c));

    assert_eq!(false, c.dominates(&a));
    assert_eq!(false, c.dominates(&b));
    assert_eq!(false, c.dominates(&c));
}

#[test]
fn test_abc() {
    use mo::MultiObjective2;
    use domination::fast_non_dominated_sort;
    use selection::select_solutions;
    use crowding_distance::crowding_distance_assignment;

    let mut solutions: Vec<MultiObjective2<f32>> = Vec::new();
    solutions.push(MultiObjective2::from((1.0, 0.1)));
    solutions.push(MultiObjective2::from((0.1, 0.1)));
    solutions.push(MultiObjective2::from((0.1, 1.0)));
    solutions.push(MultiObjective2::from((0.5, 0.5)));
    solutions.push(MultiObjective2::from((0.5, 0.5)));

    println!("solutions: {:?}", solutions);
    let selection = select_solutions(&solutions[..], 5, 2, &mut DominationHelper);
    println!("selection: {:?}", selection);

    let fronts = fast_non_dominated_sort(&solutions[..], 10, &mut DominationHelper);
    println!("solutions: {:?}", solutions);
    println!("fronts: {:?}", fronts);

    for (rank, front) in fronts.iter().enumerate() {
        let distances = crowding_distance_assignment(&solutions[..], rank as u32, &front[..], 2);
        println!("front: {:?}", front);
        println!("distances: {:?}", distances);
    }
}
