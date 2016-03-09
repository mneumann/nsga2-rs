extern crate rand;
extern crate rayon;
extern crate time;

use std::cmp::{self, Ordering};
use std::f32;
use rand::Rng;
use selection::tournament_selection_fast;
use mo::MultiObjective;
pub use self::domination::Dominate;
use domination::fast_non_dominated_sort;
use rayon::par_iter::*;
use std::convert::{AsRef, From};
use std::iter::FromIterator;

pub mod selection;
pub mod mo;
pub mod domination;

impl<T: MultiObjective> Dominate for T {
    fn dominates(&self, other: &Self) -> bool {
        let mut less_cnt = 0;
        for i in 0..cmp::min(self.num_objectives(), other.num_objectives()) {
            match self.cmp_objective(other, i) {
                Ordering::Greater => {
                    return false;
                }
                Ordering::Less => {
                    less_cnt += 1;
                }
                Ordering::Equal => {}
            }
        }
        return less_cnt > 0;
    }
}

#[derive(Debug)]
pub struct SolutionRankDist {
    pub idx: usize,
    pub rank: u32,
    pub dist: f32,
}

impl PartialEq for SolutionRankDist {
    #[inline]
    fn eq(&self, other: &SolutionRankDist) -> bool {
        self.rank == other.rank && self.dist == other.dist
    }
}

// Implement the crowding-distance comparison operator.
impl PartialOrd for SolutionRankDist {
    #[inline]
    // compare on rank first (ASC), then on dist (DESC)
    fn partial_cmp(&self, other: &SolutionRankDist) -> Option<Ordering> {
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

fn crowding_distance_assignment<P: MultiObjective>(solutions: &[P],
                                                   common_rank: u32,
                                                   individuals_idx: &[usize],
                                                   num_objectives: usize)
                                                   -> Vec<SolutionRankDist> {
    assert!(num_objectives > 0);

    let l = individuals_idx.len();
    let mut distance: Vec<f32> = (0..l).map(|_| 0.0).collect();
    let mut indices: Vec<usize> = (0..l).map(|i| i).collect();

    for m in 0..num_objectives {
        // sort using objective `m`
        indices.sort_by(|&a, &b| {
            solutions[individuals_idx[a]].cmp_objective(&solutions[individuals_idx[b]], m)
        });
        distance[indices[0]] = f32::INFINITY;
        distance[indices[l - 1]] = f32::INFINITY;

        let min_idx = individuals_idx[indices[0]];
        let max_idx = individuals_idx[indices[l - 1]];

        let dist_max_min = solutions[max_idx].dist_objective(&solutions[min_idx], m);
        if dist_max_min != 0.0 {
            let norm = num_objectives as f32 * dist_max_min;
            debug_assert!(norm != 0.0);
            for i in 1..(l - 1) {
                let next_idx = individuals_idx[indices[i + 1]];
                let prev_idx = individuals_idx[indices[i - 1]];
                let dist = solutions[next_idx].dist_objective(&solutions[prev_idx], m);
                debug_assert!(dist >= 0.0);
                distance[indices[i]] += dist / norm;
            }
        }
    }

    return indices.iter()
                  .map(|&i| {
                      SolutionRankDist {
                          idx: individuals_idx[i],
                          rank: common_rank,
                          dist: distance[i],
                      }
                  })
                  .collect();
}

/// Select `n` out of the `solutions`, assigning rank and distance using the first `num_objectives`
/// objectives.
fn select_solutions<P: Dominate + MultiObjective>(solutions: &[P],
                                                  n: usize,
                                                  num_objectives: usize)
                                                  -> Vec<SolutionRankDist> {
    let mut selection = Vec::with_capacity(cmp::min(solutions.len(), n));

    let pareto_fronts = fast_non_dominated_sort(solutions, n, &mut|a,b| a.dominate(b));

    for (rank, front) in pareto_fronts.iter().enumerate() {
        if selection.len() >= n {
            break;
        }
        let missing: usize = n - selection.len();

        let mut solution_rank_dist = crowding_distance_assignment(solutions,
                                                                  rank as u32,
                                                                  &front[..],
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

    return selection;
}

/// An unrated Population of individuals.
pub struct UnratedPopulation<I>
    where I: Clone
{
    individuals: Vec<I>,
}

/// A rated Population of individuals, i.e.
/// each individual has a fitness assigned.
pub struct RatedPopulation<I, F>
    where I: Clone,
          F: Dominate + MultiObjective + Clone
{
    individuals: Vec<I>,
    fitness: Vec<F>,
}

/// A selected (ranked) Population.
pub struct SelectedPopulation<I, F>
    where I: Clone,
          F: Dominate + MultiObjective + Clone
{
    individuals: Vec<I>,
    fitness: Vec<F>,
    rank_dist: Vec<SolutionRankDist>,
}

impl<I> From<Vec<I>> for UnratedPopulation<I>
    where I: Clone
{
    fn from(v: Vec<I>) -> UnratedPopulation<I> {
        UnratedPopulation { individuals: v }
    }
}

impl<I> FromIterator<I> for UnratedPopulation<I>
    where I: Clone
{
    fn from_iter<T>(iterator: T) -> UnratedPopulation<I>
        where T: IntoIterator<Item = I>
    {
        UnratedPopulation { individuals: iterator.into_iter().collect() }
    }
}

impl<I> AsRef<[I]> for UnratedPopulation<I>
    where I: Clone
{
    fn as_ref(&self) -> &[I] {
        &self.individuals
    }
}

impl<I> UnratedPopulation<I>
    where I: Clone + Sync
{
    pub fn individuals(&self) -> &[I] {
        &self.individuals
    }

    pub fn new() -> UnratedPopulation<I> {
        UnratedPopulation { individuals: Vec::new() }
    }

    pub fn push(&mut self, ind: I) {
        self.individuals.push(ind);
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Rate the individuals.
    pub fn rate<F, E>(self, eval: &E) -> RatedPopulation<I, F>
        where F: Dominate + MultiObjective + Clone,
              E: Fn(&I) -> F
    {
        let fitness = self.individuals.iter().map(eval).collect();
        RatedPopulation {
            individuals: self.individuals,
            fitness: fitness,
        }
    }

    /// Rate the individuals in parallel.
    pub fn rate_in_parallel<F, E>(self, eval: &E, weight: f64) -> RatedPopulation<I, F>
        where F: Dominate + MultiObjective + Clone + Send,
              E: Sync + Fn(&I) -> F
    {
        let mut fitness = Vec::new();
        self.individuals.into_par_iter().map(eval).weight(weight).collect_into(&mut fitness);
        assert!(fitness.len() == self.individuals.len());

        RatedPopulation {
            individuals: self.individuals,
            fitness: fitness,
        }
    }
}

impl<I, F> RatedPopulation<I, F>
    where I: Clone,
          F: Dominate + MultiObjective + Clone
{
    pub fn select(self, population_size: usize, num_objectives: usize) -> SelectedPopulation<I, F> {
        // evaluate rank and crowding distance
        let rank_dist = select_solutions(&self.fitness, population_size, num_objectives);
        assert!(rank_dist.len() <= population_size);
        SelectedPopulation {
            individuals: self.individuals,
            fitness: self.fitness,
            rank_dist: rank_dist,
        }
    }

    pub fn new() -> RatedPopulation<I, F> {
        RatedPopulation {
            individuals: Vec::new(),
            fitness: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    pub fn fitness(&self) -> &[F] {
        &self.fitness
    }
}

impl<I, F> SelectedPopulation<I, F>
    where I: Clone,
          F: Dominate + MultiObjective + Clone
{
    /// Generate an offspring population.
    /// XXX: Factor out selection into a separate Trait  SelectionMethod
    pub fn reproduce<R, M>(&self,
                           rng: &mut R,
                           offspring_size: usize,
                           tournament_k: usize,
                           mate: &M)
                           -> UnratedPopulation<I>
        where R: Rng,
              M: Fn(&mut R, &I, &I) -> I
    {
        assert!(self.individuals.len() == self.fitness.len());
        assert!(self.individuals.len() >= self.rank_dist.len());
        assert!(tournament_k > 0);

        let rank_dist = &self.rank_dist[..];

        // create `offspring_size` new offspring using k-tournament (
        // select the best individual out of k randomly choosen individuals)
        let offspring: Vec<I> =
            (0..offspring_size)
                .map(|_| {

                    // first parent. k candidates
                    let p1 = tournament_selection_fast(rng,
                                                       |i1, i2| rank_dist[i1] < rank_dist[i2],
                                                       rank_dist.len(),
                                                       tournament_k);

                    // second parent. k candidates
                    let p2 = tournament_selection_fast(rng,
                                                       |i1, i2| rank_dist[i1] < rank_dist[i2],
                                                       rank_dist.len(),
                                                       tournament_k);

                    // cross-over the two parents and produce one child (throw away
                    // second child XXX)

                    // The potentially dominating individual is gives as first
                    // parameter.
                    let (p1, p2) = if rank_dist[p1] < rank_dist[p2] {
                        (p1, p2)
                    } else if rank_dist[p2] < rank_dist[p1] {
                        (p2, p1)
                    } else {
                        (p1, p2)
                    };

                    mate(rng, &self.individuals[p1], &self.individuals[p2])
                })
                .collect();

        assert!(offspring.len() == offspring_size);

        UnratedPopulation { individuals: offspring }
    }

    pub fn new() -> SelectedPopulation<I, F> {
        SelectedPopulation {
            individuals: Vec::new(),
            fitness: Vec::new(),
            rank_dist: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        // NOTE: the population and fitness arrays might still be larger.
        self.rank_dist.len()
    }

    /// Merging a selected population result in a rated population as the selection 
    /// criteria is no longer met.
    pub fn merge(self, offspring: RatedPopulation<I, F>) -> RatedPopulation<I, F> {
        let mut new_ind = Vec::with_capacity(self.len() + offspring.len());
        let mut new_fit = Vec::with_capacity(self.len() + offspring.len());
        self.all(&mut |ind, fit| {
            // XXX: Optimize. Avoid clone
            new_ind.push(ind.clone());
            new_fit.push(fit.clone());
        });

        new_ind.extend(offspring.individuals);
        new_fit.extend(offspring.fitness);

        assert!(new_ind.len() == new_fit.len());

        RatedPopulation {
            individuals: new_ind,
            fitness: new_fit,
        }
    }

    pub fn all<C>(&self, f: &mut C)
        where C: FnMut(&I, &F)
    {
        for rd in self.rank_dist.iter() {
            f(&self.individuals[rd.idx], &self.fitness[rd.idx]);
        }
    }

    // XXX: fitness_iter()
    pub fn fitness_to_vec(&self) -> Vec<F> {
        let mut v = Vec::new();
        self.all(&mut |_, f| v.push(f.clone()));
        v
    }

    pub fn all_of_rank<C>(&self, rank: usize, f: &mut C)
        where C: FnMut(&I, &F)
    {
        for rd in self.rank_dist.iter() {
            if rd.rank as usize == rank {
                f(&self.individuals[rd.idx], &self.fitness[rd.idx]);
            }
        }
    }
}

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
    fn mate<R: Rng>(&self, rng: &mut R, p1: &I, p2: &I) -> I;
    fn is_solution(&self, _ind: &I, _fit: &F) -> bool {
        false
    }

    fn run<R: Rng, L>(&self,
                      rng: &mut R,
                      config: &DriverConfig,
                      weight: f64,
                      logger: &L)
                      -> SelectedPopulation<I, F>
        where L: Fn(usize, u64, usize, &SelectedPopulation<I, F>)
    {
        // this is generation 0. it's empty
        let mut time_last = time::precise_time_ns();
        let mut parents = SelectedPopulation::<I, F>::new();
        let mut offspring = self.random_population(rng, config.mu);
        let mut gen: usize = 0;

        loop {
            let rated_offspring = offspring.rate_in_parallel(&|ind| self.fitness(ind), weight);
            let next_generation = parents.merge(rated_offspring);
            parents = next_generation.select(config.mu, config.num_objectives);

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

    let mut solutions: Vec<MultiObjective2<f32>> = Vec::new();
    solutions.push(MultiObjective2::from((1.0, 0.1)));
    solutions.push(MultiObjective2::from((0.1, 0.1)));
    solutions.push(MultiObjective2::from((0.1, 1.0)));
    solutions.push(MultiObjective2::from((0.5, 0.5)));
    solutions.push(MultiObjective2::from((0.5, 0.5)));

    println!("solutions: {:?}", solutions);
    let selection = select_solutions(&solutions[..], 5, 2);
    println!("selection: {:?}", selection);

    let fronts = fast_non_dominated_sort(&solutions[..], 10, &mut|a,b| a.dominate(b));
    println!("solutions: {:?}", solutions);
    println!("fronts: {:?}", fronts);

    for (rank, front) in fronts.iter().enumerate() {
        let distances = crowding_distance_assignment(&solutions[..], rank as u32, &front[..], 2);
        println!("front: {:?}", front);
        println!("distances: {:?}", distances);
    }
}
