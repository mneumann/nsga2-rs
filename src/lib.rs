extern crate rand;

use std::cmp::{self, Ordering};
use std::f32;
use rand::Rng;
use selection::tournament_selection_fast;
use mo::MultiObjective;
pub use self::domination::Dominate;
use domination::fast_non_dominated_sort;

pub mod selection;
pub mod mo;
pub mod domination;

impl<T: MultiObjective> Dominate<T> for T {
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
                distance[indices[i]] += solutions[next_idx]
                    .dist_objective(&solutions[prev_idx], m) /
                    norm;
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

    let pareto_fronts = fast_non_dominated_sort(solutions, n);

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

impl<I> UnratedPopulation<I> where I: Clone
{
    pub fn new() -> UnratedPopulation<I> {
        UnratedPopulation { individuals: Vec::new() }
    }

    pub fn push(&mut self, ind: I) {
        self.individuals.push(ind);
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    // Rate the individuals
    pub fn rate<F, R>(self, fitness: &mut R) -> RatedPopulation<I, F>
        where F: Dominate + MultiObjective + Clone,
              R: FnMut(&I) -> F
              {
                  let fitness = self.individuals.iter().map(|ind| fitness(ind)).collect();
                  RatedPopulation {
                      individuals: self.individuals,
                      fitness: fitness
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
            rank_dist: rank_dist
        }
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }
}

impl<I, F> SelectedPopulation<I, F>
where I: Clone,
      F: Dominate + MultiObjective + Clone
{
    /// Generate an offspring population.
    pub fn reproduce<R, M>(&self,
                           rng: &mut R,
                           offspring_size: usize,
                           tournament_k: usize,
                           mate: &mut M)
        -> UnratedPopulation<I>
        where R: Rng,
              M: FnMut(&mut R, &I, &I) -> I
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

    pub fn len(&self) -> usize {
        // NOTE: the population and fitness arrays might still be larger.
        self.rank_dist.len()
    }

    /// Merging a selected population result in a rated population as the selection 
    /// criteria is no longer met.
    pub fn merge(self, offspring: RatedPopulation<I, F>) -> RatedPopulation<I, F> {
        let mut new_ind = Vec::with_capacity(self.len() + offspring.len());
        let mut new_fit = Vec::with_capacity(self.len() + offspring.len());
        for rd in self.rank_dist.iter() {
            // XXX: Optimize. Avoid clone
            new_ind.push(self.individuals[rd.idx].clone());
            new_fit.push(self.fitness[rd.idx].clone());
        }

        new_ind.extend(offspring.individuals);
        new_fit.extend(offspring.fitness);

        assert!(new_ind.len() == new_fit.len());

        RatedPopulation {
            individuals: new_ind,
            fitness: new_fit
        }
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
    let selection = select(&solutions[..], 5, 2);
    println!("selection: {:?}", selection);

    let fronts = fast_non_dominated_sort(&solutions[..], 10);
    println!("solutions: {:?}", solutions);
    println!("fronts: {:?}", fronts);

    for (rank, front) in fronts.iter().enumerate() {
        let distances = crowding_distance_assignment(&solutions[..], rank as u32, &front[..], 2);
        println!("front: {:?}", front);
        println!("distances: {:?}", distances);
    }
}
