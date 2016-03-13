use domination::Domination;
use multi_objective::MultiObjective;
use crowding_distance::SolutionRankDist;
use selection::select_solutions;
use std::iter::FromIterator;
use rayon::par_iter::*;
use selection::tournament_selection_fast;
use rand::Rng;

/// An unrated Population of individuals.
pub struct UnratedPopulation<I>
    where I: Clone + Sync
{
    individuals: Vec<I>,
}

/// A rated Population of individuals, i.e.
/// each individual has a fitness assigned.
pub struct RatedPopulation<I, F>
    where I: Clone + Sync,
          F: MultiObjective + Clone + Sync
{
    individuals: Vec<I>,
    fitness: Vec<F>,
}

/// A selected (ranked) Population.
pub struct SelectedPopulation<I, F>
    where I: Clone + Sync,
          F: MultiObjective + Clone + Sync
{
    individuals: Vec<I>,
    fitness: Vec<F>,
    rank_dist: Vec<SolutionRankDist>,
}

impl<I> From<Vec<I>> for UnratedPopulation<I> where I: Clone + Sync
{
    fn from(v: Vec<I>) -> UnratedPopulation<I> {
        UnratedPopulation { individuals: v }
    }
}

impl<I> FromIterator<I> for UnratedPopulation<I> where I: Clone + Sync
{
    fn from_iter<T>(iterator: T) -> UnratedPopulation<I>
        where T: IntoIterator<Item = I>
    {
        UnratedPopulation { individuals: iterator.into_iter().collect() }
    }
}

impl<I> AsRef<[I]> for UnratedPopulation<I> where I: Clone + Sync
{
    fn as_ref(&self) -> &[I] {
        &self.individuals
    }
}

impl<I> UnratedPopulation<I> where I: Clone + Sync
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

    /// Rate the individuals in parallel.

    pub fn rate_in_parallel<F, E>(self, eval: &E, weight: f64) -> RatedPopulation<I, F>
        where F: MultiObjective + Clone + Send + Sync,
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
    where I: Clone + Sync,
          F: MultiObjective + Clone + Sync
{
    /// Maps the fitness `F` to a fitness `G`. The evaluator function
    /// gets called with the old `RatedPopulation<I,F>` as first argument,
    /// so the evaluation can be based upon the old population (e.g. this is
    /// used by HyperNSGA to determine the behavioral diversity which is
    /// a population measure).

    pub fn map_fitness<E, G>(self, eval: &E, weight: f64) -> RatedPopulation<I, G>
        where E: Sync + Fn(&RatedPopulation<I, F>, usize) -> G,
              G: MultiObjective + Clone + Send + Sync
    {
        let mut new_fitness = Vec::new();
        (0..self.individuals.len())
            .into_par_iter()
            .map(|i| eval(&self, i))
            .weight(weight)
            .collect_into(&mut new_fitness);
        assert!(new_fitness.len() == self.individuals.len());

        RatedPopulation {
            individuals: self.individuals,
            fitness: new_fitness,
        }
    }

    pub fn select<D>(self,
                     population_size: usize,
                     num_objectives: usize,
                     domination: &mut D)
                     -> SelectedPopulation<I, F>
        where D: Domination<F>
    {
        // evaluate rank and crowding distance
        let rank_dist = select_solutions(&self.fitness,
                                         population_size,
                                         num_objectives,
                                         domination);
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
    where I: Clone + Sync,
          F: MultiObjective + Clone + Sync
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

    pub fn all_with_rank_dist<C>(&self, f: &mut C)
        where C: FnMut(&I, &F, usize, f32)
    {
        for rd in self.rank_dist.iter() {
            f(&self.individuals[rd.idx],
              &self.fitness[rd.idx],
              rd.rank as usize,
              rd.dist);
        }
    }


    // XXX: fitness_iter()
    pub fn fitness_to_vec(&self) -> Vec<F> {
        let mut v = Vec::new();
        self.all(&mut |_, f| v.push(f.clone()));
        v
    }

    pub fn max_rank(&self) -> Option<usize> {
        self.rank_dist.iter().map(|rd| rd.rank).max().map(|r| r as usize)
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
