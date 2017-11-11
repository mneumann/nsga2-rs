use domination::Domination;
use multi_objective::MultiObjective;
use crowding_distance::CrowdingDistanceAssignment; // XXX: Change name
use selection::SelectSolutions;
use rayon::par_iter::*;
use selection::tournament_selection_fast;
use rand::Rng;
use std::u32;

pub struct Individual<G, F>
where
    F: MultiObjective + Domination,
    G: Send,
{
    genome: G,

    // The individuals fitness.
    fitness: Option<F>,

    // which pareto front this rated individual belongs to (only valid if fitness.is_some() and
    // pareto fronts have been determined)
    pareto_rank: u32,

    // The crowding distance to neighboring individuals (individuals with similar fitness)
    crowding_distance: f64,

    // the number of individuals in each crowd (points of equal fitness)
    crowd_size: u32,

    selected: bool,
}

impl<G, F> CrowdingDistanceAssignment<F> for Individual<G, F>
where
    F: MultiObjective + Domination,
    G: Send,
{
    fn fitness(&self) -> &F {
        self.fitness.as_ref().unwrap()
    }
    fn rank_mut(&mut self) -> &mut u32 {
        &mut self.pareto_rank
    }
    fn dist_mut(&mut self) -> &mut f64 {
        &mut self.crowding_distance
    }
    fn rank(&self) -> u32 {
        self.pareto_rank
    }
    fn dist(&self) -> f64 {
        self.crowding_distance
    }

    fn select(&mut self) {
        assert!(self.selected == false);
        self.selected = true;
    }
    fn unselect(&mut self) {
        self.selected = false;
    }
    fn is_selected(&self) -> bool {
        self.selected
    }

    fn crowd(&self) -> usize {
        self.crowd_size as usize
    }

    fn set_crowd(&mut self, crowd: usize) {
        self.crowd_size = crowd as u32;
    }
}

impl<G, F> Individual<G, F>
where
    F: MultiObjective + Domination,
    G: Send,
{
    fn from_genome(genome: G) -> Self {
        Individual {
            genome: genome,
            fitness: None,
            pareto_rank: u32::MAX,
            crowding_distance: 0.0,
            crowd_size: 1,
            selected: false,
        }
    }

    pub fn genome(&self) -> &G {
        &self.genome
    }

    pub fn into_genome(self) -> G {
        self.genome
    }

    pub fn fitness(&self) -> &F {
        self.fitness.as_ref().unwrap()
    }
    pub fn fitness_mut(&mut self) -> &mut F {
        self.fitness.as_mut().unwrap()
    }
}

/// An unrated Population of individuals.

pub struct UnratedPopulation<G, F>
where
    F: MultiObjective + Domination,
    G: Send,
{
    individuals: Vec<Individual<G, F>>,
}

/// A rated Population of individuals, i.e. each individual has a fitness assigned.

pub struct RatedPopulation<G, F>
where
    F: MultiObjective + Domination,
    G: Send,
{
    individuals: Vec<Individual<G, F>>,
}

/// A ranked (selected) Population (pareto_rank and crowding_distance).

pub struct RankedPopulation<G, F>
where
    F: MultiObjective + Domination,
    G: Send,
{
    individuals: Vec<Individual<G, F>>,
}

impl<G, F> UnratedPopulation<G, F>
where
    F: MultiObjective + Domination,
    G: Send,
{
    pub fn individuals(&self) -> &[Individual<G, F>] {
        &self.individuals
    }

    pub fn as_vec(self) -> Vec<Individual<G, F>> {
        self.individuals
    }

    pub fn merge(self, other: Self) -> Self {
        let mut ind = self.individuals;
        ind.extend(other.individuals.into_iter());
        UnratedPopulation { individuals: ind }
    }

    pub fn new() -> Self {
        UnratedPopulation { individuals: Vec::new() }
    }

    pub fn push(&mut self, genome: G) {
        self.individuals.push(Individual::from_genome(genome));
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Rate the individuals in parallel.

    pub fn rate_in_parallel<E>(self, eval: &E, weight: f64) -> RatedPopulation<G, F>
    where
        E: Fn(&G) -> F + Sync,
    {
        let UnratedPopulation { mut individuals } = self;

        individuals.par_iter_mut().weight(weight).for_each(|ind| {
            assert!(ind.fitness.is_none());
            let fitness = eval(&ind.genome);
            ind.fitness = Some(fitness);
        });

        RatedPopulation { individuals: individuals }
    }
}

impl<G, F> RatedPopulation<G, F>
where
    F: MultiObjective + Domination,
    G: Send,
{
    pub fn select<S, R>(
        self,
        population_size: usize,
        objectives: &[usize],
        selection: &S,
        rng: &mut R,
    ) -> RankedPopulation<G, F>
    where
        S: SelectSolutions<Individual<G, F>, F>,
        R: Rng,
    {
        let RatedPopulation { mut individuals } = self;

        // evaluate rank and crowding distance: XXX: replace by Select trait
        selection.select_solutions(&mut individuals, population_size, objectives, rng);

        // only keep individuals which are selected
        individuals.retain(|i| i.is_selected());

        // sort by rank and crowding-distance
        individuals.sort_by(|a, b| a.rank_and_crowding_order(b));

        assert!(individuals.len() == population_size);

        RankedPopulation { individuals: individuals }
    }

    pub fn individuals(&self) -> &[Individual<G, F>] {
        &self.individuals
    }

    pub fn individuals_mut(&mut self) -> &mut [Individual<G, F>] {
        &mut self.individuals
    }

    pub fn new() -> Self {
        RatedPopulation { individuals: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    pub fn fitness(&self, index: usize) -> &F {
        self.individuals[index].fitness.as_ref().unwrap()
    }

    pub fn fitness_mut(&mut self, index: usize) -> &mut F {
        self.individuals[index].fitness.as_mut().unwrap()
    }
}

impl<G, F> RankedPopulation<G, F>
where
    F: MultiObjective + Domination,
    G: Send,
{
    pub fn into_unrated(self) -> UnratedPopulation<G, F> {
        let RankedPopulation { individuals } = self;
        let mut pop = UnratedPopulation::new();
        for ind in individuals {
            pop.push(ind.genome);
        }
        pop
    }

    /// Generate an unrated offspring population.
    pub fn reproduce<R, M>(
        &self,
        rng: &mut R,
        offspring_size: usize,
        tournament_k: usize,
        mate: &M,
    ) -> UnratedPopulation<G, F>
    where
        R: Rng,
        M: Fn(&mut R, &G, &G) -> G,
    {
        assert!(tournament_k > 0);
        assert!(self.len() > 0);

        // create `offspring_size` new offspring using k-tournament (
        // select the best individual out of k randomly choosen individuals)
        let offspring: Vec<_> =
            (0..offspring_size)
                .map(|_| {

                    // first parent. k candidates
                    let p1 = tournament_selection_fast(rng,
                                                       |i1, i2|self.individuals[i1].has_better_rank_and_crowding(&self.individuals[i2]),
                                                       self.len(),
                                                       tournament_k);

                    // second parent. k candidates
                    let p2 = tournament_selection_fast(rng,
                                                       |i1, i2| self.individuals[i1].has_better_rank_and_crowding(&self.individuals[i2]),
                                                       self.len(),
                                                       tournament_k);

                    // cross-over the two parents and produce one child (throw away
                    // second child XXX)

                    // The potentially dominating individual is gives as first
                    // parameter.
                    //let (p1, p2) = if self.individuals[p1].has_better_rank_and_crowding(&self.individuals[p2]) {
                    //    (p1, p2)
                    //} else if self.individuals[p2].has_better_rank_and_crowding(&self.individuals[p1]) {
                    //    (p2, p1)
                    //} else {
                    //    (p1, p2)
                    //};

                    Individual::from_genome(mate(rng, &self.individuals[p1].genome, &self.individuals[p2].genome))
                })
                .collect();

        assert!(offspring.len() == offspring_size);

        UnratedPopulation { individuals: offspring }
    }

    pub fn new() -> Self {
        RankedPopulation { individuals: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Merging a ranked population with a rated population results in a rated population as the selection
    /// criteria is no longer met.
    pub fn merge(self, offspring: RatedPopulation<G, F>) -> RatedPopulation<G, F> {
        let RankedPopulation { mut individuals } = self;

        // XXX: Reset rank and crowding distance.

        individuals.extend(offspring.individuals);

        RatedPopulation { individuals: individuals }
    }

    pub fn all<C>(&self, f: &mut C)
    where
        C: FnMut(&G, &F),
    {
        for ind in &self.individuals {
            f(&ind.genome, ind.fitness.as_ref().unwrap());
        }
    }

    pub fn max_rank(&self) -> Option<usize> {
        self.individuals
            .iter()
            .map(|ind| ind.pareto_rank)
            .max()
            .map(|r| r as usize)
    }

    pub fn all_of_rank<C>(&self, rank: usize, f: &mut C)
    where
        C: FnMut(&G, &F),
    {
        for ind in &self.individuals {
            if ind.pareto_rank as usize == rank {
                f(&ind.genome, ind.fitness.as_ref().unwrap());
            }
        }
    }

    pub fn individuals(&self) -> &[Individual<G, F>] {
        &self.individuals
    }
}
