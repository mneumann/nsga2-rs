/// This example shows how to optimize the zdt1 function using NSGA-II.
extern crate nsga2;
extern crate rand;

use rand::{Closed01, Rng};
use nsga2::objective::Objective;
use nsga2::multi_objective::MultiObjective;
use nsga2::tournament_selection::tournament_selection_fast;
use nsga2::selection::SelectAndRank;
use nsga2::select_nsga::{RankedSolution, SelectNSGA};
use std::cmp::{Ordering, PartialOrd};

/// optimal pareto front (f_1, 1 - sqrt(f_1))
/// 0 <= x[i] <= 1.0
fn zdt1(x: &[f32]) -> (f32, f32) {
    let n = x.len();
    debug_assert!(n >= 2);

    let f1 = x[0];
    let g = 1.0 + (9.0 / (n - 1) as f32) * x[1..].iter().fold(0.0, |b, &i| b + i);
    let f2 = g * (1.0 - (f1 / g).sqrt());

    (f1, f2)
}

fn _sbx_beta(u: f32, eta: f32) -> f32 {
    debug_assert!(u >= 0.0 && u < 1.0);

    if u <= 0.5 {
        2.0 * u
    } else {
        1.0 / (2.0 * (1.0 - u))
    }.powf(1.0 / (eta + 1.0))
}

fn sbx_beta_bounded(u: f32, eta: f32, gamma: f32) -> f32 {
    debug_assert!(u >= 0.0 && u < 1.0);

    let g = 1.0 - gamma;
    let ug = u * g;

    if u <= 0.5 / g {
        2.0 * ug
    } else {
        1.0 / (2.0 * (1.0 - ug))
    }.powf(1.0 / (eta + 1.0))
}

fn _sbx_single_var<R: Rng>(rng: &mut R, p: (f32, f32), eta: f32) -> (f32, f32) {
    let u = rng.gen::<f32>();
    let beta = _sbx_beta(u, eta);

    (
        0.5 * (((1.0 + beta) * p.0) + ((1.0 - beta) * p.1)),
        0.5 * (((1.0 - beta) * p.0) + ((1.0 + beta) * p.1)),
    )
}

fn _sbx_single_var_bounded<R: Rng>(
    rng: &mut R,
    p: (f32, f32),
    bounds: (f32, f32),
    eta: f32,
) -> (f32, f32) {
    let (a, b) = bounds;
    let p_diff = p.1 - p.0;

    debug_assert!(a <= b);
    debug_assert!(p_diff > 0.0);
    debug_assert!(p.0 >= a && p.0 <= b);
    debug_assert!(p.1 >= a && p.1 <= b);

    let beta_a = 1.0 + (p.0 - a) / p_diff;
    let beta_b = 1.0 + (b - p.1) / p_diff;

    fn gamma(beta: f32, eta: f32) -> f32 {
        1.0 / (2.0 * beta.powf(eta + 1.0))
    }

    let gamma_a = gamma(beta_a, eta);
    let gamma_b = gamma(beta_b, eta);

    let u = rng.gen::<f32>();
    let beta_ua = sbx_beta_bounded(u, eta, gamma_a);
    let beta_ub = sbx_beta_bounded(u, eta, gamma_b);

    let c = (
        0.5 * (((1.0 + beta_ua) * p.0) + ((1.0 - beta_ua) * p.1)),
        0.5 * (((1.0 - beta_ub) * p.0) + ((1.0 + beta_ub) * p.1)),
    );

    debug_assert!(c.0 >= a && c.0 <= b);
    debug_assert!(c.1 >= a && c.1 <= b);

    return c;
}

fn sbx_single_var_bounded<R: Rng>(
    rng: &mut R,
    p: (f32, f32),
    bounds: (f32, f32),
    eta: f32,
) -> (f32, f32) {
    if p.0 < p.1 {
        _sbx_single_var_bounded(rng, (p.0, p.1), bounds, eta)
    } else if p.0 > p.1 {
        let r = _sbx_single_var_bounded(rng, (p.1, p.0), bounds, eta);
        (r.1, r.0)
    } else {
        debug_assert!(p.0 == p.1);
        (p.0, p.1)
    }
}

// ------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ZdtGenome {
    xs: Vec<f32>,
}

type ZdtFitness = (f32, f32);

struct ZdtObjective1;
struct ZdtObjective2;

impl Objective for ZdtObjective1 {
    type Solution = ZdtFitness;
    type Distance = f64;

    fn total_order(&self, a: &Self::Solution, b: &Self::Solution) -> Ordering {
        a.0.partial_cmp(&b.0).unwrap()
    }

    fn distance(&self, a: &Self::Solution, b: &Self::Solution) -> Self::Distance {
        (a.0 - b.0) as Self::Distance
    }
}

impl Objective for ZdtObjective2 {
    type Solution = ZdtFitness;
    type Distance = f64;

    fn total_order(&self, a: &Self::Solution, b: &Self::Solution) -> Ordering {
        a.1.partial_cmp(&b.1).unwrap()
    }

    fn distance(&self, a: &Self::Solution, b: &Self::Solution) -> Self::Distance {
        (a.1 - b.1) as Self::Distance
    }
}

impl ZdtGenome {
    fn new(xs: Vec<f32>) -> Self {
        assert!(xs.len() >= 2);
        for &x in xs.iter() {
            assert!(x >= 0.0 && x <= 1.0);
        }
        ZdtGenome { xs: xs }
    }

    fn random<R: Rng>(rng: &mut R, n: usize) -> Self {
        ZdtGenome::new((0..n).map(|_| rng.gen::<Closed01<f32>>().0).collect())
    }

    fn fitness(&self) -> ZdtFitness {
        zdt1(&self.xs[..])
    }

    fn len(&self) -> usize {
        self.xs.len()
    }

    fn crossover1<R: Rng>(rng: &mut R, parents: (&Self, &Self), eta: f32) -> Self {
        assert!(parents.0.len() == parents.1.len());
        let xs: Vec<_> = parents
            .0
            .xs
            .iter()
            .zip(parents.1.xs.iter())
            .map(|(&x1, &x2)| {
                let (c1, _c2) = sbx_single_var_bounded(rng, (x1, x2), (0.0, 1.0), eta);
                c1
            })
            .collect();
        ZdtGenome::new(xs)
    }
}

// ------------------------------------------------------------------

struct ZdtDriver {
    zdt_order: usize,
    mating_eta: f32,
}

impl ZdtDriver {
    fn random_genome<R>(&self, rng: &mut R) -> ZdtGenome
    where
        R: Rng,
    {
        ZdtGenome::random(rng, self.zdt_order)
    }

    fn fitness(&self, individual: &ZdtGenome) -> ZdtFitness {
        individual.fitness()
    }

    fn mate<R>(&self, rng: &mut R, parent1: &ZdtGenome, parent2: &ZdtGenome) -> ZdtGenome
    where
        R: Rng,
    {
        ZdtGenome::crossover1(rng, (parent1, parent2), self.mating_eta)
    }
}

// ------------------------------------------------------------------

struct EvoConfig {
    /// size of population
    mu: usize,
    /// size of offspring population
    lambda: usize,
    /// tournament size
    k: usize,
    /// max number of generations
    ngen: usize,
}

fn generational_step<R: Rng>(
    rng: &mut R,
    driver: &ZdtDriver,
    evo_config: &EvoConfig,
    population: &[ZdtGenome],
    mo: &MultiObjective<ZdtFitness, f64>,
) -> Vec<ZdtGenome> {
    // rate population (calculate fitness)
    let rated_population: Vec<ZdtFitness> = population.iter().map(|i| driver.fitness(i)).collect();

    // assign rank and crowding distance, and reduce to `mu` individuals
    let ranked_population: Vec<RankedSolution<ZdtFitness>> =
        SelectNSGA.select_and_rank(&rated_population[..], evo_config.mu, &mo);

    // ------------------------------------------------------
    // generate offspring (reproduce)
    // ------------------------------------------------------

    // select two parents.
    let (parent1, parent2) = {
        let mut mate_partner_iter = (1..).map(|_| {
            tournament_selection_fast(
                rng,
                &ranked_population[..],
                |a, b| {
                    let order = a.rank.cmp(&b.rank).then_with(|| {
                        a.crowding_distance
                            .partial_cmp(&b.crowding_distance)
                            .unwrap()
                            .reverse()
                    });
                    match order {
                        Ordering::Less => true,
                        _ => false,
                    }
                },
                evo_config.k,
            )
        });

        let parent1 = mate_partner_iter.next().unwrap();
        let parent2 = mate_partner_iter.next().unwrap();
        (parent1, parent2)
    };

    // produce offspring population
    let mut offspring_population: Vec<_> = (0..evo_config.lambda)
        .map(|_| {
            // Create offspring
            driver.mate(rng, &population[parent1.index], &population[parent2.index])
        })
        .collect();

    // we now have a population with mu + lambda individuals
    offspring_population.extend_from_slice(population);

    return offspring_population;
}

fn main() {
    let mut rng = rand::thread_rng();

    let driver = ZdtDriver {
        zdt_order: 2,    // ZDT1 order
        mating_eta: 2.0, // cross-over variance
    };

    let evo_config = EvoConfig {
        mu: 100,     // size of population
        lambda: 100, // size of offspring population
        k: 2,        // tournament
        ngen: 2,     // max number of generations
    };

    // The objectives to use
    let mo = MultiObjective::new(&[&ZdtObjective1, &ZdtObjective2]);

    // generate initial population
    let mut population: Vec<_> = (0..evo_config.mu)
        .map(|_| driver.random_genome(&mut rng))
        .collect();

    for _gen in 0..evo_config.ngen {
        let next_generation = generational_step(&mut rng, &driver, &evo_config, &population, &mo);
        population = next_generation;
    }

    // -----------------------------------------
    // Final step
    // -----------------------------------------

    // rate population (calculate fitness)
    let rated_population: Vec<_> = population.iter().map(|i| driver.fitness(i)).collect();

    // assign rank and crowding distance, and reduce to `mu` individuals
    let ranked_population = SelectNSGA.select_and_rank(&rated_population[..], evo_config.mu, &mo);

    let max_rank = ranked_population
        .iter()
        .max_by_key(|i| i.rank)
        .unwrap()
        .rank;

    for rank in 0..max_rank + 1 {
        println!("# front {}", rank);

        let mut xys: Vec<_> = ranked_population
            .iter()
            .filter(|i| i.rank == rank)
            .map(|i| (i.solution.0, i.solution.1))
            .collect();

        xys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        println!("x\ty");
        for &(x, y) in xys.iter() {
            println!("{:.3}\t{:.3}", x, y);
        }
    }
}
