/// Optimizes the zdt1 function using NSGA-II

extern crate rand;
extern crate nsga2;

use rand::{Rng, Closed01};
use nsga2::driver::{Driver, DriverConfig};
use nsga2::multi_objective::MultiObjective2;
use nsga2::selection::SelectNSGA;

/// optimal pareto front (f_1, 1 - sqrt(f_1))
/// 0 <= x[i] <= 1.0
fn zdt1(x: &[f32]) -> (f32, f32) {
    let n = x.len();
    assert!(n >= 2);

    let f1 = x[0];
    let g = 1.0 + (9.0 / (n - 1) as f32) * x[1..].iter().fold(0.0, |b, &i| b + i);
    let f2 = g * (1.0 - (f1 / g).sqrt());

    (f1, f2)
}

#[inline]
fn sbx_beta(u: f32, eta: f32) -> f32 {
    assert!(u >= 0.0 && u < 1.0);

    if u <= 0.5 {
        2.0 * u
    } else {
        1.0 / (2.0 * (1.0 - u))
    }
    .powf(1.0 / (eta + 1.0))
}

#[inline]
fn sbx_beta_bounded(u: f32, eta: f32, gamma: f32) -> f32 {
    assert!(u >= 0.0 && u < 1.0);

    let g = 1.0 - gamma;
    let ug = u * g;

    if u <= 0.5 / g {
        2.0 * ug
    } else {
        1.0 / (2.0 * (1.0 - ug))
    }
    .powf(1.0 / (eta + 1.0))
}

#[inline]
pub fn sbx_single_var<R: Rng>(rng: &mut R, p: (f32, f32), eta: f32) -> (f32, f32) {
    let u = rng.gen::<f32>();
    let beta = sbx_beta(u, eta);

    (0.5 * (((1.0 + beta) * p.0) + ((1.0 - beta) * p.1)),
     0.5 * (((1.0 - beta) * p.0) + ((1.0 + beta) * p.1)))
}

#[inline]
fn _sbx_single_var_bounded<R: Rng>(rng: &mut R,
                                   p: (f32, f32),
                                   bounds: (f32, f32),
                                   eta: f32)
                                   -> (f32, f32) {
    let (a, b) = bounds;
    let p_diff = p.1 - p.0;

    assert!(a <= b);
    assert!(p_diff > 0.0);
    assert!(p.0 >= a && p.0 <= b);
    assert!(p.1 >= a && p.1 <= b);

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

    let c = (0.5 * (((1.0 + beta_ua) * p.0) + ((1.0 - beta_ua) * p.1)),
             0.5 * (((1.0 - beta_ub) * p.0) + ((1.0 + beta_ub) * p.1)));

    assert!(c.0 >= a && c.0 <= b);
    assert!(c.1 >= a && c.1 <= b);

    return c;
}

#[inline]
pub fn sbx_single_var_bounded<R: Rng>(rng: &mut R,
                                      p: (f32, f32),
                                      bounds: (f32, f32),
                                      eta: f32)
                                      -> (f32, f32) {
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

#[derive(Clone, Debug)]
struct ZdtGenome {
    xs: Vec<f32>,
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

    fn fitness(&self) -> MultiObjective2<f32> {
        MultiObjective2::from(zdt1(&self.xs[..]))
    }

    fn len(&self) -> usize {
        self.xs.len()
    }

    fn crossover1<R: Rng>(rng: &mut R, parents: (&Self, &Self), eta: f32) -> Self {
        assert!(parents.0.len() == parents.1.len());
        let xs: Vec<_> = parents.0
                                .xs
                                .iter()
                                .zip(parents.1.xs.iter())
                                .map(|(&x1, &x2)| {
                                    let (c1, _c2) = sbx_single_var_bounded(rng,
                                                                           (x1, x2),
                                                                           (0.0, 1.0),
                                                                           eta);
                                    c1
                                })
                                .collect();
        ZdtGenome::new(xs)
    }
}

struct ZdtDriver {
    zdt_order: usize,
    mating_eta: f32,
}

impl Driver for ZdtDriver {
    type GENOME = ZdtGenome;
    type FIT = MultiObjective2<f32>;
    type SELECTION = SelectNSGA;

    fn random_genome<R>(&self, rng: &mut R) -> Self::GENOME where R: Rng{
        ZdtGenome::random(rng, self.zdt_order)
    }

    fn fitness(&self, ind: &Self::GENOME) -> Self::FIT {
        ind.fitness()
    }

    fn mate<R>(&self, rng: &mut R, parent1: &Self::GENOME, parent2: &Self::GENOME) -> Self::GENOME where R: Rng{
        ZdtGenome::crossover1(rng, (parent1, parent2), self.mating_eta)
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    let driver = ZdtDriver {
        zdt_order: 2, // ZDT1 order
        mating_eta: 2.0, // cross-over variance
    };

    let driver_config = DriverConfig {
        mu: 100, // size of population
        lambda: 100, // size of offspring population
        k: 2, // tournament
        ngen: 2, // max number of generations
        num_objectives: 2, // number of objectives
        parallel_weight: 1.0, // rayon's weight
    };

    let final_population = driver.run(&mut rng, &driver_config, &SelectNSGA, &|_, _, _, _| {});

    let max_rank = final_population.max_rank().unwrap();
    for rank in 0..max_rank + 1 {
        println!("# front {}", rank);

        println!("x\ty");
        let mut xys = Vec::new();
        final_population.all_of_rank(rank,
                                     &mut |_, fitness| {
                                         xys.push((fitness.objectives[0], fitness.objectives[1]));
                                     });

        xys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for &(x, y) in xys.iter() {
            println!("{:.3}\t{:.3}", x, y);
        }
    }

    // println!("x\ty\tfront\tcrowding");
    // final_population.all_with_rank_dist(&mut|_, fitness, rank, dist| {
    //    println!("{:.3}\t{:.3}\t{}\t{:.4}", fitness.objectives[0], fitness.objectives[1], rank, dist);
    // });
}
