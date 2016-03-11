use prob::Prob;
use rand::Rng;
use multi_objective::MultiObjective;
use domination::Domination;
use std::cmp::{self, Ordering};

/// Probabilistic domination helper (PNSGA)
///
/// Each objective has a corresponding probability which is used to determine whether
/// this objective is taken into account for the dominance relation.

pub struct ProbabilisticDominationHelper<'a, R>
    where R: Rng + 'a
{
    objective_probabilities: &'a [Prob],
    rng: &'a mut R,
}

impl<'a, R> ProbabilisticDominationHelper<'a, R> where R: Rng + 'a
{
    pub fn new(objective_probabilities: &'a [Prob], rng: &'a mut R) -> Self {
        ProbabilisticDominationHelper {
            objective_probabilities: objective_probabilities,
            rng: rng,
        }
    }
}

impl<'a, R, T> Domination<T> for ProbabilisticDominationHelper<'a, R>
    where R: Rng,
          T: MultiObjective
{
    fn domination_ord(&mut self, lhs: &T, rhs: &T) -> Ordering {
        debug_assert!(lhs.num_objectives() == rhs.num_objectives());
        debug_assert!(lhs.num_objectives() == self.objective_probabilities.len());
        let nobjs = cmp::min(cmp::min(lhs.num_objectives(), rhs.num_objectives()),
                             self.objective_probabilities.len());

        let mut left_dom_cnt = 0;
        let mut right_dom_cnt = 0;

        for i in 0..nobjs {
            let prob = self.objective_probabilities[i];
            if !prob.flip(self.rng) {
                continue;
            }

            match lhs.cmp_objective(rhs, i) {
                Ordering::Less => {
                    left_dom_cnt += 1;
                }
                Ordering::Greater => {
                    right_dom_cnt += 1;
                }
                Ordering::Equal => {}
            }
        }

        if left_dom_cnt > 0 && right_dom_cnt == 0 {
            Ordering::Less
        } else if right_dom_cnt > 0 && left_dom_cnt == 0 {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}


#[test]
fn test_probabilistic_domination() {
    use prob::Prob;
    use multi_objective::MultiObjective2 as MO2;
    use domination::Dominate;
    use rand;

    let m12: MO2<f32> = MO2::from((1.0, 2.0));
    let m21: MO2<f32> = MO2::from((2.0, 1.0));
    let m13: MO2<f32> = MO2::from((1.0, 3.0));
    let m02: MO2<f32> = MO2::from((0.0, 2.0));

    assert_eq!(Ordering::Equal, m12.domination_ord(&m12));
    assert_eq!(Ordering::Equal, m12.domination_ord(&m21));
    assert_eq!(Ordering::Less, m12.domination_ord(&m13));
    assert_eq!(Ordering::Less, m02.domination_ord(&m12));
    assert_eq!(Ordering::Greater, m13.domination_ord(&m12));
    assert_eq!(Ordering::Greater, m12.domination_ord(&m02));


    {
        let mut rng = rand::thread_rng();
        let objective_probs = [Prob::new(1.0), Prob::new(1.0)];
        let mut prob_domination = ProbabilisticDominationHelper::new(&objective_probs, &mut rng);

        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m12, &m12));
        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m12, &m21));
        assert_eq!(Ordering::Less, prob_domination.domination_ord(&m12, &m13));
        assert_eq!(Ordering::Less, prob_domination.domination_ord(&m02, &m12));
        assert_eq!(Ordering::Greater,
                   prob_domination.domination_ord(&m13, &m12));
        assert_eq!(Ordering::Greater,
                   prob_domination.domination_ord(&m12, &m02));
    }

    {
        let mut rng = rand::thread_rng();
        let objective_probs = [Prob::new(1.0), Prob::new(0.0)];
        let mut prob_domination = ProbabilisticDominationHelper::new(&objective_probs, &mut rng);

        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m12, &m12));
        assert_eq!(Ordering::Less, prob_domination.domination_ord(&m12, &m21));
        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m12, &m13));
        assert_eq!(Ordering::Less, prob_domination.domination_ord(&m02, &m12));
        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m13, &m12));
        assert_eq!(Ordering::Greater,
                   prob_domination.domination_ord(&m12, &m02));
    }

    {
        let mut rng = rand::thread_rng();
        let objective_probs = [Prob::new(0.0), Prob::new(1.0)];
        let mut prob_domination = ProbabilisticDominationHelper::new(&objective_probs, &mut rng);

        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m12, &m12));
        assert_eq!(Ordering::Greater,
                   prob_domination.domination_ord(&m12, &m21));
        assert_eq!(Ordering::Less, prob_domination.domination_ord(&m12, &m13));
        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m02, &m12));
        assert_eq!(Ordering::Greater,
                   prob_domination.domination_ord(&m13, &m12));
        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m12, &m02));
    }

    {
        let mut rng = rand::thread_rng();
        let objective_probs = [Prob::new(0.0), Prob::new(0.0)];
        let mut prob_domination = ProbabilisticDominationHelper::new(&objective_probs, &mut rng);

        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m12, &m12));
        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m12, &m21));
        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m12, &m13));
        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m02, &m12));
        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m13, &m12));
        assert_eq!(Ordering::Equal, prob_domination.domination_ord(&m12, &m02));
    }



}
