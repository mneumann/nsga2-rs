use std::cmp::Ordering;
use std::ops::Sub;
use std::convert::From;
use domination::Domination;

pub trait MultiObjective: Send {
    const NUM_OBJECTIVES: usize;

    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering;

    /// Calculates the distance of objective between self and other
    fn dist_objective(&self, other: &Self, objective: usize) -> f64;

    fn cmp(&self, other: &Self) -> Ordering {
        for i in 0..Self::NUM_OBJECTIVES {
            match self.cmp_objective(other, i) {
                Ordering::Equal => {}
                ordering => return ordering
            }
        }
        Ordering::Equal
    }

    fn similar_to(&self, other: &Self, min_max_objective_distances: &[f64], objective_eps: f64) -> bool {
        debug_assert!(min_max_objective_distances.len() == Self::NUM_OBJECTIVES);
        let mut eq_cnt = 0;
        for i in 0..Self::NUM_OBJECTIVES {
            let dist = self.dist_objective(other, i).abs();
            let min_max_dist = min_max_objective_distances[i];
            if min_max_dist == 0.0 {
                if dist == 0.0 { // XXX: This could be removed, as it should be equal
                    eq_cnt += 1;
                }
            }
            else {
                assert!(min_max_dist > 0.0);
                debug_assert!(dist >= 0.0);
                if (dist / min_max_dist) < objective_eps {
                    eq_cnt += 1;
                }
            }
        }

        eq_cnt == Self::NUM_OBJECTIVES
    }
}

#[derive(Debug, Clone)]
pub struct MultiObjective2<T>
    where T: Sized + PartialOrd + Copy + Clone + Send
{
    pub objectives: [T; 2],
}

impl<T> From<(T, T)> for MultiObjective2<T>
    where T: Sized + PartialOrd + Copy + Clone + Send
{
    #[inline]
    fn from(t: (T, T)) -> MultiObjective2<T> {
        MultiObjective2 { objectives: [t.0, t.1] }
    }
}

impl<T, R> MultiObjective for MultiObjective2<T>
    where T: Copy + PartialOrd + Sub<Output = R> + Send,
          R: Into<f64>
{
    const NUM_OBJECTIVES: usize = 2;

    #[inline]
    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering {
        self.objectives[objective].partial_cmp(&other.objectives[objective]).unwrap()
    }
    #[inline]
    fn dist_objective(&self, other: &Self, objective: usize) -> f64 {
        (self.objectives[objective] - other.objectives[objective]).into()
    }
}

pub fn dominates_helper<T>(lhs: &T, rhs: &T) -> bool
    where T: MultiObjective
{
    let mut less_cnt = 0;
    for i in 0..T::NUM_OBJECTIVES {
        match lhs.cmp_objective(rhs, i) {
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

impl<T, R> Domination for MultiObjective2<T>
    where T: Copy + PartialOrd + Sub<Output = R> + Send,
          R: Into<f64>
{
    fn dominates(&self, other: &Self) -> bool {
        dominates_helper(self, other)
    }
}

#[test]
fn test_dominates() {
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
