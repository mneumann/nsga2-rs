use std::cmp::Ordering;
use std::marker::PhantomData;
use objective::Objective;
use non_dominated_sort::DominationOrd;

pub struct MultiObjective<'a, S, D>
where
    S: 'a,
    D: 'a,
{
    pub objectives: &'a [&'a Objective<Solution = S, Distance = D>],
    _solution: PhantomData<S>,
    _distance: PhantomData<D>,
}

impl<'a, S, D> MultiObjective<'a, S, D>
where
    S: 'a,
    D: 'a,
{
    pub fn new(objectives: &'a [&'a Objective<Solution = S, Distance = D>]) -> Self {
        Self {
            objectives,
            _solution: PhantomData,
            _distance: PhantomData,
        }
    }
}

impl<'a, S, D> DominationOrd for MultiObjective<'a, S, D>
where
    S: 'a,
    D: 'a,
{
    type Solution = S;

    fn domination_ord(&self, a: &Self::Solution, b: &Self::Solution) -> Ordering {
        let mut less_cnt = 0;
        let mut greater_cnt = 0;

        for objective in self.objectives.iter() {
            match objective.total_order(a, b) {
                Ordering::Less => {
                    less_cnt += 1;
                }
                Ordering::Greater => {
                    greater_cnt += 1;
                }
                Ordering::Equal => {}
            }
        }

        if less_cnt > 0 && greater_cnt == 0 {
            Ordering::Less
        } else if greater_cnt > 0 && less_cnt == 0 {
            Ordering::Greater
        } else {
            debug_assert!((less_cnt > 0 && greater_cnt > 0) || (less_cnt == 0 && greater_cnt == 0));
            Ordering::Equal
        }
    }
}

#[test]
fn test_multi_objective() {
    use test_helper_objective::{Objective1, Objective2, Objective3, Tuple};

    // construct a multi objective over a Tuple
    let mo = MultiObjective::<Tuple, f64>::new(&[&Objective1, &Objective2, &Objective3]);

    let a = Tuple(1, 2);
    let b = Tuple(2, 1);
    let c = Tuple(1, 3);

    assert_eq!(Ordering::Equal, mo.domination_ord(&a, &a));
    assert_eq!(Ordering::Equal, mo.domination_ord(&a, &b));
    assert_eq!(Ordering::Equal, mo.domination_ord(&b, &b));

    assert_eq!(Ordering::Less, mo.domination_ord(&a, &c));
    assert_eq!(Ordering::Greater, mo.domination_ord(&c, &a));
}
