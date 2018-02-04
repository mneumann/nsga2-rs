use objective::Objective;
use std::marker::PhantomData;

pub struct MultiObjective<'a, S, D>
where
    S: 'a,
    D: 'a,
{
    objectives: &'a [&'a Objective<Solution = S, Distance = D>],
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

#[test]
fn test_multi_objective() {
    use test_helper_objective::{Objective1, Objective2, Objective3, Tuple};

    // construct a multi objective over a Tuple
    let _ = MultiObjective::<Tuple, f64>::new(&[&Objective1, &Objective2, &Objective3]);
}
