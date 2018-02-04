use std::cmp::Ordering;
use objective::Objective;

// Our multi-variate fitness/solution value
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct Tuple(pub usize, pub usize);

// We define three objectives
pub struct Objective1;
pub struct Objective2;
pub struct Objective3;

impl Objective for Objective1 {
    type Solution = Tuple;
    type Distance = f64;

    fn total_order(&self, a: &Self::Solution, b: &Self::Solution) -> Ordering {
        a.0.cmp(&b.0)
    }

    fn distance(&self, a: &Self::Solution, b: &Self::Solution) -> Self::Distance {
        (a.0 as f64) - (b.0 as f64)
    }
}

impl Objective for Objective2 {
    type Solution = Tuple;
    type Distance = f64;

    fn total_order(&self, a: &Self::Solution, b: &Self::Solution) -> Ordering {
        a.1.cmp(&b.1)
    }

    fn distance(&self, a: &Self::Solution, b: &Self::Solution) -> Self::Distance {
        (a.1 as f64) - (b.1 as f64)
    }
}

// Objective3 is defined on the sum of the tuple values.
impl Objective for Objective3 {
    type Solution = Tuple;
    type Distance = f64;

    fn total_order(&self, a: &Self::Solution, b: &Self::Solution) -> Ordering {
        (a.0 + a.1).cmp(&(b.0 + b.1))
    }

    fn distance(&self, a: &Self::Solution, b: &Self::Solution) -> Self::Distance {
        (a.0 + a.1) as f64 - (b.0 + b.1) as f64
    }
}
