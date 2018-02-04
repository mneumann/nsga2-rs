use std::cmp::Ordering;

/// An *objective* defines a total ordering relation and a distance
/// metric on `solutions` in order to be able to compare any two
/// solutions according to the following two questions:
///
/// - "which solution is better" (total order)
///
/// - "how much better" (distance metric)
///
/// Objectives can be seen as a projection of a (possibly) multi-variate
/// solution value to a scalar value. There can be any number of
/// different projections (objectives) for any given solution value. Of
/// course solution values need not be multi-variate.
///
/// We use the term "solution" here, ignoring the fact that in pratice
/// we often have to evaluate the "fitness" of a solution prior of being
/// able to define any useful ordering relation or distance metric. As
/// the fitness generally is a function of the solution, this is more or
/// less an implementation detail or that of an optimization. Nothing
/// prevents you from using the fitness value here as the solution
/// value.

pub trait Objective {
    /// The solution value type that we define the objective on.
    type Solution;

    /// The output type of the distance metric.
    type Distance: Sized;

    /// An objective defines a total order between two solution values.
    fn total_order(&self, a: &Self::Solution, b: &Self::Solution) -> Ordering;

    /// An objective defines a distance metric between two solution
    /// values. Distance values can be negative, i.e. the caller is
    /// responsible for obtaining absolute values.
    fn distance(&self, a: &Self::Solution, b: &Self::Solution) -> Self::Distance;
}

#[test]
fn test_objectives() {
    use test_helper_objective::{Objective1, Objective2, Objective3, Tuple};

    let a = &Tuple(1, 2);
    let b = &Tuple(2, 1);
    assert_eq!(Ordering::Less, Objective1.total_order(a, b));
    assert_eq!(Ordering::Greater, Objective2.total_order(a, b));
    assert_eq!(Ordering::Equal, Objective3.total_order(a, b));

    assert_eq!(-1.0, Objective1.distance(a, b));
    assert_eq!(1.0, Objective2.distance(a, b));
    assert_eq!(0.0, Objective3.distance(a, b));
}
