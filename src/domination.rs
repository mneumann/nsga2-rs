use std::cmp::Ordering;

pub trait Domination<T> {
    fn domination(&mut self, lhs: &T, rhs: &T) -> Ordering;
}

/// The dominance relation.

pub trait Dominate {

    /// If `self` dominates `other`, return `Less`.
    /// If `other` dominates `self`, return `Greater`.
    /// If neither `self` nor `other` dominates the other, `Equal`.

    fn dominate(&self, other: &Self) -> Ordering {
        if self.dominates(other) {
            Ordering::Less
        } else if other.dominates(self) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    /// Returns true if `self` dominates `other`.

    fn dominates(&self, other: &Self) -> bool {
        match self.dominate(other) {
            Ordering::Less => true,
            _ => false,
        }
    }
}

type ArrayIndex = u32;

/// Perform a non dominated sort of `solutions`.
///
/// Stop after we have found `n` solutions, but include the whole pareto front the `n`-th solution
/// is in. That is, probably more than `n` solutions are returned.
///
/// Returned are the indices of solutions of each front.
///
/// It is guaranteed that `domination` is exactly called once (and no more than once) for every
/// unordered pair of solutions. That is, for two distinct solutions` i` and `j` either
/// `domination(i, j)` or `domination(j, i)` is called, but never both. This is important for
/// probabilistic dominance, where two calls could lead to different results.

pub fn fast_non_dominated_sort<T, DOM>(solutions: &[T],
                                       n: usize,
                                       domination: &mut DOM)
                                       -> Vec<Vec<usize>> // ArrayIndex
    where DOM: FnMut(&T, &T) -> Ordering
{
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front = Vec::new();

    let mut domination_count: Vec<usize> = solutions.iter()
                                                    .map(|_| 0)
                                                    .collect();

    let mut dominated_solutions: Vec<Vec<usize>> = solutions.iter()
                                                            .map(|_| Vec::new())
                                                            .collect();

    for i in 0..solutions.len() {
        for j in i + 1..solutions.len() {
            let p = &solutions[i];
            let q = &solutions[j];

            match domination(p, q) {
                Ordering::Less => {
                    // p dominates q
                    // Add `q` to the set of solutions dominated by `p`.
                    dominated_solutions[i].push(j);
                    // q is dominated by p
                    domination_count[j] += 1;
                }
                Ordering::Greater => {
                    // p is dominated by q
                    // Add `p` to the set of solutions dominated by `q`.
                    dominated_solutions[j].push(i);
                    // q dominates p
                    // Increment domination counter of `p`.
                    domination_count[i] += 1;
                }
                _ => {}
            }
        }

        if domination_count[i] == 0 {
            // `p` belongs to the first front as it is not dominated by any
            // other solution.
            current_front.push(i);
        }
    }

    let mut found_solutions: usize = 0;

    while current_front.len() > 0 {
        found_solutions += current_front.len();
        if found_solutions >= n {
            fronts.push(current_front);
            break;
        } else {
            // we haven't found enough solutions yet.
            let mut next_front = Vec::new();
            for &p_i in current_front.iter() {
                for &q_i in dominated_solutions[p_i].iter() {
                    domination_count[q_i] -= 1;
                    if domination_count[q_i] == 0 {
                        // q belongs to the next front
                        next_front.push(q_i);
                    }
                }
            }
            fronts.push(current_front);
            current_front = next_front;
        }
    }

    return fronts;
}

#[cfg(test)]
mod tests {
    use super::{fast_non_dominated_sort, Dominate};
    use std::cmp::Ordering;

    struct T(u32, u32);

    impl Dominate for T {
        fn dominate(&self, other: &Self) -> Ordering {
            if self.0 < other.0 && self.1 <= other.1 {
                return Ordering::Less;
            }
            if self.0 <= other.0 && self.1 < other.1 {
                return Ordering::Less;
            }

            if self.0 > other.0 && self.1 >= other.1 {
                return Ordering::Greater;
            }
            if self.0 >= other.0 && self.1 > other.1 {
                return Ordering::Greater;
            }

            return Ordering::Equal;
        }
    }

    #[test]
    fn test_dominate() {
        assert_eq!(Ordering::Equal, T(1, 2).dominate(&T(1, 2)));
        assert_eq!(Ordering::Equal, T(1, 2).dominate(&T(2, 1)));
        assert_eq!(Ordering::Less, T(1, 2).dominate(&T(1, 3)));
        assert_eq!(Ordering::Less, T(0, 2).dominate(&T(1, 2)));
        assert_eq!(Ordering::Greater, T(1, 3).dominate(&T(1, 2)));
        assert_eq!(Ordering::Greater, T(1, 2).dominate(&T(0, 2)));
    }

    #[test]
    fn test_non_dominated_sort() {
        let solutions = vec![T(1, 2), T(1, 2), T(2, 1), T(1, 3), T(0, 2)];
        let fronts = fast_non_dominated_sort(&solutions,
                                             solutions.len(),
                                             &mut |a, b| a.dominate(b));

        assert_eq!(3, fronts.len());
        assert_eq!(&vec![2, 4], &fronts[0]);
        assert_eq!(&vec![0, 1], &fronts[1]);
        assert_eq!(&vec![3], &fronts[2]);
    }
}
