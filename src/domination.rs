use std::cmp::Ordering;
use std::mem;

/// The dominance relation between two items.

pub trait Dominate {

    /// Returns true if `self` dominates `other`.

    fn dominates(&self, other: &Self) -> bool {
        match self.domination_ord(other) {
            Ordering::Less => true,
            _ => false,
        }
    }

    /// Returns the domination order.

    fn domination_ord(&self, other: &Self) -> Ordering {
        if self.dominates(other) {
            Ordering::Less
        } else if other.dominates(self) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

/// Determines the domination between two items.
///
/// To apply probabilistic domination, we need `&mut self` here.

pub trait Domination<T> {

    /// If `lhs` dominates `rhs`, returns `Less`.
    /// If `rhs` dominates `lhs`, returns `Greater`.
    /// If neither `lhs` nor `rhs` dominates the other, returns `Equal`.

    fn domination_ord(&mut self, lhs: &T, rhs: &T) -> Ordering;
}

/// Helpers struct that implements the Domination trait for any type `T : Dominate`.

pub struct DominationHelper;

impl<T: Dominate> Domination<T> for DominationHelper {
    fn domination_ord(&mut self, lhs: &T, rhs: &T) -> Ordering {
        lhs.domination_ord(rhs)
    }
}

pub struct FastNonDominatedSorter {
    domination_count: Vec<usize>,
    dominated_solutions: Vec<Vec<usize>>,
    current_front: Vec<usize>,
}

impl FastNonDominatedSorter {
    /// Perform a non dominated sort of `solutions`.
    ///
    /// Each pareto front (the indices of the `solutions`) can be obtained by calling `next()`.
    ///
    /// It is guaranteed that `domination` is exactly called once (and no more than once) for every
    /// unordered pair of solutions. That is, for two distinct solutions` i` and `j` either
    /// `domination(i, j)` or `domination(j, i)` is called, but never both. This is important for
    /// probabilistic dominance, where two calls could lead to different results.

    pub fn new<T, D>(solutions: &[T], domination: &mut D) -> Self
        where D: Domination<T>
    {
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

                match domination.domination_ord(p, q) {
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

        FastNonDominatedSorter {
            domination_count: domination_count,
            dominated_solutions: dominated_solutions,
            current_front: current_front,
        }
    }
}

/// Iterates over each pareto front.

impl Iterator for FastNonDominatedSorter {
    type Item = Vec<usize>;

    /// Returns the next front

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_front.is_empty() {
            return None;
        }

        // Calculate the next front

        let mut next_front = Vec::new();
        for &p_i in self.current_front.iter() {
            for &q_i in self.dominated_solutions[p_i].iter() {
                self.domination_count[q_i] -= 1;
                if self.domination_count[q_i] == 0 {
                    // q belongs to the next front
                    next_front.push(q_i);
                }
            }
        }

        // and return the current front, swapping it with the next

        let current_front = mem::replace(&mut self.current_front, next_front);

        return Some(current_front);
    }
}

pub fn fast_non_dominated_sort<T, D>(solutions: &[T],
                                     n: usize,
                                     domination: &mut D)
                                     -> Vec<Vec<usize>>
    where D: Domination<T>
{
    let sorter = FastNonDominatedSorter::new(solutions, domination);
    let mut found_solutions: usize = 0;
    let mut fronts = Vec::new();

    for front in sorter {
        found_solutions += front.len();
        fronts.push(front);
        if found_solutions >= n {
            break;
        }
    }
    return fronts;
}

#[cfg(test)]
mod tests {
    use super::{fast_non_dominated_sort, FastNonDominatedSorter, Domination, Dominate,
                DominationHelper};
    use std::cmp::Ordering;

    struct T(u32, u32);

    impl Dominate for T {
        fn domination_ord(&self, other: &Self) -> Ordering {
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
        assert_eq!(Ordering::Equal, T(1, 2).domination_ord(&T(1, 2)));
        assert_eq!(Ordering::Equal, T(1, 2).domination_ord(&T(2, 1)));
        assert_eq!(Ordering::Less, T(1, 2).domination_ord(&T(1, 3)));
        assert_eq!(Ordering::Less, T(0, 2).domination_ord(&T(1, 2)));
        assert_eq!(Ordering::Greater, T(1, 3).domination_ord(&T(1, 2)));
        assert_eq!(Ordering::Greater, T(1, 2).domination_ord(&T(0, 2)));
    }

    #[test]
    fn test_non_dominated_sort() {
        let solutions = vec![T(1, 2), T(1, 2), T(2, 1), T(1, 3), T(0, 2)];
        let fronts = fast_non_dominated_sort(&solutions, solutions.len(), &mut DominationHelper);

        assert_eq!(3, fronts.len());
        assert_eq!(&vec![2, 4], &fronts[0]);
        assert_eq!(&vec![0, 1], &fronts[1]);
        assert_eq!(&vec![3], &fronts[2]);
    }

    #[test]
    fn test_non_dominated_sort_iter() {
        let solutions = vec![T(1, 2), T(1, 2), T(2, 1), T(1, 3), T(0, 2)];
        let mut fronts = FastNonDominatedSorter::new(&solutions, &mut DominationHelper);

        assert_eq!(Some(vec![2, 4]), fronts.next());
        assert_eq!(Some(vec![0, 1]), fronts.next());
        assert_eq!(Some(vec![3]), fronts.next());
        assert_eq!(None, fronts.next());
    }

}
