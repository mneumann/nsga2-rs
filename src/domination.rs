use std::cmp::Ordering;

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
            _ => false
        }
    }
}

/// Perform a non dominated sort of the `solutions`.
///
/// Stop after we have found `n` solutions. As we include the whole pareto front, there are
/// probably more solutions returned.

pub fn fast_non_dominated_sort<P: Dominate>(solutions: &[P], n: usize) -> Vec<Vec<usize>> {
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front = Vec::new();

    let mut domination_count: Vec<usize> = (0..solutions.len()).map(|_| 0).collect();
    let mut dominated_solutions: Vec<Vec<usize>> = (0..solutions.len())
                                                       .map(|_| Vec::new())
                                                       .collect();
    let mut found_solutions: usize = 0;

    for (i, p) in solutions.iter().enumerate() {
        for (j, q) in solutions.iter().enumerate() {
            if i == j {
                continue;
            }
            match p.dominate(q) {
                Ordering::Less => {
                    // p dominates q
                    // Add `q` to the set of solutions dominated by `p`.
                    dominated_solutions[i].push(j);
                }
                Ordering::Greater => {
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

    while !current_front.is_empty() {
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
        assert_eq!(Ordering::Greater, T(1,2).dominate(&T(0, 2)));
    }

    #[test]
    fn test_non_dominated_sort() {
        let solutions = vec![T(1, 2), T(1, 2), T(2, 1), T(1, 3), T(0, 2)];
        let fronts = fast_non_dominated_sort(&solutions, solutions.len());

        assert_eq!(3, fronts.len());
        assert_eq!(&vec![2, 4], &fronts[0]);
        assert_eq!(&vec![0, 1], &fronts[1]);
        assert_eq!(&vec![3], &fronts[2]);
    }
}
