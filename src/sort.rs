use domination::DominationOrd;
use std::cmp::Ordering;
use std::mem;

pub struct NonDominatedSorter {
    domination_count: Vec<usize>,
    dominated_solutions: Vec<Vec<usize>>,
    current_front: Vec<usize>,
}

impl NonDominatedSorter {
    /// Perform a non-dominated sort of `solutions`.
    ///
    /// Each pareto front (the indices of the `solutions`) can be obtained by calling `next()`.

    pub fn new<S, D>(solutions: &[S], domination: &D) -> Self
    where
        D: DominationOrd<Solution = S>,
    {
        let mut current_front = Vec::new();
        let mut domination_count: Vec<usize> = solutions.iter().map(|_| 0).collect();

        // XXX: create an array with Item { t: &T, dominated_solutions: Vec<usize>,
        // domination_count: usize }

        let mut dominated_solutions: Vec<Vec<usize>> =
            solutions.iter().map(|_| Vec::new()).collect();

        // XXX: Use mut_split on the array, to obtain two mutable slices.
        for i in 0..solutions.len() {
            for j in i + 1..solutions.len() {
                let p = &solutions[i];
                let q = &solutions[j];

                /// XXX: We do not want to wrap our solutions within a DominationOrd
                /// struct. So it's easier to pass in the relevant DominationOrd struct
                /// and call it's domination_ord function passing the solution in.
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
                    Ordering::Equal => {}
                }
            }

            if domination_count[i] == 0 {
                // `p` belongs to the first front as it is not dominated by any
                // other solution.
                current_front.push(i);
            }
        }

        NonDominatedSorter {
            domination_count: domination_count,
            dominated_solutions: dominated_solutions,
            current_front: current_front,
        }
    }
}

/// Iterates over each pareto front.
impl Iterator for NonDominatedSorter {
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
                debug_assert!(self.domination_count[q_i] > 0);
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

pub fn non_dominated_sort<S, D>(
    solutions: &[S],
    domination: &D,
    cap_fronts_at: usize,
) -> Vec<Vec<usize>>
where
    D: DominationOrd<Solution = S>,
{
    let sorter = NonDominatedSorter::new(solutions, domination);
    let mut found_solutions: usize = 0;
    let mut fronts = Vec::new();

    for front in sorter {
        found_solutions += front.len();
        fronts.push(front);
        if found_solutions >= cap_fronts_at {
            break;
        }
    }
    return fronts;
}

#[cfg(test)]
mod tests {
    use super::{non_dominated_sort, NonDominatedSorter};
    use test_helper_domination::{Tuple, TupleDominationOrd};

    fn get_solutions() -> Vec<Tuple> {
        vec![
            Tuple(1, 2),
            Tuple(1, 2),
            Tuple(2, 1),
            Tuple(1, 3),
            Tuple(0, 2),
        ]
    }

    #[test]
    fn test_non_dominated_sort() {
        let solutions = get_solutions();
        let fronts = non_dominated_sort(&solutions, &TupleDominationOrd, solutions.len());

        assert_eq!(3, fronts.len());
        assert_eq!(&vec![2, 4], &fronts[0]);
        assert_eq!(&vec![0, 1], &fronts[1]);
        assert_eq!(&vec![3], &fronts[2]);
    }

    #[test]
    fn test_non_dominated_sort_iter() {
        let solutions = get_solutions();
        let mut fronts = NonDominatedSorter::new(&solutions, &TupleDominationOrd);

        assert_eq!(Some(vec![2, 4]), fronts.next());
        assert_eq!(Some(vec![0, 1]), fronts.next());
        assert_eq!(Some(vec![3]), fronts.next());
        assert_eq!(None, fronts.next());
    }
}
