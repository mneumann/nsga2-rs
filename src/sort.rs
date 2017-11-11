use domination::Domination;
use std::cmp::Ordering;
use std::mem;

pub struct FastNonDominatedSorter {
    domination_count: Vec<usize>,
    dominated_solutions: Vec<Vec<usize>>,
    current_front: Vec<usize>,
}

impl FastNonDominatedSorter {
    /// Perform a non dominated sort of `solutions`.
    ///
    /// Each pareto front (the indices of the `solutions`) can be obtained by calling `next()`.

    pub fn new<T, FIT, MAP>(solutions: &[T], map: &MAP, objectives: &[usize]) -> Self
    where
        MAP: Fn(&T) -> &FIT,
        FIT: Domination,
    {
        let mut current_front = Vec::new();
        let mut domination_count: Vec<usize> = solutions.iter().map(|_| 0).collect();

        let mut dominated_solutions: Vec<Vec<usize>> =
            solutions.iter().map(|_| Vec::new()).collect();

        for i in 0..solutions.len() {
            for j in i + 1..solutions.len() {
                let p = &solutions[i];
                let q = &solutions[j];

                match map(p).domination_ord(map(q), objectives) {
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

pub fn fast_non_dominated_sort<T>(
    solutions: &[T],
    n: usize,
    objectives: &[usize],
) -> Vec<Vec<usize>>
where
    T: Domination,
{
    let sorter = FastNonDominatedSorter::new(solutions, &|f| f, objectives);
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
