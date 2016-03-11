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

