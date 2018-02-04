use domination::DominationOrd;
use std::cmp::Ordering;
use std::mem;

pub struct NonDominatedSorter {
    domination_count: Vec<usize>,
    dominated_solutions: Vec<Vec<usize>>,
    current_front: Vec<usize>,
}

struct Entry<'a, S, I>
where
    S: 'a,
{
    /// A reference to the solution
    solution: &'a S,

    /// The index that `solution` has within the `solutions` array.
    index: I,

    /// By how many other solutions is this solution dominated
    domination_count: I,

    /// Which solutions do we dominate
    dominated_solutions: Vec<I>,
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

        let mut arr: Vec<Entry<S, usize>> = solutions
            .iter()
            .enumerate()
            .map(|(index, solution)| Entry {
                solution,
                index,
                domination_count: 0,
                dominated_solutions: Vec::new(),
            })
            .collect();

        for mid in 1..arr.len() + 1 {
            let (front_slice, tail_slice) = arr.split_at_mut(mid);
            debug_assert!(front_slice.len() > 0);
            let p = front_slice.last_mut().unwrap();
            for q in tail_slice.iter_mut() {
                match domination.domination_ord(p.solution, q.solution) {
                    Ordering::Less => {
                        // p dominates q
                        // Add `q` to the set of solutions dominated by `p`.
                        p.dominated_solutions.push(q.index);
                        // q is dominated by p
                        q.domination_count += 1;
                    }
                    Ordering::Greater => {
                        // p is dominated by q
                        // Add `p` to the set of solutions dominated by `q`.
                        q.dominated_solutions.push(p.index);
                        // q dominates p
                        // Increment domination counter of `p`.
                        p.domination_count += 1;
                    }
                    Ordering::Equal => {}
                }
            }
            if p.domination_count == 0 {
                // `p` belongs to the first front as it is not dominated by any
                // other solution.
                current_front.push(p.index);
            }
        }

        // XXX: this is inefficient
        NonDominatedSorter {
            domination_count: arr.iter().map(|e| e.domination_count).collect(),
            dominated_solutions: arr.iter().map(|e| e.dominated_solutions.clone()).collect(),
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
mod helper {
    use test_helper_domination::Tuple;
    pub fn get_solutions() -> Vec<Tuple> {
        vec![
            Tuple(1, 2),
            Tuple(1, 2),
            Tuple(2, 1),
            Tuple(1, 3),
            Tuple(0, 2),
        ]
    }
}

#[test]
fn test_non_dominated_sort() {
    use test_helper_domination::TupleDominationOrd;
    let solutions = helper::get_solutions();
    let fronts = non_dominated_sort(&solutions, &TupleDominationOrd, solutions.len());

    assert_eq!(3, fronts.len());
    assert_eq!(&vec![2, 4], &fronts[0]);
    assert_eq!(&vec![0, 1], &fronts[1]);
    assert_eq!(&vec![3], &fronts[2]);
}

#[test]
fn test_non_dominated_sort_iter() {
    use test_helper_domination::TupleDominationOrd;
    let solutions = helper::get_solutions();
    let mut fronts = NonDominatedSorter::new(&solutions, &TupleDominationOrd);

    assert_eq!(Some(vec![2, 4]), fronts.next());
    assert_eq!(Some(vec![0, 1]), fronts.next());
    assert_eq!(Some(vec![3]), fronts.next());
    assert_eq!(None, fronts.next());
}
