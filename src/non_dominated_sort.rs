use domination::DominationOrd;
use std::cmp::Ordering;
use std::mem;
use std::collections::VecDeque;

struct Entry<'a, S, I = usize>
where
    S: 'a,
{
    /// A reference to the solution
    solution: &'a S,

    /// The index that `solution` has within the `solutions` array.
    index: I,

    /// By how many other solutions is this solution dominated
    domination_count: I,

    /// Which solutions we dominate
    dominated_solutions: VecDeque<I>,
}

pub struct NonDominatedSorter<'a, S, I = usize>
where
    S: 'a,
{
    entries: Vec<Entry<'a, S, I>>,
    current_front: Vec<(I, &'a S)>,
}

impl<'a, S> NonDominatedSorter<'a, S> {
    /// Perform a non-dominated sort of `solutions`.
    ///
    /// Each pareto front (the indices of the `solutions`) can be obtained by calling `next()`.

    pub fn new<D>(solutions: &'a [S], domination: &D) -> Self
    where
        D: DominationOrd<Solution = S>,
    {
        let mut current_front = Vec::new();

        let mut entries: Vec<_> = solutions
            .iter()
            .enumerate()
            .map(|(index, solution)| Entry {
                solution,
                index,
                domination_count: 0,
                dominated_solutions: VecDeque::new(),
            })
            .collect();

        for mid in 1..entries.len() + 1 {
            let (front_slice, tail_slice) = entries.split_at_mut(mid);
            debug_assert!(front_slice.len() > 0);
            let p = front_slice.last_mut().unwrap();
            for q in tail_slice.iter_mut() {
                match domination.domination_ord(p.solution, q.solution) {
                    Ordering::Less => {
                        // p dominates q
                        // Add `q` to the set of solutions dominated by `p`.
                        p.dominated_solutions.push_back(q.index);
                        // q is dominated by p
                        q.domination_count += 1;
                    }
                    Ordering::Greater => {
                        // p is dominated by q
                        // Add `p` to the set of solutions dominated by `q`.
                        q.dominated_solutions.push_back(p.index);
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
                current_front.push((p.index, p.solution));
            }
        }

        Self {
            entries,
            current_front,
        }
    }

    // Returns an iterator that yields all pareto fronts
    // pub fn pareto_front_iter()
}

/// Iterate over the pareto fronts. Each call to next() will yield the
/// next pareto front.
impl<'a, S> Iterator for NonDominatedSorter<'a, S> {
    type Item = Vec<(usize, &'a S)>;

    /// Return the next pareto front

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_front.is_empty() {
            return None;
        }

        // Calculate the next front

        let mut next_front = Vec::new();
        for &(p_i, _) in self.current_front.iter() {
            // to calculate the next front, we have to remove the
            // solutions of the current front, and as such, decrease the
            // domination_count of they dominated_solutions. We can
            // destruct the dominated_solutions array here, as we will
            // no longer need it.
            // The only problem with poping off solutions off the end is
            // that we will populate the fronts in reverse order. For
            // that reason, we are using a VecDeque. This should give us
            // a stable sort.

            while let Some(q_i) = self.entries[p_i].dominated_solutions.pop_front() {
                let q = &mut self.entries[q_i];
                debug_assert!(q.domination_count > 0);
                q.domination_count -= 1;
                if q.domination_count == 0 {
                    // q is not dominated by any other solution. it belongs to the next front.
                    next_front.push((q_i, q.solution));
                }
            }
        }

        // and return the current front, swapping it with the next

        let current_front = mem::replace(&mut self.current_front, next_front);

        return Some(current_front);
    }
}

pub fn non_dominated_sort<'a, S, D>(
    solutions: &'a [S],
    domination: &D,
    cap_fronts_at: usize,
) -> Vec<Vec<(usize, &'a S)>>
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
    pub fn keep_only_index(front: &[(usize, &Tuple)]) -> Vec<usize> {
        front.iter().map(|&(i, _)| i).collect()
    }

    pub fn assert_front_eq(expected: &[usize], front: &[(usize, &Tuple)]) {
        assert_eq!(expected.to_owned(), keep_only_index(front));
    }
}

#[test]
fn test_non_dominated_sort() {
    use test_helper_domination::TupleDominationOrd;
    let solutions = helper::get_solutions();
    let fronts = non_dominated_sort(&solutions, &TupleDominationOrd, solutions.len());

    assert_eq!(3, fronts.len());
    helper::assert_front_eq(&[2, 4], &fronts[0]);
    helper::assert_front_eq(&[0, 1], &fronts[1]);
    helper::assert_front_eq(&[3], &fronts[2]);
}

#[test]
fn test_non_dominated_sort_iter() {
    use test_helper_domination::TupleDominationOrd;
    let solutions = helper::get_solutions();
    let mut fronts = NonDominatedSorter::new(&solutions, &TupleDominationOrd);
    let f = &|front: Vec<_>| helper::keep_only_index(&front);

    assert_eq!(Some(vec![2, 4]), fronts.next().map(f));
    assert_eq!(Some(vec![0, 1]), fronts.next().map(f));
    assert_eq!(Some(vec![3]), fronts.next().map(f));
    assert_eq!(None, fronts.next().map(f));
}
