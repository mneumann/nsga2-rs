use std::cmp::Ordering;

/// The dominance relation between two items.

pub trait Domination {

    /// Returns true if `self` dominates `other`, taking only `objectives` into account.

    fn dominates(&self, other: &Self, objectives: &[usize]) -> bool {
        match self.domination_ord(other, objectives) {
            Ordering::Less => true,
            _ => false,
        }
    }

    /// Returns the domination order, taking only `objectives` into account.
    fn domination_ord(&self, other: &Self, objectives: &[usize]) -> Ordering {
        if self.dominates(other, objectives) {
            Ordering::Less
        } else if other.dominates(self, objectives) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

#[cfg(test)]
mod tests {
    use sort::{fast_non_dominated_sort, FastNonDominatedSorter};
    use super::Domination;
    use std::cmp::Ordering;

    struct T(u32, u32);

    impl Domination for T {
        fn domination_ord(&self, other: &Self, _objectives: &[usize]) -> Ordering {
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
        assert_eq!(Ordering::Equal, T(1, 2).domination_ord(&T(1, 2), &[0,1]));
        assert_eq!(Ordering::Equal, T(1, 2).domination_ord(&T(2, 1), &[0,1]));
        assert_eq!(Ordering::Less, T(1, 2).domination_ord(&T(1, 3), &[0,1]));
        assert_eq!(Ordering::Less, T(0, 2).domination_ord(&T(1, 2), &[0,1]));
        assert_eq!(Ordering::Greater, T(1, 3).domination_ord(&T(1, 2), &[0,1]));
        assert_eq!(Ordering::Greater, T(1, 2).domination_ord(&T(0, 2), &[0,1]));
    }

    #[test]
    fn test_non_dominated_sort() {
        let solutions = vec![T(1, 2), T(1, 2), T(2, 1), T(1, 3), T(0, 2)];
        let fronts = fast_non_dominated_sort(&solutions, solutions.len(), &[0,1]);

        assert_eq!(3, fronts.len());
        assert_eq!(&vec![2, 4], &fronts[0]);
        assert_eq!(&vec![0, 1], &fronts[1]);
        assert_eq!(&vec![3], &fronts[2]);
    }

    #[test]
    fn test_non_dominated_sort_iter() {
        let solutions = vec![T(1, 2), T(1, 2), T(2, 1), T(1, 3), T(0, 2)];
        let mut fronts = FastNonDominatedSorter::new(&solutions, &|s| s, &[0,1]);

        assert_eq!(Some(vec![2, 4]), fronts.next());
        assert_eq!(Some(vec![0, 1]), fronts.next());
        assert_eq!(Some(vec![3]), fronts.next());
        assert_eq!(None, fronts.next());
    }

}
