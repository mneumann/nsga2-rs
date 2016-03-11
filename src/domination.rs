use std::cmp::Ordering;

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


#[cfg(test)]
mod tests {
    use sort::{fast_non_dominated_sort, FastNonDominatedSorter};
    use super::{Domination, Dominate, DominationHelper};
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
