use std::cmp::Ordering;

/// The dominance relation between two solutions. There are three possible situations:
///
/// - Either solution `a` dominates solution `b` ("a < b")
/// - Or solution `b` dominates solution `a` ("a > b")
/// - Or neither solution `a` nor `b` dominates each other ("a == b")
///
pub trait DominationOrd {
    /// The solution value type on which the dominance relation is defined.
    type Solution;

    /// Returns the domination order.
    fn domination_ord(&self, a: &Self::Solution, b: &Self::Solution) -> Ordering {
        if self.dominates(a, b) {
            Ordering::Less
        } else if self.dominates(b, a) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    /// Returns true if `a` dominates `b` ("a < b").
    fn dominates(&self, a: &Self::Solution, b: &Self::Solution) -> bool {
        match self.domination_ord(a, b) {
            Ordering::Less => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DominationOrd;
    use std::cmp::Ordering;

    // Our multi-variate fitness/solution value
    struct Tuple(usize, usize);

    // We can have multiple dominance relations defined on a single
    // type, without having to wrap the "Tuple" itself.
    struct TupleDominationOrd;

    impl DominationOrd for TupleDominationOrd {
        type Solution = Tuple;

        fn domination_ord(&self, a: &Self::Solution, b: &Self::Solution) -> Ordering {
            if a.0 < b.0 && a.1 <= b.1 {
                Ordering::Less
            } else if a.0 <= b.0 && a.1 < b.1 {
                Ordering::Less
            } else if a.0 > b.0 && a.1 >= b.1 {
                Ordering::Greater
            } else if a.0 >= b.0 && a.1 > b.1 {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
    }

    #[test]
    fn test_non_domination() {
        let a = &Tuple(1, 2);
        let b = &Tuple(2, 1);

        // Non-domination due to reflexitivity
        assert_eq!(Ordering::Equal, TupleDominationOrd.domination_ord(a, a));
        assert_eq!(Ordering::Equal, TupleDominationOrd.domination_ord(b, b));

        // Non-domination
        assert_eq!(Ordering::Equal, TupleDominationOrd.domination_ord(a, b));
        assert_eq!(Ordering::Equal, TupleDominationOrd.domination_ord(b, a));
    }

    #[test]
    fn test_domination() {
        let a = &Tuple(1, 2);
        let b = &Tuple(1, 3);
        let c = &Tuple(0, 2);

        // a < b
        assert_eq!(Ordering::Less, TupleDominationOrd.domination_ord(a, b));
        // c < a
        assert_eq!(Ordering::Less, TupleDominationOrd.domination_ord(c, a));
        // transitivity => c < b
        assert_eq!(Ordering::Less, TupleDominationOrd.domination_ord(c, b));

        // Just reverse the relation: forall a, b: a < b => b > a

        // b > a
        assert_eq!(Ordering::Greater, TupleDominationOrd.domination_ord(b, a));
        // a > c
        assert_eq!(Ordering::Greater, TupleDominationOrd.domination_ord(a, c));
        // transitivity => b > c
        assert_eq!(Ordering::Greater, TupleDominationOrd.domination_ord(b, c));
    }
}
