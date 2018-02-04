use rand::Rng;

/// Select the best individual out of `k` randomly choosen.  This gives
/// individuals with better fitness a higher chance to be selected.
///
/// NOTE: We are not using `sample_iter(rng, values, k)` as it is *very*
/// expensive.  Instead we call `rng.choose()` k-times. The drawn items
/// could be the same for each call, but the probability is very low if
/// the number of `values` is high compared to `k`.

#[inline]
pub fn tournament_selection_fast<'a, T, R: Rng, F>(
    rng: &mut R,
    values: &'a [T],
    better_than: F,
    k: usize,
) -> &'a T
where
    F: Fn(&'a T, &'a T) -> bool,
{
    assert!(values.len() > 0);

    let mut best = rng.choose(values).unwrap();

    for _ in 1..k {
        let next = rng.choose(values).unwrap();
        if better_than(next, best) {
            best = next;
        }
    }

    return best;
}
