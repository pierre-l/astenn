use crate::map::Bucket;

// ---------------------------------------------------------------------------
// Iter — immutable iterator over (&K, &V)
// ---------------------------------------------------------------------------

/// An iterator over the entries of a `HashMap`.
pub struct Iter<'a, K, V> {
    buckets: std::slice::Iter<'a, Bucket<K, V>>,
    current: std::slice::Iter<'a, (K, V)>,
}

impl<'a, K, V> Iter<'a, K, V> {
    pub(crate) fn new(buckets: &'a [Bucket<K, V>]) -> Self {
        Self {
            buckets: buckets.iter(),
            current: [].iter(),
        }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((k, v)) = self.current.next() {
                return Some((k, v));
            }
            let bucket = self.buckets.next()?;
            self.current = bucket.entries_slice().iter();
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_current = self.current.len();
        (remaining_current, None)
    }
}

// ---------------------------------------------------------------------------
// IterMut — mutable iterator over (&K, &mut V)
// ---------------------------------------------------------------------------

/// A mutable iterator over the entries of a `HashMap`.
pub struct IterMut<'a, K, V> {
    buckets: std::slice::IterMut<'a, Bucket<K, V>>,
    current: std::slice::IterMut<'a, (K, V)>,
}

impl<'a, K, V> IterMut<'a, K, V> {
    pub(crate) fn new(buckets: &'a mut [Bucket<K, V>]) -> Self {
        Self {
            buckets: buckets.iter_mut(),
            current: [].iter_mut(),
        }
    }
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((k, v)) = self.current.next() {
                // Reborrow to split into shared key ref and exclusive value ref.
                return Some((&*k, v));
            }
            let bucket = self.buckets.next()?;
            self.current = bucket.entries_slice_mut().iter_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// IntoIter — consuming iterator over (K, V)
// ---------------------------------------------------------------------------

/// An owning iterator over the entries of a `HashMap`. Constructed by
/// consuming the map; entries are collected eagerly from the inline
/// bucket storage.
pub struct IntoIter<K, V> {
    iter: std::vec::IntoIter<(K, V)>,
}

impl<K, V> IntoIter<K, V> {
    pub(crate) fn new(mut buckets: Vec<Bucket<K, V>>) -> Self {
        let total: usize = buckets.iter().map(|b| b.len()).sum();
        let mut entries = Vec::with_capacity(total);
        for bucket in &mut buckets {
            for i in 0..bucket.len() {
                // SAFETY: `i < bucket.len()`, so `entries[i]` is initialized.
                // `assume_init_read` moves the value out; we set `len = 0`
                // after the loop so the bucket's Drop won't double-free.
                entries.push(unsafe { bucket.entries[i].assume_init_read() });
            }
            bucket.len = 0;
        }
        Self {
            iter: entries.into_iter(),
        }
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

// ---------------------------------------------------------------------------
// Keys, Values, ValuesMut — thin wrappers around Iter / IterMut
// ---------------------------------------------------------------------------

/// An iterator over the keys of a `HashMap`.
pub struct Keys<'a, K, V>(pub(crate) Iter<'a, K, V>);

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, _)| k)
    }
}

/// An iterator over the values of a `HashMap`.
pub struct Values<'a, K, V>(pub(crate) Iter<'a, K, V>);

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, v)| v)
    }
}

/// A mutable iterator over the values of a `HashMap`.
pub struct ValuesMut<'a, K, V>(pub(crate) IterMut<'a, K, V>);

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, v)| v)
    }
}

// ---------------------------------------------------------------------------
// Drain — eagerly collects entries, leaves map empty
// ---------------------------------------------------------------------------

/// A draining iterator over the entries of a `HashMap`. The map is empty
/// after the `drain` call; remaining entries are dropped when `Drain` is
/// dropped.
pub struct Drain<'a, K, V> {
    iter: std::vec::IntoIter<(K, V)>,
    /// Hold the lifetime to prevent the map from being used while draining.
    _marker: std::marker::PhantomData<&'a mut (K, V)>,
}

impl<K, V> Drain<'_, K, V> {
    pub(crate) fn new(entries: Vec<(K, V)>) -> Self {
        Self {
            iter: entries.into_iter(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<K, V> Iterator for Drain<'_, K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
