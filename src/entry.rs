use crate::map::HashMapInner;

/// A view into a single entry in a map, which may be either vacant or
/// occupied. This is constructed via [`HashMap::entry`].
pub enum Entry<'a, K, V> {
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V>),
    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V>),
}

/// A view into an occupied entry in a `HashMap`. It is part of the
/// [`Entry`] enum.
pub struct OccupiedEntry<'a, K, V> {
    inner: &'a mut HashMapInner<K, V>,
    bucket_idx: usize,
    entry_idx: usize,
}

/// A view into a vacant entry in a `HashMap`. It is part of the
/// [`Entry`] enum.
pub struct VacantEntry<'a, K, V> {
    inner: &'a mut HashMapInner<K, V>,
    key: K,
    hash: u64,
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

impl<'a, K: Eq, V> Entry<'a, K, V> {
    /// Ensures a value is in the entry by inserting the default if empty, and
    /// returns a mutable reference to the value in the entry.
    #[inline]
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default
    /// function if empty, and returns a mutable reference to the value.
    #[inline]
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default()),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default
    /// function (which receives the entry's key) if empty, and returns a
    /// mutable reference to the value.
    #[inline]
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let val = default(&e.key);
                e.insert(val)
            }
        }
    }

    /// Returns a reference to this entry's key.
    #[inline]
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(e) => e.key(),
            Entry::Vacant(e) => e.key(),
        }
    }

    /// Provides in-place mutable access to an occupied entry before any
    /// potential inserts into the map.
    #[inline]
    pub fn and_modify<F: FnOnce(&mut V)>(mut self, f: F) -> Self {
        if let Entry::Occupied(ref mut e) = self {
            f(e.get_mut());
        }
        self
    }
}

impl<'a, K: Eq, V: Default> Entry<'a, K, V> {
    /// Ensures a value is in the entry by inserting the default value if
    /// empty, and returns a mutable reference to the value.
    #[inline]
    pub fn or_default(self) -> &'a mut V {
        self.or_insert_with(V::default)
    }
}

// ---------------------------------------------------------------------------
// OccupiedEntry
// ---------------------------------------------------------------------------

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    pub(crate) fn new(
        inner: &'a mut HashMapInner<K, V>,
        bucket_idx: usize,
        entry_idx: usize,
    ) -> Self {
        Self {
            inner,
            bucket_idx,
            entry_idx,
        }
    }

    /// Gets a reference to the key in the entry.
    #[inline]
    pub fn key(&self) -> &K {
        &self.inner.buckets[self.bucket_idx].entries[self.entry_idx].1
    }

    /// Gets a reference to the value in the entry.
    #[inline]
    pub fn get(&self) -> &V {
        &self.inner.buckets[self.bucket_idx].entries[self.entry_idx].2
    }

    /// Gets a mutable reference to the value in the entry.
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.inner.buckets[self.bucket_idx].entries[self.entry_idx].2
    }

    /// Converts the `OccupiedEntry` into a mutable reference to the value in
    /// the entry with a lifetime bound to the map itself.
    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        &mut self.inner.buckets[self.bucket_idx].entries[self.entry_idx].2
    }

    /// Sets the value of the entry and returns the old value.
    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.get_mut(), value)
    }

    /// Takes ownership of the key and value from the map.
    #[inline]
    pub fn remove_entry(self) -> (K, V) {
        let (_, k, v) = self.inner.buckets[self.bucket_idx]
            .entries
            .swap_remove(self.entry_idx);
        self.inner.len -= 1;
        (k, v)
    }

    /// Takes the value out of the entry, and returns it.
    #[inline]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }
}

// ---------------------------------------------------------------------------
// VacantEntry
// ---------------------------------------------------------------------------

impl<'a, K: Eq, V> VacantEntry<'a, K, V> {
    pub(crate) fn new(inner: &'a mut HashMapInner<K, V>, key: K, hash: u64) -> Self {
        Self { inner, key, hash }
    }

    /// Gets a reference to the key that would be used when inserting a value
    /// through the `VacantEntry`.
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Take ownership of the key.
    #[inline]
    pub fn into_key(self) -> K {
        self.key
    }

    /// Sets the value of the entry with the `VacantEntry`'s key, and returns
    /// a mutable reference to it.
    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        let (bi, ei) = self.inner.insert_entry(self.hash, self.key, value);
        &mut self.inner.buckets[bi].entries[ei].2
    }
}
