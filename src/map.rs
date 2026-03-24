use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::ops::Index;

use crate::entry::{Entry, OccupiedEntry, VacantEntry};
use crate::iter::{Drain, IntoIter, Iter, IterMut, Keys, Values, ValuesMut};

/// Entries per bucket before a split is attempted. 8 entries keeps buckets
/// compact (~320 bytes for typical Melin key/value sizes) while amortising
/// the cost of linear scan within a bucket.
const BUCKET_CAPACITY: usize = 8;

/// Maximum local depth — bounded by the number of usable bits in a u64 hash.
/// Once a bucket reaches this depth, further splits are impossible (all entries
/// share the same hash prefix) so the bucket is allowed to overflow.
const MAX_LOCAL_DEPTH: u8 = 64;

// ---------------------------------------------------------------------------
// Bucket
// ---------------------------------------------------------------------------

/// A single bucket in the extendible hash table. Multiple directory slots may
/// point to the same bucket (when `local_depth < global_depth`). Each entry
/// stores the full 64-bit hash alongside key and value to avoid rehashing
/// during bucket splits.
pub(crate) struct Bucket<K, V> {
    pub(crate) local_depth: u8,
    /// Entries are `(hash, key, value)`. We store the hash to enable:
    /// 1. Fast-reject during lookup (compare u64 before calling `Eq`)
    /// 2. Zero-cost redistribution during bucket splits
    pub(crate) entries: Vec<(u64, K, V)>,
}

impl<K: Clone, V: Clone> Clone for Bucket<K, V> {
    fn clone(&self) -> Self {
        Self {
            local_depth: self.local_depth,
            entries: self.entries.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// HashMapInner — hasher-agnostic core (allows Entry API without S parameter)
// ---------------------------------------------------------------------------

/// The guts of the hash map, deliberately separated from the hasher so that
/// [`Entry`], [`OccupiedEntry`], and [`VacantEntry`] do not need to carry the
/// `S` type parameter (matching the `std::collections::HashMap` entry API).
pub(crate) struct HashMapInner<K, V> {
    /// Directory of bucket indices. Length is always `2^global_depth`.
    /// Multiple slots may map to the same bucket index when a bucket's
    /// `local_depth < global_depth`.
    pub(crate) directory: Vec<usize>,

    /// Pool of buckets. Each bucket appears exactly once; the directory
    /// references buckets by their index in this vec. Buckets are never
    /// removed (no merge on delete) — this keeps the pool dense and avoids
    /// index invalidation.
    pub(crate) buckets: Vec<Bucket<K, V>>,

    pub(crate) global_depth: u8,
    pub(crate) len: usize,
}

impl<K, V> HashMapInner<K, V> {
    /// Map a hash to its directory slot.
    #[inline]
    fn dir_index(&self, hash: u64) -> usize {
        // Use the low `global_depth` bits. When global_depth == 0 the
        // directory has a single entry and the mask is 0, which is correct.
        (hash as usize) & ((1usize << self.global_depth) - 1)
    }
}

impl<K: Eq, V> HashMapInner<K, V> {
    /// Find `(bucket_pool_idx, entry_idx)` for a key, or `None`.
    #[inline]
    fn find<Q>(&self, hash: u64, key: &Q) -> Option<(usize, usize)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let dir_idx = self.dir_index(hash);
        let bucket_idx = self.directory[dir_idx];
        let bucket = &self.buckets[bucket_idx];
        for (i, (h, k, _)) in bucket.entries.iter().enumerate() {
            if *h == hash && k.borrow() == key {
                return Some((bucket_idx, i));
            }
        }
        None
    }

    /// Insert a new entry (caller guarantees key is absent). Returns the
    /// `(bucket_pool_idx, entry_idx)` where the entry landed.
    pub(crate) fn insert_entry(&mut self, hash: u64, key: K, value: V) -> (usize, usize) {
        loop {
            let dir_idx = self.dir_index(hash);
            let bucket_pool_idx = self.directory[dir_idx];

            if self.buckets[bucket_pool_idx].entries.len() < BUCKET_CAPACITY
                || self.buckets[bucket_pool_idx].local_depth >= MAX_LOCAL_DEPTH
            {
                let entry_idx = self.buckets[bucket_pool_idx].entries.len();
                self.buckets[bucket_pool_idx]
                    .entries
                    .push((hash, key, value));
                self.len += 1;
                return (bucket_pool_idx, entry_idx);
            }

            self.split_bucket(bucket_pool_idx);
        }
    }

    /// Split a full bucket: double the directory if needed, create a new
    /// sibling bucket, and redistribute entries based on the next hash bit.
    /// Only the overflowing bucket is touched — all other buckets remain
    /// undisturbed. This is the key latency advantage over full-table rehash.
    fn split_bucket(&mut self, bucket_pool_idx: usize) {
        let old_depth = self.buckets[bucket_pool_idx].local_depth;
        let new_depth = old_depth + 1;

        // Double the directory if the bucket's new depth exceeds global depth.
        if new_depth > self.global_depth {
            let old_size = self.directory.len();
            self.directory.reserve(old_size);
            for i in 0..old_size {
                let v = self.directory[i];
                self.directory.push(v);
            }
            self.global_depth += 1;
        }

        // Create the sibling bucket.
        let new_bucket_pool_idx = self.buckets.len();
        self.buckets.push(Bucket {
            local_depth: new_depth,
            entries: Vec::with_capacity(BUCKET_CAPACITY),
        });
        self.buckets[bucket_pool_idx].local_depth = new_depth;

        // Redistribute entries. The distinguishing bit is at position
        // `old_depth` (0-indexed from the LSB). Entries whose hash has this
        // bit set move to the new bucket; the rest stay.
        let bit = 1u64 << old_depth;
        let entries = std::mem::take(&mut self.buckets[bucket_pool_idx].entries);
        for entry in entries {
            if entry.0 & bit != 0 {
                self.buckets[new_bucket_pool_idx].entries.push(entry);
            } else {
                self.buckets[bucket_pool_idx].entries.push(entry);
            }
        }

        // Update directory: every slot that pointed to the old bucket and has
        // the distinguishing bit set now points to the sibling.
        for slot in 0..self.directory.len() {
            if self.directory[slot] == bucket_pool_idx && (slot as u64 & bit) != 0 {
                self.directory[slot] = new_bucket_pool_idx;
            }
        }
    }
}

impl<K: Clone, V: Clone> Clone for HashMapInner<K, V> {
    fn clone(&self) -> Self {
        Self {
            directory: self.directory.clone(),
            buckets: self.buckets.clone(),
            global_depth: self.global_depth,
            len: self.len,
        }
    }
}

// ---------------------------------------------------------------------------
// HashMap
// ---------------------------------------------------------------------------

/// A hash map using **extendible hashing**.
///
/// Instead of rehashing the entire table on growth (like `hashbrown`), or
/// maintaining two live tables during incremental migration (like `griddle`),
/// extendible hashing splits only the overflowing bucket. The directory — a
/// power-of-two array of bucket pointers — doubles when needed (a single
/// `realloc` of pointer-sized elements), keeping worst-case insert cost
/// proportional to the bucket size, not the table size.
///
/// The API mirrors [`std::collections::HashMap`] so the crate can serve as a
/// drop-in replacement.
pub struct HashMap<K, V, S = RandomState> {
    pub(crate) inner: HashMapInner<K, V>,
    hash_builder: S,
}

// --- Construction ----------------------------------------------------------

impl<K, V> HashMap<K, V, RandomState> {
    /// Creates an empty `HashMap`.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::with_hasher(RandomState::new())
    }

    /// Creates an empty `HashMap` with space for at least `capacity` entries
    /// without splitting.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::new())
    }
}

impl<K, V, S> HashMap<K, V, S> {
    /// Creates an empty `HashMap` using the given hash builder.
    #[inline]
    #[must_use]
    pub fn with_hasher(hash_builder: S) -> Self {
        Self::with_capacity_and_hasher(0, hash_builder)
    }

    /// Creates an empty `HashMap` with the given capacity and hash builder.
    ///
    /// Pre-allocates enough buckets so that `capacity` entries can be inserted
    /// before any bucket split occurs.
    #[must_use]
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        // Determine how many buckets we need. Each bucket holds
        // BUCKET_CAPACITY entries, and the directory size must be a power of
        // two. For capacity == 0 we start with a single bucket (global_depth
        // 0, directory size 1).
        let num_buckets = if capacity == 0 {
            1
        } else {
            capacity.div_ceil(BUCKET_CAPACITY).next_power_of_two()
        };
        let global_depth = num_buckets.trailing_zeros() as u8;
        let dir_size = 1usize << global_depth;

        let mut buckets = Vec::with_capacity(dir_size);
        let mut directory = Vec::with_capacity(dir_size);
        for i in 0..dir_size {
            buckets.push(Bucket {
                local_depth: global_depth,
                entries: Vec::with_capacity(BUCKET_CAPACITY),
            });
            directory.push(i);
        }

        Self {
            inner: HashMapInner {
                directory,
                buckets,
                global_depth,
                len: 0,
            },
            hash_builder,
        }
    }

    /// Returns a reference to the map's [`BuildHasher`].
    #[inline]
    #[must_use]
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Returns the number of elements in the map.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len
    }

    /// Returns `true` if the map contains no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    /// Returns the total number of entries the map can hold across all buckets
    /// without a bucket split.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.buckets.len() * BUCKET_CAPACITY
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter::new(&self.inner.buckets)
    }

    /// A mutable iterator visiting all key-value pairs in arbitrary order.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut::new(&mut self.inner.buckets)
    }

    /// An iterator visiting all keys in arbitrary order.
    #[inline]
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys(self.iter())
    }

    /// An iterator visiting all values in arbitrary order.
    #[inline]
    pub fn values(&self) -> Values<'_, K, V> {
        Values(self.iter())
    }

    /// A mutable iterator visiting all values in arbitrary order.
    #[inline]
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut(self.iter_mut())
    }

    /// Clears the map, removing all key-value pairs. Retains allocated
    /// buckets and directory.
    pub fn clear(&mut self) {
        for bucket in &mut self.inner.buckets {
            bucket.entries.clear();
        }
        self.inner.len = 0;
    }
}

// --- Hashing helper --------------------------------------------------------

impl<K, V, S: BuildHasher> HashMap<K, V, S> {
    #[inline]
    fn hash_key<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + ?Sized,
    {
        self.hash_builder.hash_one(key)
    }
}

// --- Lookup / mutation (requires K: Eq + Hash, S: BuildHasher) -------------

impl<K: Eq + Hash, V, S: BuildHasher> HashMap<K, V, S> {
    /// Returns a reference to the value corresponding to the key.
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash_key(key);
        let (bi, ei) = self.inner.find(hash, key)?;
        Some(&self.inner.buckets[bi].entries[ei].2)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    #[inline]
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash_key(key);
        let (bi, ei) = self.inner.find(hash, key)?;
        let entry = &self.inner.buckets[bi].entries[ei];
        Some((&entry.1, &entry.2))
    }

    /// Returns a mutable reference to the value corresponding to the key.
    #[inline]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash_key(key);
        let (bi, ei) = self.inner.find(hash, key)?;
        Some(&mut self.inner.buckets[bi].entries[ei].2)
    }

    /// Returns `true` if the map contains a value for the specified key.
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Inserts a key-value pair into the map. If the key was already present,
    /// the old value is returned and the entry is updated.
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = self.hash_key(&key);

        // If the key already exists, overwrite in place.
        if let Some((bi, ei)) = self.inner.find(hash, &key) {
            let old = std::mem::replace(&mut self.inner.buckets[bi].entries[ei].2, value);
            return Some(old);
        }

        self.inner.insert_entry(hash, key, value);
        None
    }

    /// Removes a key from the map, returning the value if the key was present.
    #[inline]
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.remove_entry(key).map(|(_, v)| v)
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was present.
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash_key(key);
        let (bi, ei) = self.inner.find(hash, key)?;
        // swap_remove is O(1) and order within a bucket is irrelevant.
        let (_, k, v) = self.inner.buckets[bi].entries.swap_remove(ei);
        self.inner.len -= 1;
        Some((k, v))
    }

    /// Gets the given key's corresponding entry in the map for in-place
    /// manipulation.
    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        let hash = self.hash_key(&key);
        if let Some((bi, ei)) = self.inner.find(hash, &key) {
            Entry::Occupied(OccupiedEntry::new(&mut self.inner, bi, ei))
        } else {
            Entry::Vacant(VacantEntry::new(&mut self.inner, key, hash))
        }
    }

    /// Creates a draining iterator that removes all entries from the map and
    /// yields them. The map is empty after this call returns.
    pub fn drain(&mut self) -> Drain<'_, K, V> {
        // Eagerly collect entries from all buckets. This avoids the complexity
        // of a lazy drain that must handle bucket/directory aliasing.
        let mut entries = Vec::with_capacity(self.inner.len);
        for bucket in &mut self.inner.buckets {
            for (_, k, v) in bucket.entries.drain(..) {
                entries.push((k, v));
            }
        }
        self.inner.len = 0;
        Drain::new(entries)
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        for bucket in &mut self.inner.buckets {
            let before = bucket.entries.len();
            bucket.entries.retain_mut(|entry| f(&entry.1, &mut entry.2));
            self.inner.len -= before - bucket.entries.len();
        }
    }
}

// --- Trait impls -----------------------------------------------------------

impl<K, V, S: Default> Default for HashMap<K, V, S> {
    #[inline]
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K: Clone, V: Clone, S: Clone> Clone for HashMap<K, V, S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug, S> fmt::Debug for HashMap<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Eq + Hash, V: PartialEq, S: BuildHasher> PartialEq for HashMap<K, V, S> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter()
            .all(|(k, v)| other.get(k).is_some_and(|ov| ov == v))
    }
}

impl<K: Eq + Hash, V: Eq, S: BuildHasher> Eq for HashMap<K, V, S> {}

impl<K: Eq + Hash, Q: ?Sized, V, S: BuildHasher> Index<&Q> for HashMap<K, V, S>
where
    K: Borrow<Q>,
    Q: Eq + Hash,
{
    type Output = V;

    /// Returns a reference to the value corresponding to the supplied key.
    ///
    /// # Panics
    ///
    /// Panics if the key is not present in the `HashMap`.
    #[inline]
    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<K: Eq + Hash, V, S: BuildHasher + Default> FromIterator<(K, V)> for HashMap<K, V, S> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = Self::with_capacity_and_hasher(lower, S::default());
        map.extend(iter);
        map
    }
}

impl<K: Eq + Hash, V, S: BuildHasher> Extend<(K, V)> for HashMap<K, V, S> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<'a, K: Eq + Hash + Copy, V: Copy, S: BuildHasher> Extend<(&'a K, &'a V)> for HashMap<K, V, S> {
    fn extend<I: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: I) {
        for (&k, &v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K, V, S> IntoIterator for HashMap<K, V, S> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        IntoIter::new(self.inner.buckets)
    }
}

impl<'a, K, V, S> IntoIterator for &'a HashMap<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a mut HashMap<K, V, S> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // -----------------------------------------------------------------------
    // Basic CRUD
    // -----------------------------------------------------------------------

    #[test]
    fn empty_map() {
        let map: HashMap<u32, u32> = HashMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.get(&1), None);
        assert!(!map.contains_key(&1));
    }

    #[test]
    fn insert_and_get() {
        let mut map = HashMap::new();
        assert_eq!(map.insert("hello", 1), None);
        assert_eq!(map.insert("world", 2), None);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("hello"), Some(&1));
        assert_eq!(map.get("world"), Some(&2));
        assert_eq!(map.get("missing"), None);
    }

    #[test]
    fn insert_overwrite() {
        let mut map = HashMap::new();
        assert_eq!(map.insert(1, "a"), None);
        assert_eq!(map.insert(1, "b"), Some("a"));
        assert_eq!(map.len(), 1);
        assert_eq!(map[&1], "b");
    }

    #[test]
    fn get_key_value() {
        let mut map = HashMap::new();
        map.insert(1, "one");
        assert_eq!(map.get_key_value(&1), Some((&1, &"one")));
        assert_eq!(map.get_key_value(&2), None);
    }

    #[test]
    fn get_mut() {
        let mut map = HashMap::new();
        map.insert(1, 10);
        *map.get_mut(&1).unwrap() += 5;
        assert_eq!(map[&1], 15);
    }

    #[test]
    fn remove() {
        let mut map = HashMap::new();
        map.insert(1, "a");
        map.insert(2, "b");
        assert_eq!(map.remove(&1), Some("a"));
        assert_eq!(map.len(), 1);
        assert_eq!(map.remove(&1), None);
        assert_eq!(map.remove(&2), Some("b"));
        assert!(map.is_empty());
    }

    #[test]
    fn remove_entry() {
        let mut map = HashMap::new();
        map.insert(1, "a");
        assert_eq!(map.remove_entry(&1), Some((1, "a")));
        assert!(map.is_empty());
    }

    #[test]
    fn clear() {
        let mut map = HashMap::new();
        for i in 0..100 {
            map.insert(i, i * 2);
        }
        assert_eq!(map.len(), 100);
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.get(&50), None);
    }

    // -----------------------------------------------------------------------
    // Splits / growth
    // -----------------------------------------------------------------------

    #[test]
    fn insert_triggers_splits() {
        // Insert enough entries to force multiple bucket splits and directory
        // doublings. Verify all entries are retrievable afterwards.
        let mut map = HashMap::new();
        let n = 1000;
        for i in 0..n {
            map.insert(i, i * 3);
        }
        assert_eq!(map.len(), n);
        for i in 0..n {
            assert_eq!(map.get(&i), Some(&(i * 3)), "missing key {i}");
        }
    }

    #[test]
    fn with_capacity_avoids_early_splits() {
        let map: HashMap<u32, u32> = HashMap::with_capacity(100);
        // Should have at least 100 / BUCKET_CAPACITY buckets, rounded up to
        // power of two.
        assert!(map.capacity() >= 100);
    }

    #[test]
    fn large_dataset() {
        let mut map = HashMap::new();
        let n = 50_000;
        for i in 0u64..n {
            map.insert(i, i.wrapping_mul(0x517cc1b727220a95));
        }
        assert_eq!(map.len(), n as usize);
        for i in 0u64..n {
            assert_eq!(
                map.get(&i),
                Some(&i.wrapping_mul(0x517cc1b727220a95)),
                "missing key {i}"
            );
        }
        // Remove half
        for i in (0u64..n).step_by(2) {
            assert!(map.remove(&i).is_some());
        }
        assert_eq!(map.len(), (n / 2) as usize);
        // Remaining half still intact
        for i in (1u64..n).step_by(2) {
            assert!(map.contains_key(&i));
        }
    }

    // -----------------------------------------------------------------------
    // Entry API
    // -----------------------------------------------------------------------

    #[test]
    fn entry_or_insert() {
        let mut map = HashMap::new();
        map.entry("a").or_insert(1);
        map.entry("a").or_insert(2);
        assert_eq!(map["a"], 1);
    }

    #[test]
    fn entry_or_insert_with() {
        let mut map = HashMap::new();
        map.entry(1).or_insert_with(|| 42);
        assert_eq!(map[&1], 42);
    }

    #[test]
    fn entry_or_insert_with_key() {
        let mut map = HashMap::new();
        map.entry(5).or_insert_with_key(|&k| k * 10);
        assert_eq!(map[&5], 50);
    }

    #[test]
    fn entry_and_modify() {
        let mut map = HashMap::new();
        map.insert(1, 10);
        map.entry(1).and_modify(|v| *v += 5).or_insert(0);
        assert_eq!(map[&1], 15);

        map.entry(2).and_modify(|v| *v += 5).or_insert(0);
        assert_eq!(map[&2], 0);
    }

    #[test]
    fn entry_or_default() {
        let mut map: HashMap<u32, Vec<u32>> = HashMap::new();
        map.entry(1).or_default().push(42);
        assert_eq!(map[&1], vec![42]);
    }

    #[test]
    fn entry_occupied_insert_remove() {
        let mut map = HashMap::new();
        map.insert(1, "a");
        match map.entry(1) {
            crate::Entry::Occupied(mut e) => {
                assert_eq!(e.key(), &1);
                assert_eq!(e.get(), &"a");
                assert_eq!(e.insert("b"), "a");
                assert_eq!(e.get(), &"b");
                assert_eq!(e.remove(), "b");
            }
            crate::Entry::Vacant(_) => panic!("expected occupied"),
        }
        assert!(map.is_empty());
    }

    #[test]
    fn entry_vacant_into_key() {
        let mut map: HashMap<String, u32> = HashMap::new();
        match map.entry("hello".to_string()) {
            crate::Entry::Vacant(e) => {
                assert_eq!(e.key(), "hello");
                let key = e.into_key();
                assert_eq!(key, "hello");
            }
            crate::Entry::Occupied(_) => panic!("expected vacant"),
        }
    }

    // -----------------------------------------------------------------------
    // Iterators
    // -----------------------------------------------------------------------

    #[test]
    fn iter_visits_all_entries() {
        let mut map = HashMap::new();
        let n = 200;
        for i in 0..n {
            map.insert(i, i);
        }
        let collected: HashSet<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(collected.len(), n);
        for i in 0..n {
            assert!(collected.contains(&(i, i)));
        }
    }

    #[test]
    fn iter_mut_modifies_values() {
        let mut map = HashMap::new();
        for i in 0..50 {
            map.insert(i, i);
        }
        for (_, v) in map.iter_mut() {
            *v *= 2;
        }
        for i in 0..50 {
            assert_eq!(map[&i], i * 2);
        }
    }

    #[test]
    fn into_iter_consumes() {
        let mut map = HashMap::new();
        for i in 0..50 {
            map.insert(i, i);
        }
        let collected: HashSet<_> = map.into_iter().collect();
        assert_eq!(collected.len(), 50);
    }

    #[test]
    fn keys_values() {
        let mut map = HashMap::new();
        map.insert(1, "a");
        map.insert(2, "b");
        let keys: HashSet<_> = map.keys().copied().collect();
        assert_eq!(keys, HashSet::from([1, 2]));
        let values: HashSet<_> = map.values().copied().collect();
        assert_eq!(values, HashSet::from(["a", "b"]));
    }

    #[test]
    fn values_mut() {
        let mut map = HashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        for v in map.values_mut() {
            *v += 1;
        }
        assert_eq!(map[&1], 11);
        assert_eq!(map[&2], 21);
    }

    // -----------------------------------------------------------------------
    // Drain / retain
    // -----------------------------------------------------------------------

    #[test]
    fn drain() {
        let mut map = HashMap::new();
        for i in 0..100 {
            map.insert(i, i);
        }
        let drained: Vec<_> = map.drain().collect();
        assert_eq!(drained.len(), 100);
        assert!(map.is_empty());
    }

    #[test]
    fn retain() {
        let mut map = HashMap::new();
        for i in 0..100 {
            map.insert(i, i);
        }
        map.retain(|&k, _| k % 2 == 0);
        assert_eq!(map.len(), 50);
        for i in 0..100 {
            if i % 2 == 0 {
                assert!(map.contains_key(&i));
            } else {
                assert!(!map.contains_key(&i));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Trait impls
    // -----------------------------------------------------------------------

    #[test]
    fn clone_and_eq() {
        let mut map = HashMap::new();
        for i in 0..50 {
            map.insert(i, i * 2);
        }
        let cloned = map.clone();
        assert_eq!(map, cloned);
    }

    #[test]
    fn partial_eq_different_content() {
        let mut a = HashMap::new();
        a.insert(1, 1);
        let mut b = HashMap::new();
        b.insert(1, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn partial_eq_different_len() {
        let mut a = HashMap::new();
        a.insert(1, 1);
        let b = HashMap::new();
        assert_ne!(a, b);
    }

    #[test]
    fn default_is_empty() {
        let map: HashMap<u32, u32> = HashMap::default();
        assert!(map.is_empty());
    }

    #[test]
    fn debug_output() {
        let mut map = HashMap::new();
        map.insert(1, 2);
        let s = format!("{map:?}");
        assert!(s.contains("1"));
        assert!(s.contains("2"));
    }

    #[test]
    fn from_iterator() {
        let map: HashMap<u32, u32> = vec![(1, 2), (3, 4)].into_iter().collect();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&1], 2);
        assert_eq!(map[&3], 4);
    }

    #[test]
    fn extend_from_iter() {
        let mut map = HashMap::new();
        map.insert(1, 1);
        map.extend(vec![(2, 2), (3, 3)]);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn extend_ref() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.extend([(&1, &2), (&3, &4)]);
        assert_eq!(map[&1], 2);
    }

    #[test]
    #[should_panic(expected = "no entry found for key")]
    fn index_missing_panics() {
        let map: HashMap<u32, u32> = HashMap::new();
        let _ = map[&1];
    }

    // -----------------------------------------------------------------------
    // Composite keys (matching Melin's patterns)
    // -----------------------------------------------------------------------

    #[test]
    fn composite_key_account_order() {
        // Simulates (AccountId, OrderId) → (Side, Price)
        type AccountId = u32;
        type OrderId = u64;
        let mut map: HashMap<(AccountId, OrderId), (u8, u64)> = HashMap::with_capacity(4096);

        for acct in 0..10u32 {
            for oid in 0..100u64 {
                map.insert((acct, oid), (acct as u8 % 2, oid * 100));
            }
        }
        assert_eq!(map.len(), 1000);
        assert_eq!(map.get(&(5, 50)), Some(&(1, 5000)));
        assert_eq!(map.remove(&(5, 50)), Some((1, 5000)));
        assert!(!map.contains_key(&(5, 50)));
    }

    #[test]
    fn composite_key_account_currency() {
        // Simulates (AccountId, CurrencyId) → Balance { available, reserved }
        let mut map: HashMap<(u32, u32), (u64, u64)> = HashMap::new();
        for acct in 0..100u32 {
            for cur in 0..5u32 {
                map.insert((acct, cur), (1_000_000, 0));
            }
        }
        assert_eq!(map.len(), 500);

        // Simulate reserve
        let bal = map.get_mut(&(42, 2)).unwrap();
        bal.0 -= 1000;
        bal.1 += 1000;
        assert_eq!(map[&(42, 2)], (999_000, 1000));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn single_element() {
        let mut map = HashMap::new();
        map.insert(42, "the answer");
        assert_eq!(map.len(), 1);
        assert_eq!(map[&42], "the answer");
        assert_eq!(map.remove(&42), Some("the answer"));
        assert!(map.is_empty());
    }

    #[test]
    fn reinsert_after_remove() {
        let mut map = HashMap::new();
        map.insert(1, "first");
        map.remove(&1);
        map.insert(1, "second");
        assert_eq!(map[&1], "second");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn many_removes_then_reinsert() {
        let mut map = HashMap::new();
        for i in 0..500 {
            map.insert(i, i);
        }
        for i in 0..500 {
            map.remove(&i);
        }
        assert!(map.is_empty());
        // Reinsert — buckets still exist but are empty.
        for i in 0..500 {
            map.insert(i, i + 1);
        }
        assert_eq!(map.len(), 500);
        for i in 0..500 {
            assert_eq!(map[&i], i + 1);
        }
    }

    #[test]
    fn with_capacity_zero() {
        let map: HashMap<u32, u32> = HashMap::with_capacity(0);
        assert!(map.is_empty());
        assert!(map.capacity() >= 1); // at least one bucket
    }

    #[test]
    fn hasher_accessor() {
        let map: HashMap<u32, u32> = HashMap::new();
        let _ = map.hasher(); // just verify it compiles and doesn't panic
    }

    #[test]
    fn for_loop_ref() {
        let mut map = HashMap::new();
        map.insert(1, 2);
        let mut seen = false;
        for (&k, &v) in &map {
            assert_eq!(k, 1);
            assert_eq!(v, 2);
            seen = true;
        }
        assert!(seen);
    }

    #[test]
    fn for_loop_mut() {
        let mut map = HashMap::new();
        map.insert(1, 10);
        for (_, v) in &mut map {
            *v += 1;
        }
        assert_eq!(map[&1], 11);
    }
}
