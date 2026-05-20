use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::mem::MaybeUninit;
use std::ops::Index;

use crate::DEFAULT_BUCKET_CAPACITY;
use crate::entry::{Entry, OccupiedEntry, VacantEntry};
use crate::iter::{Drain, IntoIter, Iter, IterMut, Keys, Values, ValuesMut};

/// Maximum local depth — bounded by the 32 low-order hash bits stored per
/// entry for split redistribution. 2^32 directory slots × N entries per
/// bucket far exceeds any realistic workload.
const MAX_LOCAL_DEPTH: u8 = 32;

/// Extract a non-zero fingerprint from the upper bits of a hash.
///
/// Uses bits 56–63 (independent of the directory's low-bit routing) and
/// forces bit 0 on so the result is **never zero**. Zero is reserved as
/// the "empty slot" sentinel, letting [`HashMapInner::find`] skip the
/// `len` read entirely on the scan path.
#[inline(always)]
fn fingerprint(hash: u64) -> u8 {
    (hash >> 56) as u8 | 1
}

// ---------------------------------------------------------------------------
// Bucket — inline storage, struct-of-arrays layout
// ---------------------------------------------------------------------------

/// A single bucket in the extendible hash table. Entries are stored **inline**
/// (not heap-allocated) for cache locality.
///
/// `repr(C, align(64))` guarantees the fingerprint array starts at offset 0 —
/// the beginning of a cache line. A lookup scans only this 8-byte array
/// (for the default N=8) before touching any keys or values, so the common
/// case (no match) reads just **one cache line**.
///
/// ## Layout (N = 8, representative KV types)
///
/// ```text
/// cache line 0:  fingerprints[8] + len + depth + pad + hash_low[8] + entries[0]…
/// cache line 1+: remaining entries
/// ```
///
/// The const generic `N` controls the number of entries per bucket. The
/// default (8) keeps the fingerprint array at 8 bytes—tiny compared to
/// the 64-byte hash array in the previous design—while `hash_low` stores
/// the 32 low-order hash bits needed for bucket splits.
///
/// Multiple directory slots may point to the same bucket when
/// `local_depth < global_depth`.
#[repr(C, align(64))]
pub(crate) struct Bucket<K, V, const N: usize> {
    /// Non-zero fingerprints for fast scan, indexed `0..len`. Derived from
    /// the upper 8 bits of each entry's hash (`(hash >> 56) | 1`). Slots
    /// beyond `len` are zero — the sentinel value that never matches any
    /// fingerprint, so `find` can skip the `len` read entirely.
    pub(crate) fingerprints: [u8; N],
    /// Number of initialized entries. Invariant: `len <= N`.
    pub(crate) len: u8,
    pub(crate) local_depth: u8,
    /// Lower 32 bits of each entry's hash. Used during bucket splits to
    /// determine which sibling an entry belongs to. 32 bits supports up to
    /// 2^32 directory slots—far beyond any realistic workload.
    pub(crate) hash_low: [u32; N],
    /// Key-value pairs, indexed `0..len`. Only `entries[0..len]` are
    /// initialized; the rest are `MaybeUninit`.
    pub(crate) entries: [MaybeUninit<(K, V)>; N],
}

impl<K, V, const N: usize> Bucket<K, V, N> {
    fn new(local_depth: u8) -> Self {
        Self {
            fingerprints: [0; N],
            len: 0,
            local_depth,
            hash_low: [0; N],
            // SAFETY: An array of MaybeUninit doesn't require initialization.
            entries: [const { MaybeUninit::uninit() }; N],
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len as usize
    }

    /// Push a new entry. The caller must ensure the bucket is not full
    /// (enforced by the split-before-insert loop in `insert_entry`).
    #[inline]
    fn push(&mut self, fp: u8, hash_low: u32, key: K, value: V) {
        let idx = self.len as usize;
        debug_assert!(
            idx < N,
            "bucket overflow at local_depth={}",
            self.local_depth
        );
        self.fingerprints[idx] = fp;
        self.hash_low[idx] = hash_low;
        self.entries[idx].write((key, value));
        self.len += 1;
    }

    /// Remove the entry at `idx` by swapping it with the last entry.
    /// Returns the removed `(key, value)`.
    pub(crate) fn swap_remove(&mut self, idx: usize) -> (K, V) {
        debug_assert!(idx < self.len as usize);
        let last = (self.len - 1) as usize;

        // SAFETY: `idx < len`, so `entries[idx]` is initialized. `assume_init_read`
        // copies the value out without dropping the source (we overwrite it below
        // or leave it as the now-uninit last slot).
        let (key, value) = unsafe { self.entries[idx].assume_init_read() };

        if idx != last {
            self.fingerprints[idx] = self.fingerprints[last];
            self.hash_low[idx] = self.hash_low[last];
            // SAFETY: `last < len` (old len), so `entries[last]` is initialized.
            // Move it into the vacated slot; `entries[last]` becomes uninit.
            unsafe {
                self.entries[idx].write(self.entries[last].assume_init_read());
            }
        }

        // Clear the vacated slot's fingerprint — zero is the "empty" sentinel.
        self.fingerprints[last] = 0;
        self.len -= 1;
        (key, value)
    }

    /// Return a slice over the initialized key-value pairs.
    pub(crate) fn entries_slice(&self) -> &[(K, V)] {
        // SAFETY: `entries[0..len]` are all initialized, and `MaybeUninit<T>`
        // has the same layout as `T`.
        unsafe { std::slice::from_raw_parts(self.entries.as_ptr().cast(), self.len as usize) }
    }

    /// Return a mutable slice over the initialized key-value pairs.
    pub(crate) fn entries_slice_mut(&mut self) -> &mut [(K, V)] {
        // SAFETY: same as `entries_slice`.
        unsafe {
            std::slice::from_raw_parts_mut(self.entries.as_mut_ptr().cast(), self.len as usize)
        }
    }

    /// Drop all initialized entries and reset len to 0.
    fn clear(&mut self) {
        for i in 0..self.len as usize {
            // SAFETY: `i < len`, so `entries[i]` is initialized.
            unsafe {
                self.entries[i].assume_init_drop();
            }
        }
        self.len = 0;
        // Zero fingerprints so the sentinel invariant holds (0 = empty slot).
        self.fingerprints = [0; N];
    }
}

impl<K, V, const N: usize> Drop for Bucket<K, V, N> {
    fn drop(&mut self) {
        for i in 0..self.len as usize {
            // SAFETY: `i < len`, so `entries[i]` is initialized.
            unsafe {
                self.entries[i].assume_init_drop();
            }
        }
    }
}

impl<K: Clone, V: Clone, const N: usize> Clone for Bucket<K, V, N> {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.local_depth);
        for i in 0..self.len as usize {
            // SAFETY: `i < len`, so `entries[i]` is initialized.
            let (k, v) = unsafe { self.entries[i].assume_init_ref() };
            new.push(self.fingerprints[i], self.hash_low[i], k.clone(), v.clone());
        }
        new
    }
}

// ---------------------------------------------------------------------------
// HashMapInner — hasher-agnostic core (allows Entry API without S parameter)
// ---------------------------------------------------------------------------

/// The guts of the hash map, deliberately separated from the hasher so that
/// [`Entry`], [`OccupiedEntry`], and [`VacantEntry`] do not need to carry the
/// `S` type parameter (matching the `std::collections::HashMap` entry API).
pub(crate) struct HashMapInner<K, V, const N: usize = DEFAULT_BUCKET_CAPACITY> {
    /// Directory of bucket indices. Length is always `2^global_depth`.
    /// `u32` instead of `usize` halves the directory footprint (e.g. a
    /// 32K-entry directory drops from 256 KB to 128 KB), keeping it
    /// tighter in L2 cache. Max 4 billion buckets is far beyond any
    /// realistic workload.
    pub(crate) directory: Vec<u32>,

    /// Pool of buckets. Each bucket appears at most once in the directory;
    /// the directory references buckets by their index in this vec. Indices
    /// in `freelist` are not referenced by the directory and are reused by
    /// the next `split_bucket` instead of growing the pool.
    pub(crate) buckets: Vec<Bucket<K, V, N>>,

    /// Indices of bucket slots in `buckets` that have been merged away and
    /// are no longer referenced by the directory. `split_bucket` pops from
    /// here before growing the pool; without this, high-churn workloads
    /// (insert-then-delete cycles with unique keys) would leak pool
    /// capacity as new splits keep pushing fresh buckets while merged
    /// ones sit unused.
    pub(crate) freelist: Vec<u32>,

    pub(crate) global_depth: u8,

    /// Precomputed `(1 << global_depth) - 1`. Stored to avoid a
    /// shift+subtract on every `dir_index` call — the hottest instruction
    /// in the lookup path. Always equal to `directory.len() - 1`.
    mask: usize,

    pub(crate) len: usize,
}

impl<K, V, const N: usize> HashMapInner<K, V, N> {
    /// Map a hash to its directory slot. Uses the precomputed mask to avoid
    /// a shift+subtract on every call.
    #[inline]
    fn dir_index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }
}

impl<K: Eq, V, const N: usize> HashMapInner<K, V, N> {
    /// Find `(bucket_pool_idx, entry_idx)` for a key, or `None`.
    ///
    /// Hot-path optimizations:
    /// - `get_unchecked` on directory and bucket access (indices are always
    ///   valid — `dir_idx` is masked, `bucket_idx` comes from our directory).
    /// - **Fingerprint scan**: compares N single-byte fingerprints instead
    ///   of N full u64 hashes — 8× less data touched (8 bytes vs 64 for
    ///   the default N=8), fitting in a single register load.
    /// - **Sentinel-based empty detection**: fingerprints are always non-zero
    ///   (`(hash >> 56) | 1`); empty slots hold zero. A lookup that misses
    ///   never reads `len` — it only touches the fingerprint array (first
    ///   cache line), avoiding a separate cache-line load.
    #[inline]
    fn find<Q>(&self, hash: u64, key: &Q) -> Option<(usize, usize)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let dir_idx = self.dir_index(hash);
        // SAFETY: `dir_idx = hash & mask` where `mask = directory.len() - 1`,
        // so `dir_idx` is always in bounds.
        let bucket_idx = unsafe { *self.directory.get_unchecked(dir_idx) } as usize;
        // SAFETY: all directory entries are valid bucket pool indices,
        // maintained by `with_capacity_and_hasher` and `split_bucket`.
        let bucket = unsafe { self.buckets.get_unchecked(bucket_idx) };

        let fp = fingerprint(hash);

        // Fixed-iteration fingerprint scan. Empty slots have fingerprint 0,
        // and `fp` is always non-zero (bit 0 forced on), so empty slots
        // never match — no need to mask by `len`. The fixed trip count lets
        // the compiler unroll this into branchless straight-line code.
        let mut match_bits: u32 = 0;
        for i in 0..N {
            if unsafe { *bucket.fingerprints.get_unchecked(i) } == fp {
                match_bits |= 1 << i;
            }
        }

        // Iterate only the matching positions (usually 0 or 1).
        while match_bits != 0 {
            let i = match_bits.trailing_zeros() as usize;
            match_bits &= match_bits - 1; // clear lowest set bit
            // SAFETY: `i` had a non-zero fingerprint, so it is an occupied
            // slot (`i < len`).
            let (k, _) = unsafe { bucket.entries.get_unchecked(i).assume_init_ref() };
            if k.borrow() == key {
                return Some((bucket_idx, i));
            }
        }
        None
    }

    /// Insert a new entry (caller guarantees key is absent). Returns the
    /// `(bucket_pool_idx, entry_idx)` where the entry landed.
    pub(crate) fn insert_entry(&mut self, hash: u64, key: K, value: V) -> (usize, usize) {
        let fp = fingerprint(hash);
        let hl = hash as u32;
        loop {
            let dir_idx = self.dir_index(hash);
            // SAFETY: same as `find` — dir_idx is masked, directory entries
            // are valid bucket indices.
            let bucket_pool_idx = unsafe { *self.directory.get_unchecked(dir_idx) } as usize;
            // SAFETY: all directory entries are valid bucket pool indices.
            let bucket = unsafe { self.buckets.get_unchecked(bucket_pool_idx) };

            if bucket.len() < N || bucket.local_depth >= MAX_LOCAL_DEPTH {
                // SAFETY: same index, now mutable.
                let bucket = unsafe { self.buckets.get_unchecked_mut(bucket_pool_idx) };
                let entry_idx = bucket.len();
                bucket.push(fp, hl, key, value);
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
    ///
    /// Cost breakdown:
    /// - Directory doubling (when needed): O(directory_size) memcpy
    /// - Entry redistribution: O(N)
    /// - Directory pointer update: O(2^(global_depth - local_depth)),
    ///   i.e. proportional to the number of directory slots aliasing this
    ///   bucket, NOT the full directory size.
    #[cold]
    fn split_bucket(&mut self, bucket_pool_idx: usize) {
        let old_depth = self.buckets[bucket_pool_idx].local_depth;
        let new_depth = old_depth + 1;

        // Capture the hash prefix before any structural changes. The low
        // `old_depth` bits of any entry's hash identify which directory
        // slots map to this bucket — we use this to target the update.
        let base_pattern =
            self.buckets[bucket_pool_idx].hash_low[0] as usize & ((1usize << old_depth) - 1);

        // Double the directory if the bucket's new depth exceeds global depth.
        // `extend_from_within` compiles to a single memcpy.
        if new_depth > self.global_depth {
            let old_size = self.directory.len();
            self.directory.extend_from_within(0..old_size);
            self.global_depth += 1;
            self.mask = self.directory.len() - 1;
        }

        // Create the sibling bucket — reuse a merged-away slot if one is
        // available so high-churn workloads don't leak pool capacity.
        let new_bucket_idx = if let Some(idx) = self.freelist.pop() {
            let idx = idx as usize;
            // Reset the recycled bucket. Invariants the freelist guarantees:
            // len == 0, fingerprints all zero (no double-drop, no false fp
            // matches on lookup). Setting local_depth completes the reset.
            debug_assert_eq!(self.buckets[idx].len, 0, "freed bucket must be empty");
            self.buckets[idx].local_depth = new_depth;
            idx
        } else {
            let idx = self.buckets.len();
            self.buckets.push(Bucket::new(new_depth));
            idx
        };
        self.buckets[bucket_pool_idx].local_depth = new_depth;

        // Collect entries from the old bucket into a stack-allocated buffer.
        // No heap allocation — N is small (typically 4-8).
        let bit = 1u32 << old_depth;
        let n = self.buckets[bucket_pool_idx].len();
        let mut fps_buf = [0u8; N];
        let mut hash_low_buf = [0u32; N];
        let mut entries_buf: [MaybeUninit<(K, V)>; N] = [const { MaybeUninit::uninit() }; N];
        for i in 0..n {
            fps_buf[i] = self.buckets[bucket_pool_idx].fingerprints[i];
            hash_low_buf[i] = self.buckets[bucket_pool_idx].hash_low[i];
            // SAFETY: `i < len`, so `entries[i]` is initialized. `assume_init_read`
            // copies the value; we reset `len = 0` below so the bucket won't
            // double-drop.
            entries_buf[i]
                .write(unsafe { self.buckets[bucket_pool_idx].entries[i].assume_init_read() });
        }
        self.buckets[bucket_pool_idx].len = 0;
        // Clear fingerprints so the sentinel invariant holds (0 = empty).
        self.buckets[bucket_pool_idx].fingerprints = [0; N];

        // Redistribute entries based on the distinguishing bit at position
        // `old_depth` (0-indexed from LSB) of the stored hash_low.
        for i in 0..n {
            let hl = hash_low_buf[i];
            let fp = fps_buf[i];
            // SAFETY: `entries_buf[i]` was initialized in the loop above for
            // all `i < n`.
            let (key, value) = unsafe { entries_buf[i].assume_init_read() };
            if hl & bit != 0 {
                self.buckets[new_bucket_idx].push(fp, hl, key, value);
            } else {
                self.buckets[bucket_pool_idx].push(fp, hl, key, value);
            }
        }

        // Targeted directory update: only visit slots that pointed to the
        // old bucket AND have the distinguishing bit set (these move to the
        // sibling). The affected slots start at `base_pattern | bit` and
        // repeat every `1 << new_depth` entries — touching exactly
        // 2^(global_depth - new_depth) slots instead of the full directory.
        let bucket_pool_idx_u32 = bucket_pool_idx as u32;
        let new_bucket_idx_u32 = new_bucket_idx as u32;
        let new_start = base_pattern | bit as usize;
        let step = 1usize << new_depth;
        let dir_size = self.directory.len();
        let mut slot = new_start;
        while slot < dir_size {
            debug_assert_eq!(
                self.directory[slot], bucket_pool_idx_u32,
                "directory slot {slot} should point to bucket {bucket_pool_idx} but points to {}",
                self.directory[slot]
            );
            self.directory[slot] = new_bucket_idx_u32;
            slot += step;
        }
    }

    /// Try to merge the bucket at `bucket_pool_idx` with its sibling. Called
    /// from `remove_entry` after a deletion. Two siblings can merge when they
    /// both sit at the same `local_depth` and their combined entry count fits
    /// in a single bucket — extendible hashing's symmetric counterpart to
    /// [`Self::split_bucket`]. Without this path, the bucket pool grows
    /// monotonically under high-churn workloads (insert/delete cycles with
    /// unique keys) as new inserts keep triggering splits while earlier
    /// removes leave previously-split buckets empty.
    ///
    /// `hash` is the hash of the just-removed key — used to locate the
    /// sibling without scanning. We deliberately do NOT shrink the directory
    /// even when global_depth could be decremented; the directory is 4 bytes
    /// per slot and the shrink would require scanning all buckets for the
    /// new max local_depth. Bucket-pool reclamation (where the KV data lives)
    /// is the load-bearing win; directory shrink is cosmetic.
    ///
    /// Cost:
    /// - Sibling lookup: O(1) (one directory read)
    /// - Entry move: O(N) (bucket capacity, small constant)
    /// - Directory pointer update: O(2^(global_depth - local_depth)),
    ///   same shape as split's targeted update
    #[cold]
    fn try_merge_bucket(&mut self, bucket_pool_idx: usize, hash: u64) {
        let bucket_local_depth = self.buckets[bucket_pool_idx].local_depth;
        if bucket_local_depth == 0 {
            // depth-0 bucket is the only one; nothing to merge with.
            return;
        }

        // The sibling sits at the directory slot whose distinguishing bit
        // (position `local_depth - 1`) is flipped vs our bucket's pattern.
        // We need only the low `local_depth` bits of `hash` to locate it.
        let d = bucket_local_depth as usize;
        let bucket_pattern = (hash as usize) & ((1usize << d) - 1);
        let sibling_pattern = bucket_pattern ^ (1usize << (d - 1));
        let sibling_dir_idx = sibling_pattern;
        // SAFETY: sibling_pattern < 2^d <= 2^global_depth = directory.len(),
        // and all directory entries are valid bucket pool indices.
        let sibling_pool_idx = unsafe { *self.directory.get_unchecked(sibling_dir_idx) } as usize;

        if sibling_pool_idx == bucket_pool_idx {
            // Same bucket — this happens only if our state is corrupt
            // (local_depth out of sync with directory). Be defensive.
            return;
        }

        // Sibling must also be at depth `d`. If it's deeper, it was split
        // further and merge would violate the depth invariant. (It can't be
        // shallower than `d` and still be a sibling at this level.)
        let sibling_depth = self.buckets[sibling_pool_idx].local_depth;
        if sibling_depth != bucket_local_depth {
            return;
        }

        // Hysteresis threshold: merge only when the combined load is at
        // most N/2 (half-full after merge). With the naive `> N` threshold,
        // a workload that oscillates around a bucket's split boundary
        // (insert→split→remove→merge→insert→split→...) pays two structural
        // reorganisations per cycle; each merge is O(2^(global_depth -
        // local_depth)) directory writes, which dominates throughput on
        // deep directories. Requiring 50% headroom after merge means a
        // subsequent split needs at least N/2 + 1 inserts before firing —
        // amortising the merge cost over many operations.
        let bucket_len = self.buckets[bucket_pool_idx].len();
        let sibling_len = self.buckets[sibling_pool_idx].len();
        if bucket_len + sibling_len > N / 2 {
            return;
        }

        // Move sibling's entries into a stack buffer first to avoid a
        // double mutable borrow on `self.buckets` during the push loop —
        // same pattern as `split_bucket`.
        let mut fps_buf = [0u8; N];
        let mut hash_low_buf = [0u32; N];
        let mut entries_buf: [MaybeUninit<(K, V)>; N] = [const { MaybeUninit::uninit() }; N];
        for i in 0..sibling_len {
            fps_buf[i] = self.buckets[sibling_pool_idx].fingerprints[i];
            hash_low_buf[i] = self.buckets[sibling_pool_idx].hash_low[i];
            // SAFETY: `i < sibling_len`, so `entries[i]` is initialized.
            // We zero `len` and `fingerprints` below so sibling won't
            // double-drop the moved-out values.
            entries_buf[i]
                .write(unsafe { self.buckets[sibling_pool_idx].entries[i].assume_init_read() });
        }
        // Sibling is now logically empty. Reset state to satisfy the
        // freelist invariant: empty bucket with zeroed fingerprints so
        // future reuse via `split_bucket` starts from a clean slate and
        // no false fingerprint matches can fire during `find` (which
        // would walk through this freed bucket otherwise, but only via
        // the directory — which we update below to stop pointing here).
        self.buckets[sibling_pool_idx].len = 0;
        self.buckets[sibling_pool_idx].fingerprints = [0; N];

        // Push sibling's entries into the surviving bucket.
        for i in 0..sibling_len {
            // SAFETY: `entries_buf[i]` was initialized in the loop above
            // for all `i < sibling_len`.
            let (k, v) = unsafe { entries_buf[i].assume_init_read() };
            self.buckets[bucket_pool_idx].push(fps_buf[i], hash_low_buf[i], k, v);
        }

        // Surviving bucket's local_depth drops by 1; it now covers both
        // halves of the previous sibling pair.
        self.buckets[bucket_pool_idx].local_depth = bucket_local_depth - 1;

        // Targeted directory update: every slot whose low `d` bits match
        // `sibling_pattern` currently points at the sibling — redirect
        // them to the surviving bucket. Same stride pattern as split's
        // post-split fixup.
        let surviving_u32 = bucket_pool_idx as u32;
        let sibling_u32 = sibling_pool_idx as u32;
        let stride = 1usize << d;
        let dir_size = self.directory.len();
        let mut slot = sibling_pattern;
        while slot < dir_size {
            debug_assert_eq!(
                self.directory[slot], sibling_u32,
                "directory slot {slot} should point to sibling bucket {sibling_pool_idx} but points to {}",
                self.directory[slot]
            );
            self.directory[slot] = surviving_u32;
            slot += stride;
        }

        // Recycle sibling's pool slot.
        self.freelist.push(sibling_u32);
    }
}

impl<K: Clone, V: Clone, const N: usize> Clone for HashMapInner<K, V, N> {
    fn clone(&self) -> Self {
        Self {
            directory: self.directory.clone(),
            buckets: self.buckets.clone(),
            freelist: self.freelist.clone(),
            global_depth: self.global_depth,
            mask: self.mask,
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
/// Buckets use **inline storage** with a struct-of-arrays layout: hashes and
/// key-value pairs are stored in fixed-size arrays inside the bucket struct
/// (no per-bucket heap allocation). Lookups scan just the hash array before
/// touching any key data.
///
/// # Bucket capacity (`N`)
///
/// The const generic `N` controls entries per bucket. The default (8) puts
/// the hash array at exactly one cache line (64 bytes). Smaller values
/// (e.g. 4) reduce per-lookup scan cost and bucket pool footprint at the
/// expense of more frequent splits. Larger values (e.g. 16) do the opposite.
///
/// The API mirrors [`std::collections::HashMap`] so the crate can serve as a
/// drop-in replacement.
pub struct HashMap<K, V, S = RandomState, const N: usize = DEFAULT_BUCKET_CAPACITY> {
    pub(crate) inner: HashMapInner<K, V, N>,
    hash_builder: S,
}

// --- Construction ----------------------------------------------------------

impl<K, V> HashMap<K, V> {
    /// Creates an empty `HashMap` with the default bucket capacity.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::with_hasher(RandomState::new())
    }

    /// Creates an empty `HashMap` with space for at least `capacity` entries
    /// without splitting, using the default bucket capacity.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::new())
    }
}

impl<K, V, S, const N: usize> HashMap<K, V, S, N> {
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
        // Determine how many buckets we need. Each bucket holds N entries,
        // and the directory size must be a power of two. For capacity == 0
        // we start with a single bucket (global_depth 0, directory size 1).
        let num_buckets = if capacity == 0 {
            1
        } else {
            capacity.div_ceil(N).next_power_of_two()
        };
        let global_depth = num_buckets.trailing_zeros() as u8;
        let dir_size = 1usize << global_depth;

        // Reserve 2× the initial bucket count. Under uniform hashing, each
        // initial bucket splits roughly once as the map fills to capacity,
        // doubling the bucket count. Pre-allocating avoids a Vec reallocation
        // that would memcpy ALL bucket data, which would defeat the "only
        // touch one bucket per split" latency guarantee.
        let mut buckets = Vec::with_capacity(dir_size.saturating_mul(2).max(1));
        let mut directory: Vec<u32> = Vec::with_capacity(dir_size);
        for i in 0..dir_size {
            buckets.push(Bucket::new(global_depth));
            directory.push(i as u32);
        }

        Self {
            inner: HashMapInner {
                directory,
                buckets,
                freelist: Vec::new(),
                global_depth,
                mask: dir_size - 1,
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
        self.inner.buckets.len() * N
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V, N> {
        Iter::new(&self.inner.buckets)
    }

    /// A mutable iterator visiting all key-value pairs in arbitrary order.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, N> {
        IterMut::new(&mut self.inner.buckets)
    }

    /// An iterator visiting all keys in arbitrary order.
    #[inline]
    pub fn keys(&self) -> Keys<'_, K, V, N> {
        Keys(self.iter())
    }

    /// An iterator visiting all values in arbitrary order.
    #[inline]
    pub fn values(&self) -> Values<'_, K, V, N> {
        Values(self.iter())
    }

    /// A mutable iterator visiting all values in arbitrary order.
    #[inline]
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V, N> {
        ValuesMut(self.iter_mut())
    }

    /// Clears the map, removing all key-value pairs. Retains allocated
    /// buckets and directory.
    pub fn clear(&mut self) {
        for bucket in &mut self.inner.buckets {
            bucket.clear();
        }
        self.inner.len = 0;
    }
}

// --- Hashing helper --------------------------------------------------------

impl<K, V, S: BuildHasher, const N: usize> HashMap<K, V, S, N> {
    #[inline]
    fn hash_key<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + ?Sized,
    {
        self.hash_builder.hash_one(key)
    }
}

// --- Lookup / mutation (requires K: Eq + Hash, S: BuildHasher) -------------

impl<K: Eq + Hash, V, S: BuildHasher, const N: usize> HashMap<K, V, S, N> {
    /// Returns a reference to the value corresponding to the key.
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash_key(key);
        let (bi, ei) = self.inner.find(hash, key)?;
        // SAFETY: `find` guarantees `bi` is a valid bucket index and
        // `ei < bucket.len()`.
        Some(
            &unsafe {
                self.inner
                    .buckets
                    .get_unchecked(bi)
                    .entries
                    .get_unchecked(ei)
                    .assume_init_ref()
            }
            .1,
        )
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
        // SAFETY: `find` guarantees valid indices.
        let (k, v) = unsafe {
            self.inner
                .buckets
                .get_unchecked(bi)
                .entries
                .get_unchecked(ei)
                .assume_init_ref()
        };
        Some((k, v))
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
        // SAFETY: `find` guarantees valid indices.
        Some(
            &mut unsafe {
                self.inner
                    .buckets
                    .get_unchecked_mut(bi)
                    .entries
                    .get_unchecked_mut(ei)
                    .assume_init_mut()
            }
            .1,
        )
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
            // SAFETY: `find` guarantees valid indices.
            let entry = unsafe {
                self.inner
                    .buckets
                    .get_unchecked_mut(bi)
                    .entries
                    .get_unchecked_mut(ei)
                    .assume_init_mut()
            };
            let old = std::mem::replace(&mut entry.1, value);
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
        // SAFETY: `find` guarantees `bi` is a valid bucket index.
        // swap_remove is O(1) and order within a bucket is irrelevant.
        let (k, v) = unsafe { self.inner.buckets.get_unchecked_mut(bi) }.swap_remove(ei);
        self.inner.len -= 1;
        // Symmetric to split-on-overflow: attempt to merge with the sibling
        // if both buckets are now sparse enough to fit in one. Bounds the
        // directory under high-churn workloads where unique-key inserts and
        // matching deletes would otherwise grow the bucket pool unbounded.
        self.inner.try_merge_bucket(bi, hash);
        Some((k, v))
    }

    /// Gets the given key's corresponding entry in the map for in-place
    /// manipulation.
    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, N> {
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
        let mut entries = Vec::with_capacity(self.inner.len);
        for bucket in &mut self.inner.buckets {
            for i in 0..bucket.len() {
                // SAFETY: `i < bucket.len()`, so `entries[i]` is initialized.
                let entry = unsafe { bucket.entries[i].assume_init_read() };
                entries.push(entry);
            }
            bucket.len = 0;
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
            let mut i = 0;
            while i < bucket.len() {
                // SAFETY: `i < bucket.len()`, so `entries[i]` is initialized.
                let entry = unsafe { bucket.entries[i].assume_init_mut() };
                if f(&entry.0, &mut entry.1) {
                    i += 1;
                } else {
                    bucket.swap_remove(i);
                    self.inner.len -= 1;
                    // Don't increment — the swapped-in entry needs checking.
                }
            }
        }
    }
}

// --- Trait impls -----------------------------------------------------------

impl<K, V, S: Default, const N: usize> Default for HashMap<K, V, S, N> {
    #[inline]
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K: Clone, V: Clone, S: Clone, const N: usize> Clone for HashMap<K, V, S, N> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug, S, const N: usize> fmt::Debug for HashMap<K, V, S, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Eq + Hash, V: PartialEq, S: BuildHasher, const N: usize> PartialEq for HashMap<K, V, S, N> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter()
            .all(|(k, v)| other.get(k).is_some_and(|ov| ov == v))
    }
}

impl<K: Eq + Hash, V: Eq, S: BuildHasher, const N: usize> Eq for HashMap<K, V, S, N> {}

impl<K: Eq + Hash, Q: ?Sized, V, S: BuildHasher, const N: usize> Index<&Q> for HashMap<K, V, S, N>
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

impl<K: Eq + Hash, V, S: BuildHasher + Default, const N: usize> FromIterator<(K, V)>
    for HashMap<K, V, S, N>
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = Self::with_capacity_and_hasher(lower, S::default());
        map.extend(iter);
        map
    }
}

impl<K: Eq + Hash, V, S: BuildHasher, const N: usize> Extend<(K, V)> for HashMap<K, V, S, N> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<'a, K: Eq + Hash + Copy, V: Copy, S: BuildHasher, const N: usize> Extend<(&'a K, &'a V)>
    for HashMap<K, V, S, N>
{
    fn extend<I: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: I) {
        for (&k, &v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K, V, S, const N: usize> IntoIterator for HashMap<K, V, S, N> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, N>;

    fn into_iter(self) -> IntoIter<K, V, N> {
        IntoIter::new(self.inner.buckets)
    }
}

impl<'a, K, V, S, const N: usize> IntoIterator for &'a HashMap<K, V, S, N> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, N>;

    fn into_iter(self) -> Iter<'a, K, V, N> {
        self.iter()
    }
}

impl<'a, K, V, S, const N: usize> IntoIterator for &'a mut HashMap<K, V, S, N> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V, N>;

    fn into_iter(self) -> IterMut<'a, K, V, N> {
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
        for i in (0u64..n).step_by(2) {
            assert!(map.remove(&i).is_some());
        }
        assert_eq!(map.len(), (n / 2) as usize);
        for i in (1u64..n).step_by(2) {
            assert!(map.contains_key(&i));
        }
    }

    // -----------------------------------------------------------------------
    // Custom bucket capacity
    // -----------------------------------------------------------------------

    #[test]
    fn bucket_capacity_4() {
        let mut map: HashMap<u32, u32, RandomState, 4> = HashMap::with_hasher(RandomState::new());
        for i in 0..500 {
            map.insert(i, i * 2);
        }
        assert_eq!(map.len(), 500);
        for i in 0..500 {
            assert_eq!(map[&i], i * 2);
        }
    }

    #[test]
    fn bucket_capacity_16() {
        let mut map: HashMap<u32, u32, RandomState, 16> = HashMap::with_hasher(RandomState::new());
        for i in 0..500 {
            map.insert(i, i * 2);
        }
        assert_eq!(map.len(), 500);
        for i in 0..500 {
            assert_eq!(map[&i], i * 2);
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
        let mut map: HashMap<(u32, u32), (u64, u64)> = HashMap::new();
        for acct in 0..100u32 {
            for cur in 0..5u32 {
                map.insert((acct, cur), (1_000_000, 0));
            }
        }
        assert_eq!(map.len(), 500);
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
        assert!(map.capacity() >= 1);
    }

    #[test]
    fn hasher_accessor() {
        let map: HashMap<u32, u32> = HashMap::new();
        let _ = map.hasher();
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

    // -----------------------------------------------------------------------
    // Identity hasher — deterministic control over hash bit patterns
    // -----------------------------------------------------------------------

    struct IdentityHasher(u64);

    impl std::hash::Hasher for IdentityHasher {
        fn finish(&self) -> u64 {
            self.0
        }
        fn write(&mut self, bytes: &[u8]) {
            self.0 = 0;
            for (i, &b) in bytes.iter().enumerate().take(8) {
                self.0 |= (b as u64) << (i * 8);
            }
        }
    }

    #[derive(Clone, Default)]
    struct IdentityBuildHasher;

    impl std::hash::BuildHasher for IdentityBuildHasher {
        type Hasher = IdentityHasher;
        fn build_hasher(&self) -> IdentityHasher {
            IdentityHasher(0)
        }
    }

    fn identity_map() -> HashMap<u64, u64, IdentityBuildHasher> {
        HashMap::with_hasher(IdentityBuildHasher)
    }

    #[test]
    fn split_distributes_by_next_bit() {
        let mut map = identity_map();
        for i in 0u64..8 {
            map.insert(i * 2, i);
        }
        assert_eq!(map.inner.global_depth, 0);
        map.insert(1, 100);
        assert!(map.inner.global_depth >= 1);
        assert_eq!(map.len(), 9);
        assert_eq!(map[&1], 100);
        for i in 0u64..8 {
            assert_eq!(map[&(i * 2)], i);
        }
    }

    #[test]
    fn split_directory_pointers_correct() {
        let mut map = identity_map();
        for i in 0u64..64 {
            map.insert(i, i * 7);
        }
        assert_eq!(map.len(), 64);
        for i in 0u64..64 {
            assert_eq!(map.get(&i), Some(&(i * 7)), "missing key {i}");
        }
        assert_eq!(map.inner.directory.len(), 1 << map.inner.global_depth);
        for &bi in &map.inner.directory {
            let bi = bi as usize;
            assert!(bi < map.inner.buckets.len());
            assert!(map.inner.buckets[bi].local_depth <= map.inner.global_depth);
        }
    }

    #[test]
    fn split_all_same_low_bits_forces_repeated_splits() {
        let mut map = identity_map();
        let keys: Vec<u64> = (0..9).map(|i| 5 + i * 8).collect();
        for &k in &keys {
            map.insert(k, k);
        }
        assert_eq!(map.len(), 9);
        for &k in &keys {
            assert_eq!(map[&k], k);
        }
    }

    #[test]
    fn targeted_directory_update_with_deep_split() {
        let mut map = identity_map();
        for i in 0u64..64 {
            map.insert(i, i);
        }
        let gd_after_phase1 = map.inner.global_depth;
        for i in 64u64..264 {
            map.insert(i, i);
        }
        assert_eq!(map.len(), 264);
        for i in 0u64..264 {
            assert_eq!(
                map.get(&i),
                Some(&i),
                "missing key {i} (gd was {gd_after_phase1}, now {})",
                map.inner.global_depth
            );
        }
        let dir_size = map.inner.directory.len();
        assert_eq!(dir_size, 1 << map.inner.global_depth);
        for slot in 0..dir_size {
            let bi = map.inner.directory[slot] as usize;
            assert!(
                bi < map.inner.buckets.len(),
                "slot {slot} → invalid bucket {bi}"
            );
            let bucket = &map.inner.buckets[bi];
            assert!(
                bucket.local_depth <= map.inner.global_depth,
                "bucket {bi} local_depth {} > global_depth {}",
                bucket.local_depth,
                map.inner.global_depth
            );
            let local_mask = (1usize << bucket.local_depth) - 1;
            for i in 0..bucket.len() {
                let hash_low = bucket.hash_low[i];
                assert_eq!(
                    hash_low as usize & local_mask,
                    slot & local_mask,
                    "entry with hash_low {hash_low:#x} in bucket {bi} (slot {slot}) violates local_depth {} prefix",
                    bucket.local_depth
                );
            }
        }
    }

    #[test]
    fn all_entries_counted_once_via_bucket_pool() {
        let mut map = identity_map();
        for i in 0u64..500 {
            map.insert(i, i);
        }
        let iter_count = map.iter().count();
        assert_eq!(iter_count, 500);
        assert_eq!(iter_count, map.len());
        let pool_count: usize = map.inner.buckets.iter().map(|b| b.len()).sum();
        assert_eq!(pool_count, 500);
    }

    /// Inserting enough to force splits, then removing all entries, should
    /// drain entries (`len` goes to 0) AND recycle the now-empty buckets
    /// onto the freelist so the pool's live bucket count returns toward
    /// the initial allocation. Without merge, the pool monotonically
    /// grows even when the map is empty.
    #[test]
    fn merge_after_full_drain_recycles_buckets() {
        let mut map = HashMap::new();
        // Insert enough to cause many splits.
        for i in 0u64..1000 {
            map.insert(i, i);
        }
        let live_buckets_before_remove = map.inner.buckets.len() - map.inner.freelist.len();
        assert!(
            live_buckets_before_remove > 1,
            "expected splits to have grown the pool, got {} live buckets",
            live_buckets_before_remove,
        );

        for i in 0u64..1000 {
            map.remove(&i);
        }
        assert_eq!(map.len(), 0);

        // After draining, the freelist should hold most of what splits
        // allocated — only a small constant (1-2) of live buckets should
        // remain in the directory.
        let live_buckets_after_remove = map.inner.buckets.len() - map.inner.freelist.len();
        assert!(
            live_buckets_after_remove <= 2,
            "expected merge to recycle buckets, but {} are still live",
            live_buckets_after_remove,
        );
    }

    /// Under a high-churn workload (each key inserted then removed once,
    /// with unique keys), the bucket pool should stay bounded — the
    /// classic motivating case for merge. Without it, the pool grows in
    /// proportion to total inserts, not live count.
    #[test]
    fn merge_bounds_pool_under_high_churn() {
        let mut map = HashMap::new();
        // 100 cycles of "insert a batch, then delete that batch" with
        // unique keys per cycle. Total inserts = 10,000; live count
        // stays at most ~100 at any moment.
        for cycle in 0u64..100 {
            let base = cycle * 100;
            for i in 0..100 {
                map.insert(base + i, base + i);
            }
            for i in 0..100 {
                map.remove(&(base + i));
            }
        }
        assert_eq!(map.len(), 0);

        // Without merge, the pool would grow to ~1250+ buckets
        // (extendible-hashing-under-churn pathology). With merge, the
        // live bucket count stays bounded by what's needed for the peak
        // live size (~100 entries → ~13 buckets at N=8 with full
        // occupancy, plus some slack).
        let live_buckets = map.inner.buckets.len() - map.inner.freelist.len();
        assert!(
            live_buckets < 64,
            "expected merge to bound the pool, got {} live buckets",
            live_buckets,
        );
    }

    /// After a merge, lookups for surviving entries must still work and
    /// every directory slot must point to a bucket whose local_depth is
    /// consistent with the slot's hash prefix.
    #[test]
    fn merge_preserves_directory_invariants() {
        let mut map = HashMap::new();
        for i in 0u64..500 {
            map.insert(i, i);
        }
        // Remove half — many of the resulting empty buckets should merge.
        for i in 0u64..250 {
            map.remove(&i);
        }
        assert_eq!(map.len(), 250);

        // Surviving entries are still findable.
        for i in 250u64..500 {
            assert_eq!(map.get(&i), Some(&i), "surviving key {i} lost after merge",);
        }

        // Directory invariants: every slot points to a valid bucket; bucket's
        // local_depth matches the slot's hash prefix (slot & local_mask must
        // equal the bucket's hash_low prefix for every entry in it).
        let dir_size = map.inner.directory.len();
        let freelist: std::collections::HashSet<u32> = map.inner.freelist.iter().copied().collect();
        for slot in 0..dir_size {
            let bi = map.inner.directory[slot] as usize;
            assert!(bi < map.inner.buckets.len(), "slot {slot} → bad bi {bi}");
            assert!(
                !freelist.contains(&(bi as u32)),
                "slot {slot} points to freed bucket {bi}",
            );
            let bucket = &map.inner.buckets[bi];
            let local_mask = (1usize << bucket.local_depth) - 1;
            for i in 0..bucket.len() {
                let hash_low = bucket.hash_low[i] as usize;
                assert_eq!(
                    hash_low & local_mask,
                    slot & local_mask,
                    "slot {slot} → bucket {bi} entry {i} hash mismatch",
                );
            }
        }
    }

    /// Re-inserting keys after a merge cycle must reuse freelist slots
    /// before extending the pool — otherwise merge would be cosmetic.
    #[test]
    fn merge_freelist_reused_by_subsequent_splits() {
        let mut map = HashMap::new();
        for i in 0u64..1000 {
            map.insert(i, i);
        }
        let pool_after_insert = map.inner.buckets.len();
        for i in 0u64..1000 {
            map.remove(&i);
        }
        let freelist_after_remove = map.inner.freelist.len();
        assert!(
            freelist_after_remove > 0,
            "expected merges to have populated the freelist",
        );

        // Re-insert: future splits should pop from the freelist first.
        for i in 1000u64..2000 {
            map.insert(i, i);
        }
        let pool_after_reinsert = map.inner.buckets.len();
        assert!(
            pool_after_reinsert <= pool_after_insert + 1,
            "pool grew during reinsert ({} → {}) instead of reusing freelist",
            pool_after_insert,
            pool_after_reinsert,
        );
        // All reinserts present.
        for i in 1000u64..2000 {
            assert_eq!(map[&i], i);
        }
    }
}
