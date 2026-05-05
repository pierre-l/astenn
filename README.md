# Astenn

An extendible-hashing `HashMap` for Rust, designed as a drop-in replacement for
`std::collections::HashMap` with deterministic latency under growth.

Built for low-latency workloads where p99.9 matters more than throughput.

## Why extendible hashing?

`std::collections::HashMap` (hashbrown) uses linear probing with full-table
rehash on growth — a single insert can trigger an O(n) copy. `griddle` softens
this by spreading the rehash across inserts, but at the cost of two live tables
during migration.

Astenn uses **extendible hashing**: a directory of pointers to fixed-size
buckets. When a bucket overflows, only that bucket splits — the directory
itself doubles in place (a single pointer-array realloc), but no entries are
re-hashed beyond the overflowing bucket.

The result: amortised O(1) inserts with a worst-case spike bounded by **one
bucket split** (tens of entries), not the whole table. No dual-table state, no
incremental migration bookkeeping.

## Design

- **Inline bucket storage.** Entries live directly inside buckets, not behind
  a pointer — keys/values are one cache-line miss away from the directory
  lookup, not two.
- **Cache-line-aligned buckets** (`#[repr(C, align(64))]`). The default
  `N = 8` keeps the fingerprint array at exactly 8 bytes, so the scan path
  reads a single cache line in the no-match case.
- **SIMD fingerprint scan.** An 8-bit non-zero fingerprint per entry, derived
  from the high hash bits. On `x86_64`, lookup uses SSE2
  (`pmovmskb` over `pcmpeqb`) — 4 instructions, ~3 cycles to test all 8 slots.
  A SWAR fallback handles other architectures in ~6 ALU ops.
- **Zero is the empty sentinel.** Fingerprints are forced non-zero, so unused
  slots never match. The scan path doesn't need to read `len`.
- **Stored low hash bits.** The lower 32 bits of each entry's hash are kept
  alongside the fingerprint, so bucket splits redistribute entries without
  re-hashing keys.
- **Const-generic bucket capacity.** `HashMap<K, V, S, N>` exposes `N` for
  workloads that want wider or narrower buckets than the 8-entry default.

## API

Drop-in compatible with `std::collections::HashMap`:

- Construction: `new`, `with_capacity`, `with_hasher`, `with_capacity_and_hasher`
- Access: `get`, `get_mut`, `contains_key`, `insert`, `remove`, `Index`
- Entry API: `entry`, `Entry`, `OccupiedEntry`, `VacantEntry`
- Iteration: `iter`, `iter_mut`, `into_iter`, `keys`, `values`, `values_mut`,
  `drain`, `retain`, `clear`
- Traits: `Debug`, `Clone`, `Default`, `PartialEq`, `Eq`, `FromIterator`, `Extend`
- Parameterised hasher (defaults to `RandomState`)

## Usage

```toml
[dependencies]
astenn = { path = "../astenn" }
```

```rust
use astenn::HashMap;

let mut map: HashMap<(u32, u64), (u8, u64)> = HashMap::new();
map.insert((1, 42), (0, 10_000));
assert_eq!(map.get(&(1, 42)), Some(&(0, 10_000)));
```

With a custom hasher and bucket capacity:

```rust
use astenn::HashMap;
use rustc_hash::FxBuildHasher;

let mut map: HashMap<u64, u64, FxBuildHasher, 16> =
    HashMap::with_capacity_and_hasher(1024, FxBuildHasher);
```

## Performance targets

| Operation     | Budget      |
|---------------|-------------|
| `get` / `get_mut` | 20–50 ns    |
| `insert`      | 20–50 ns, no spike > ~5 µs |
| `remove`      | 20–50 ns    |
| `contains_key`| 20–50 ns    |

Working sets the map is tuned for:

- ~4 K entries (fits L2)
- ~32 K entries (fits L3)
- Up to ~1 M entries

## Build

```sh
cargo build
cargo test
cargo bench       # criterion microbenchmarks vs. std::HashMap
cargo clippy
cargo fmt
```
