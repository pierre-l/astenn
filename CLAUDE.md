# Astenn

Extendible hashing hashmap for Rust, built as a drop-in replacement for
`std::collections::HashMap`. Primary consumer is **Melin**, the order matching
engine in `../trading`.

## Goals

### 1. Extendible hashing

Use extendible hashing instead of the linear-probing / Robin Hood scheme in
hashbrown (and by extension the std HashMap). The core idea: a directory of
pointers to buckets, doubling the directory (not the entire table) when a
bucket overflows. This gives **O(1) amortised insert with bounded worst-case
rehash cost** — only the overflowing bucket is split, never the full table.

This replaces griddle's incremental-rehash approach currently used in Melin
(`griddle::HashMap<K, V, FxBuildHasher>`). Where griddle keeps two live tables
during migration and spreads rehash across inserts, extendible hashing avoids
the dual-table state entirely: the directory doubles in place (a single
pointer-array realloc) and only one bucket is redistributed per split.

### 2. std HashMap API compatibility

Implement the same public API as `std::collections::HashMap` so the crate is a
drop-in replacement. At minimum:

- `new`, `with_capacity`, `len`, `is_empty`, `capacity`
- `insert`, `remove`, `get`, `get_mut`, `contains_key`
- `entry` API (`Entry`, `OccupiedEntry`, `VacantEntry`)
- `iter`, `iter_mut`, `into_iter`, `keys`, `values`, `values_mut`
- `drain`, `retain`, `clear`
- `Index` trait
- `FromIterator`, `Extend`
- `Debug`, `Clone`, `Default`, `PartialEq`, `Eq`
- Parameterised hasher: `HashMap<K, V, S = DefaultHasher>`

### 3. Performance targets (informed by Melin's hot paths)

Melin's critical hash-map operations and the latency envelope they live in:

| Operation | Where | Current budget |
|---|---|---|
| `get` / `get_mut` | order cancel, balance lookup, fill processing | 20–50 ns |
| `insert` | new order, deposit | 20–50 ns, no spike > ~5 µs |
| `remove` | order fill/cancel, balance release | 20–50 ns |
| `contains_key` | duplicate-order check | 20–50 ns |

The main improvement over griddle: **deterministic latency under growth**.
Griddle caps resize cost at ~5 µs (p99.99) by spreading it; extendible hashing
should keep per-insert overhead to a single bucket split (~tens of entries),
giving tighter p99.99 without maintaining two live tables.

Typical working-set sizes in Melin (the map must stay cache-friendly at these
scales):

- Per-order-book index: ~4 096 entries (~128 KB, fits L2)
- Global order-info map: ~32 768 entries (~1 MB, fits L3)
- Account maps: up to 1 000 000 entries

### 4. Key types in Melin

These are the concrete key/value shapes the map will be used with. Design
bucket layout and hashing with these in mind:

- `(AccountId, OrderId)` → `(Side, Price)` — 12-byte key, 16-byte value
- `(AccountId, CurrencyId)` → `Balance { available: u64, reserved: u64 }` — 8-byte key, 16-byte value
- `AccountId` → `u64` or `u32` — 4-byte key, 4–8-byte value

All keys are small, `Copy`, and non-cryptographic hashing is fine (FxHash or
similar).

## Build & Run

```sh
cargo build          # compile
cargo test           # run tests
cargo clippy         # lint
cargo fmt            # format
```

## Conventions

- Follow Rust best practices (idiomatic patterns, clippy clean, formatted with `cargo fmt`).
- Write unit tests for all non-trivial code. Skip only when genuinely unreasonable (e.g., trivial glue code).
- **Correctness is critical** — this is a building block for financial infrastructure. Correctness always comes first.
- **Reasonably optimized from the start** — don't prematurely optimize, but make performance-conscious choices by default: minimize allocations, favor cache-friendly data structures. Profile before micro-optimizing.
- **No `.unwrap()` in production code** — use proper error handling. `.unwrap()` is fine in tests.
- **No `#[ignore]` on tests** — if a test fails, fix the bug. Never suppress a failing test with `#[ignore]`.
- **No silently ignored results** — do not use `let _ =` to discard `Result` values unless there is a clear reason. Handle errors explicitly.
- **Comment data structure and type choices** — always add a comment justifying why a specific collection, data structure, or numeric type was chosen.
- **Tail latency matters** — measure p99/p99.9, not averages.
- **Extensive testing** — property-based and fuzz testing for edge cases.

### Git
- **No co-authored commits** — do not add `Co-Authored-By` trailers.
- **Conventional Commits** — all commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) spec (e.g., `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`).
- **Never commit without explicit request** — do NOT commit unless the user explicitly asks. Completing a task does NOT imply permission to commit.
- **Never push without explicit confirmation** — always ask for review before pushing.
- **Commit intermediary steps** — for large multi-step tasks, commit each logical step separately. Always ask for review after each commit before moving to the next.
- **Always check `Cargo.lock`** — when dependencies change, `Cargo.lock` must be staged and committed alongside `Cargo.toml` changes.

## Non-goals (for now)

- Concurrent / lock-free access (Melin is single-threaded per instrument).
- Cryptographic hash resistance.
- `no_std` support.
- Stable iteration order.

## Repo layout

```
astenn/
├── CLAUDE.md          ← you are here
├── Cargo.toml
└── src/
    └── lib.rs
```

This is a library crate. Melin will depend on it via a path dependency
(`astenn = { path = "../astenn" }`).
