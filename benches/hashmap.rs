//! Microbenchmarks comparing astenn's extendible-hashing HashMap against
//! std's HashMap. Uses Melin's key/value shapes and FxHash where applicable.
//!
//! Run with: `cargo bench`

use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rustc_hash::FxBuildHasher;

// ---------------------------------------------------------------------------
// Key/value types matching Melin's hot-path maps
// ---------------------------------------------------------------------------

/// (AccountId, OrderId) — 12-byte key (padded to 16 with alignment).
type Key = (u32, u64);
/// (Side, Price) — simulated 16-byte value.
type Value = (u8, u64);

/// Build a pre-populated astenn map with `n` entries and FxHash.
fn astenn_map(n: usize) -> astenn::HashMap<Key, Value, FxBuildHasher> {
    let mut map = astenn::HashMap::with_capacity_and_hasher(n, FxBuildHasher);
    for i in 0..n as u64 {
        map.insert((i as u32, i), (i as u8, i * 100));
    }
    map
}

/// Build a pre-populated std map with `n` entries and FxHash.
fn std_map(n: usize) -> std::collections::HashMap<Key, Value, FxBuildHasher> {
    let mut map = std::collections::HashMap::with_capacity_and_hasher(n, FxBuildHasher);
    for i in 0..n as u64 {
        map.insert((i as u32, i), (i as u8, i * 100));
    }
    map
}

// ---------------------------------------------------------------------------
// Map sizes matching Melin's working sets
// ---------------------------------------------------------------------------

/// Per-order-book index (~4K entries, fits L2).
const SIZE_BOOK: usize = 4_096;
/// Global order-info map (~32K entries, fits L3).
const SIZE_GLOBAL: usize = 32_768;

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_get_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_hit");

    for &size in &[SIZE_BOOK, SIZE_GLOBAL] {
        let astenn = astenn_map(size);
        let std = std_map(size);

        group.bench_with_input(BenchmarkId::new("astenn", size), &size, |b, &n| {
            let mut i = 0u64;
            b.iter(|| {
                let key = (i as u32, i);
                let v = astenn.get(black_box(&key));
                black_box(v);
                i = (i + 1) % n as u64;
            });
        });

        group.bench_with_input(BenchmarkId::new("std", size), &size, |b, &n| {
            let mut i = 0u64;
            b.iter(|| {
                let key = (i as u32, i);
                let v = std.get(black_box(&key));
                black_box(v);
                i = (i + 1) % n as u64;
            });
        });
    }
    group.finish();
}

fn bench_get_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_miss");

    for &size in &[SIZE_BOOK, SIZE_GLOBAL] {
        let astenn = astenn_map(size);
        let std = std_map(size);
        // Keys that don't exist: offset by `size`.
        let offset = size as u64;

        group.bench_with_input(BenchmarkId::new("astenn", size), &size, |b, &n| {
            let mut i = 0u64;
            b.iter(|| {
                let key = ((offset + i) as u32, offset + i);
                let v = astenn.get(black_box(&key));
                black_box(v);
                i = (i + 1) % n as u64;
            });
        });

        group.bench_with_input(BenchmarkId::new("std", size), &size, |b, &n| {
            let mut i = 0u64;
            b.iter(|| {
                let key = ((offset + i) as u32, offset + i);
                let v = std.get(black_box(&key));
                black_box(v);
                i = (i + 1) % n as u64;
            });
        });
    }
    group.finish();
}

fn bench_insert_new(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_new");

    for &size in &[SIZE_BOOK, SIZE_GLOBAL] {
        group.bench_with_input(BenchmarkId::new("astenn", size), &size, |b, &n| {
            b.iter_batched(
                || astenn::HashMap::with_capacity_and_hasher(n, FxBuildHasher),
                |mut map| {
                    for i in 0..n as u64 {
                        map.insert(black_box((i as u32, i)), (i as u8, i * 100));
                    }
                    map
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("std", size), &size, |b, &n| {
            b.iter_batched(
                || std::collections::HashMap::with_capacity_and_hasher(n, FxBuildHasher),
                |mut map| {
                    for i in 0..n as u64 {
                        map.insert(black_box((i as u32, i)), (i as u8, i * 100));
                    }
                    map
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_insert_overwrite(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_overwrite");

    for &size in &[SIZE_BOOK, SIZE_GLOBAL] {
        let mut astenn = astenn_map(size);
        let mut std = std_map(size);

        group.bench_with_input(BenchmarkId::new("astenn", size), &size, |b, &n| {
            let mut i = 0u64;
            b.iter(|| {
                let key = (i as u32, i);
                astenn.insert(black_box(key), (i as u8, i));
                i = (i + 1) % n as u64;
            });
        });

        group.bench_with_input(BenchmarkId::new("std", size), &size, |b, &n| {
            let mut i = 0u64;
            b.iter(|| {
                let key = (i as u32, i);
                std.insert(black_box(key), (i as u8, i));
                i = (i + 1) % n as u64;
            });
        });
    }
    group.finish();
}

fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove");

    for &size in &[SIZE_BOOK, SIZE_GLOBAL] {
        group.bench_with_input(BenchmarkId::new("astenn", size), &size, |b, &n| {
            b.iter_batched(
                || astenn_map(n),
                |mut map| {
                    for i in 0..n as u64 {
                        map.remove(black_box(&(i as u32, i)));
                    }
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("std", size), &size, |b, &n| {
            b.iter_batched(
                || std_map(n),
                |mut map| {
                    for i in 0..n as u64 {
                        map.remove(black_box(&(i as u32, i)));
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_contains_key(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains_key");

    for &size in &[SIZE_BOOK, SIZE_GLOBAL] {
        let astenn = astenn_map(size);
        let std = std_map(size);

        group.bench_with_input(BenchmarkId::new("astenn", size), &size, |b, &n| {
            let mut i = 0u64;
            b.iter(|| {
                let key = (i as u32, i);
                let v = astenn.contains_key(black_box(&key));
                black_box(v);
                i = (i + 1) % n as u64;
            });
        });

        group.bench_with_input(BenchmarkId::new("std", size), &size, |b, &n| {
            let mut i = 0u64;
            b.iter(|| {
                let key = (i as u32, i);
                let v = std.contains_key(black_box(&key));
                black_box(v);
                i = (i + 1) % n as u64;
            });
        });
    }
    group.finish();
}

fn bench_entry(c: &mut Criterion) {
    let mut group = c.benchmark_group("entry_or_insert");

    for &size in &[SIZE_BOOK, SIZE_GLOBAL] {
        let mut astenn = astenn_map(size);
        let mut std = std_map(size);

        // Entry on existing keys (occupied path).
        group.bench_with_input(BenchmarkId::new("astenn", size), &size, |b, &n| {
            let mut i = 0u64;
            b.iter(|| {
                let key = (i as u32, i);
                astenn.entry(black_box(key)).or_insert((0, 0));
                i = (i + 1) % n as u64;
            });
        });

        group.bench_with_input(BenchmarkId::new("std", size), &size, |b, &n| {
            let mut i = 0u64;
            b.iter(|| {
                let key = (i as u32, i);
                std.entry(black_box(key)).or_insert((0, 0));
                i = (i + 1) % n as u64;
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_get_hit,
    bench_get_miss,
    bench_insert_new,
    bench_insert_overwrite,
    bench_remove,
    bench_contains_key,
    bench_entry,
);
criterion_main!(benches);
