mod entry;
mod iter;
mod map;

pub use entry::{Entry, OccupiedEntry, VacantEntry};
pub use iter::{Drain, IntoIter, Iter, IterMut, Keys, Values, ValuesMut};
pub use map::HashMap;

/// Default bucket capacity. 8 entries keeps the hash array at exactly
/// 64 bytes (one cache line). Tune via the const generic `N` parameter
/// on [`HashMap`].
pub const DEFAULT_BUCKET_CAPACITY: usize = 8;
