/// A [`MemoryObject`](super::MemoryObject) backed by the operating system's
/// random number generator.
///
/// Every fault-in fills the page with fresh OS-provided randomness, and
/// fault-out discards the page's contents. Mapping this object therefore
/// yields memory that reads as random bytes and never persists writes.
pub struct OsRng;
