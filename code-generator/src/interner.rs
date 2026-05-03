use std::ops::Deref;
use parking_lot::{RwLock, RwLockReadGuard};
use string_interner::backend::BufferBackend;
use string_interner::StringInterner;

pub type Symbol = string_interner::symbol::SymbolU32;

pub struct Interner(RwLock<StringInterner<BufferBackend<Symbol>>>);

pub struct InternedStr<T>(T);

impl<T: Deref<Target=str>> InternedStr<T> {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Interner {
    pub fn new() -> Self {
        Self(RwLock::new(StringInterner::new()))
    }
    
    pub fn try_resolve(&self, symbol: Symbol) -> Option<InternedStr<impl Deref<Target=str>>> {
        let reader = self.0.read();
        let lock = RwLockReadGuard::try_map(
            reader,
            |reader| reader.resolve(symbol)
        );
        
        lock.ok().map(InternedStr)
    }
    
    pub fn resolve(&self, symbol: Symbol) -> InternedStr<impl Deref<Target=str>> {
        self.try_resolve(symbol).unwrap()
    }
    
    pub fn get(&self, str: &str) -> Option<Symbol> {
        self.0.read().get(str)
    }
    
    pub fn get_or_intern(&self, str: &str) -> Symbol {
        self.get(str).unwrap_or_else(|| self.0.write().get_or_intern(str))
    }
}