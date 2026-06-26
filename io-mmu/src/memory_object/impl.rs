use crate::memory_object::r#impl::io_at::IoAt;
use crate::memory_object::objects;
use anyhow::bail;
use emu_abi::convert::usize_to_u64;
use emu_abi::memory::{PAGE_SIZE, PageNumber};
use std::io;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

mod io_at {
    use std::io;

    // FIXME(https://github.com/rust-lang/rust/issues/140771)
    pub(crate) trait IoAt {
        fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;

        fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize>;
    }

    #[cfg(unix)]
    impl IoAt for std::fs::File {
        fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
            use std::os::unix::fs::FileExt;
            <Self as FileExt>::read_at(self, buf, offset)
        }

        fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
            use std::os::unix::fs::FileExt;
            <Self as FileExt>::write_at(self, buf, offset)
        }
    }

    #[cfg(windows)]
    impl IoAt for std::fs::File {
        fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
            use std::os::windows::fs::FileExt;
            <Self as FileExt>::seek_read(self, buf, offset)
        }

        fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
            use std::os::windows::fs::FileExt;
            <Self as FileExt>::seek_write(self, buf, offset)
        }
    }
}

unsafe impl super::MemoryObject for std::fs::File {
    unsafe fn fault_in_exclusive(
        &self,
        page_offset: PageNumber,
        page_ptr: NonNull<u8>,
    ) -> anyhow::Result<()> {
        // first zero-out page to initialize and also for any trailing unread bytes
        unsafe { std::ptr::write_bytes(page_ptr.as_ptr(), 0, PAGE_SIZE) }

        let page = unsafe { page_ptr.cast::<[u8; PAGE_SIZE]>().as_mut() };

        let mut page_left = &mut page[..];

        let mut offset = page_offset.vaddr_base();
        while !page_left.is_empty() {
            match self.read_at(page_left, offset) {
                Ok(0) => break,
                Ok(n) => {
                    page_left = &mut page_left[n..];

                    // since slicing succeeded, n <= PAGE_SIZE < u64::MAX
                    let n = unsafe { usize_to_u64(n).unwrap_unchecked() };

                    // this is wrapping_add specifically because
                    // if I am reading the last page, then vaddr_base + PAGE_SIZE
                    // will overflow... now the loop is correct because if I read `PAGE_SIZE bytes,
                    // then `page_left` must be zero and the loop terminates.
                    // do note that `PageNumber::MAX.vaddr_base() + PAGE_SIZE`
                    // always overflows into exactly 0
                    offset = offset.wrapping_add(n);
                }
                Err(ref e) if let io::ErrorKind::Interrupted = e.kind() => {}
                Err(e) => return Err(e.into()),
            }
        }

        // only empty pages
        if page_left.len() == PAGE_SIZE {
            bail!(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "read file out of bounds"
            ))
        }

        Ok(())
    }

    unsafe fn fault_out_exclusive(
        &self,
        page_offset: PageNumber,
        page_ptr: NonNull<u8>,
    ) -> anyhow::Result<()> {
        let page = unsafe { page_ptr.cast::<[u8; PAGE_SIZE]>().as_ref() };

        let mut page_left = &page[..];
        let mut offset = page_offset.vaddr_base();
        while !page_left.is_empty() {
            match self.write_at(page_left, offset) {
                Ok(0) => bail!(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write whole buffer",
                )),

                Ok(n) => {
                    page_left = &page_left[n..];

                    // since slicing succeeded, n <= PAGE_SIZE < u64::MAX
                    let n = unsafe { usize_to_u64(n).unwrap_unchecked() };

                    // this is wrapping_add specifically because
                    // if I am reading the last page, then vaddr_base + PAGE_SIZE
                    // will overflow... now the loop is correct because if I read `PAGE_SIZE bytes,
                    // then `page_left` must be zero and the loop terminates.
                    // do note that `PageNumber::MAX.vaddr_base() + PAGE_SIZE`
                    // always overflows into exactly 0
                    offset = offset.wrapping_add(n);
                }
                Err(ref e) if let io::ErrorKind::Interrupted = e.kind() => {}
                Err(e) => return Err(e.into()),
            }
        }

        Ok(())
    }
}

unsafe impl super::MemoryObject for objects::OsRng {
    unsafe fn fault_in_exclusive(&self, _: PageNumber, page: NonNull<u8>) -> anyhow::Result<()> {
        let page = unsafe { page.cast::<[MaybeUninit<u8>; PAGE_SIZE]>().as_mut() };
        getrandom::fill_uninit(page)?;
        Ok(())
    }

    unsafe fn fault_out_exclusive(
        &self,
        page_offset: PageNumber,
        page_ptr: NonNull<u8>,
    ) -> anyhow::Result<()> {
        let _ = (page_offset, page_ptr);
        Ok(())
    }
}
