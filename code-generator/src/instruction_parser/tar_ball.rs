use flume::TrySendError;
use rayon::iter::plumbing::UnindexedConsumer;
use std::io::{BufRead, Read};
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::{cmp, io};

enum DataPacket {
    FlushBuffer,
    NewData(Vec<u8>),
}

struct CurrentReadBuf {
    buf: Vec<u8>,
    offset: usize,
}

impl CurrentReadBuf {
    pub fn empty() -> Self {
        Self {
            buf: vec![],
            offset: 0,
        }
    }

    pub fn new(buf: Vec<u8>) -> Self {
        Self { buf, offset: 0 }
    }

    pub fn slice(&self) -> &[u8] {
        &self.buf[self.offset..]
    }

    pub fn advance(&mut self, offset: usize) {
        self.offset = self.offset.saturating_add(offset);
        self.offset = cmp::min(self.offset, self.buf.len());
    }
}

pub struct TarFileEntry {
    path: PathBuf,
    data_flow: flume::IntoIter<io::Result<DataPacket>>,
    current_read: CurrentReadBuf,
}

impl TarFileEntry {
    fn create(path: PathBuf, data_flow: flume::Receiver<io::Result<DataPacket>>) -> Self {
        Self {
            path,
            data_flow: data_flow.into_iter(),
            current_read: CurrentReadBuf::empty(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl BufRead for TarFileEntry {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.current_read.slice().is_empty() {
            loop {
                let Some(recv) = self.data_flow.next() else {
                    return Ok(&[]);
                };

                match recv? {
                    DataPacket::NewData(data) => {
                        self.current_read = CurrentReadBuf::new(data);
                        break;
                    }
                    DataPacket::FlushBuffer => {
                        self.current_read = CurrentReadBuf::empty();
                        continue;
                    }
                }
            }
        }

        Ok(self.current_read.slice())
    }

    fn consume(&mut self, amount: usize) {
        self.current_read.advance(amount)
    }
}

impl Read for TarFileEntry {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // avoid stalling an empty buffer if we need to wait for more data
        if buf.is_empty() {
            return Ok(0);
        }

        let mut reader = self.fill_buf()?;
        let read_amount = reader.read(buf)?;
        self.consume(read_amount);
        Ok(read_amount)
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let start_len = buf.len();
        buf.extend_from_slice(self.current_read.slice());
        self.current_read = CurrentReadBuf::empty();

        for packet in self.data_flow.by_ref() {
            let read_buf = match packet? {
                DataPacket::NewData(buffer) => buffer,
                // `current_read` is already empty
                DataPacket::FlushBuffer => continue,
            };

            // Move the unread tail of the current packet into `buf` before draining the
            // remaining packets. This preserves any existing prefix in `buf` while avoiding
            // an extra copy of buffered data.
            //
            // this is buf.capacity() <= read_buf.capacity()
            //         || buf.capacity() <= read_buf.len()
            // but since len is always <= read_buf.capacity()
            // this always holds
            if buf.is_empty() && buf.capacity() <= read_buf.capacity() {
                *buf = read_buf;
                continue;
            }

            buf.extend(read_buf)
        }

        let end_len = buf.len();

        Ok(end_len.strict_sub(start_len))
    }
}

type ReaderSender = flume::Sender<io::Result<DataPacket>>;

struct BudgetCalculator {
    current_budget: usize,
    pending_operations: slab::Slab<(NonZero<usize>, ReaderSender)>,
    total_budget: NonZero<usize>,
}

impl BudgetCalculator {
    pub fn new(total_budget: NonZero<usize>) -> Self {
        Self {
            current_budget: total_budget.get(),
            pending_operations: slab::Slab::new(),
            total_budget,
        }
    }

    fn try_refill_budget_from(
        current_budget: &mut usize,
        sender_budget: NonZero<usize>,
        sender: &ReaderSender,
    ) -> bool {
        match sender.try_send(Ok(DataPacket::FlushBuffer)) {
            Ok(_) | Err(TrySendError::Disconnected(_)) => {
                *current_budget = current_budget.strict_add(sender_budget.get());
                true
            }
            Err(TrySendError::Full(_)) => false,
        }
    }

    fn try_refill_budget(&mut self) {
        self.pending_operations
            .retain(|_, &mut (budget, ref sender)| {
                let sent_flush =
                    Self::try_refill_budget_from(&mut self.current_budget, budget, sender);

                !sent_flush
            })
    }

    #[cold]
    fn refill_budget(
        &mut self,
        sender: Option<(NonZero<usize>, &ReaderSender)>,
    ) -> (bool, NonZero<usize>) {
        let mut selector = flume::Selector::new();
        for (i, &(budget, ref sender)) in &self.pending_operations {
            selector = selector.send(sender, Ok(DataPacket::FlushBuffer), move |_| {
                (Some(i), budget)
            });
        }

        if let Some((budget, sender)) = sender {
            selector = selector.send(sender, Ok(DataPacket::FlushBuffer), move |_| (None, budget));
        }

        assert!(!self.pending_operations.is_empty() || sender.is_some());

        let (selector, budget_reclaimed) = selector.wait();
        if let Some(sender_idx) = selector {
            self.pending_operations.remove(sender_idx);
        }
        self.current_budget = self.current_budget.strict_add(budget_reclaimed.get());
        // try again to refill again
        self.try_refill_budget();

        (
            selector.is_none(),
            NonZero::new(self.current_budget).unwrap(),
        )
    }

    fn get_available_budget_inner(
        &mut self,
        sender: Option<(NonZero<usize>, &ReaderSender)>,
    ) -> (bool, NonZero<usize>) {
        if self.pending_operations.len() > 8 || self.current_budget <= 1024 {
            self.try_refill_budget();
        }

        let mut flushed_active_sender = false;
        if let Some((budget, sender)) = sender {
            flushed_active_sender |=
                Self::try_refill_budget_from(&mut self.current_budget, budget, sender)
        }

        match NonZero::new(self.current_budget) {
            Some(budget) => (flushed_active_sender, budget),
            None => self.refill_budget(sender),
        }
    }

    pub fn get_available_budget(&mut self) -> NonZero<usize> {
        let (flushed_sender, budget) = self.get_available_budget_inner(None);
        assert!(!flushed_sender);
        budget
    }

    pub fn consume(
        &mut self,
        budget: NonZero<usize>,
        sender: &ReaderSender,
    ) -> (bool, NonZero<usize>) {
        self.current_budget = self.current_budget.strict_sub(budget.get());
        self.get_available_budget_inner(Some((budget, sender)))
    }

    pub fn add_budget(&mut self, amt: NonZero<usize>) {
        let new_budget = self.current_budget.strict_add(amt.get());
        assert!(new_budget <= self.total_budget.get());
        self.current_budget = new_budget
    }

    pub fn enqueue(&mut self, consumed: NonZero<usize>, sender: ReaderSender) {
        self.pending_operations.insert((consumed, sender));
    }

    #[cfg(debug_assertions)]
    fn assert_budget_complete(self) {
        let mut budget = self.current_budget;
        for (_i, (in_flight, _sender)) in self.pending_operations {
            budget = budget.strict_add(in_flight.get())
        }
        assert_eq!(budget, self.total_budget.get())
    }
}

#[derive(Debug, Clone)]
pub struct TarFileStream(flume::Receiver<io::Result<TarFileEntry>>);

pub struct TarFileStreamIter(TarFileStream);

impl TarFileStream {
    pub fn size_hint(&self) -> Option<usize> {
        self.0.is_disconnected().then_some(0)
    }
}

impl Iterator for TarFileStreamIter {
    type Item = io::Result<TarFileEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.0.recv().ok()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.0.size_hint())
    }
}

impl IntoIterator for TarFileStream {
    type Item = <TarFileStreamIter as Iterator>::Item;
    type IntoIter = TarFileStreamIter;

    fn into_iter(self) -> Self::IntoIter {
        TarFileStreamIter(self)
    }
}

pub struct ParTarFileStreamIter(TarFileStream);

impl rayon::iter::plumbing::UnindexedProducer for ParTarFileStreamIter {
    type Item = io::Result<TarFileEntry>;

    fn split(self) -> (Self, Option<Self>) {
        let clone = match self.0.0.receiver_count() {
            count if count < rayon::current_num_threads() => Some(Self(self.0.clone())),
            _ => None,
        };

        (self, clone)
    }

    fn fold_with<F: rayon::iter::plumbing::Folder<Self::Item>>(self, folder: F) -> F {
        folder.consume_iter(self.0)
    }
}

impl rayon::iter::ParallelIterator for ParTarFileStreamIter {
    type Item = io::Result<TarFileEntry>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge_unindexed(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        self.0.size_hint()
    }
}

impl TarFileStream {
    pub fn into_par_iter(self) -> ParTarFileStreamIter {
        ParTarFileStreamIter(self)
    }
}

// 32 MiB
const MAX_BUFFERED_FILE_BYTES: usize = 32 * 1024 * 1024;

fn open_archive_inner(reader: Box<dyn Read + Send + 'static>) -> TarFileStream {
    let (tx, rx) = flume::bounded(192);

    let mut budget_calc =
        BudgetCalculator::new(const { NonZero::new(MAX_BUFFERED_FILE_BYTES).unwrap() });

    std::thread::spawn(move || {
        let mut archive = tar::Archive::new(reader);
        let entries = match archive.entries() {
            Ok(entries) => entries,
            Err(err) => {
                let _ = tx.send(Err(err));
                return;
            }
        };

        for entry in entries {
            let result = entry.and_then(|entry| {
                let path = entry.path()?.into_owned();
                Ok((entry, path))
            });

            let (entry, path) = match result {
                Ok(entry) => entry,
                Err(err) => {
                    if tx.send(Err(err)).is_err() {
                        break;
                    }
                    continue;
                }
            };

            let (reader_tx, reader_rx) = flume::bounded(1);

            if tx.send(Ok(TarFileEntry::create(path, reader_rx))).is_err() {
                break;
            }

            let mut entry = entry;

            // `budget_for_send` is the number of bytes we are currently allowed to
            // read ahead for this entry. The shared budget represents bytes that
            // may be read into memory but have not yet been proven drainable by
            // the consumer.
            let mut budget_for_send = budget_calc.get_available_budget();

            // The most recently sent NewData chunk is kept as `in_flight`.
            // A chunk stops being in flight only after a later send proves the
            // receiver has drained the channel slot, or after FlushBuffer is sent
            // successfully on this channel.
            let mut in_flight = None::<NonZero<usize>>;

            let flush_in_flight = {
                |in_flight: &mut Option<NonZero<usize>>, budget_calc: &mut BudgetCalculator| {
                    if let Some(remaining) = *in_flight {
                        budget_calc.add_budget(remaining);
                        *in_flight = None;
                    }
                }
            };

            'reading_loop: loop {
                let mut data = Vec::with_capacity(budget_for_send.get());
                let result = entry
                    .by_ref()
                    .take(u64::try_from(budget_for_send.get()).unwrap())
                    .read_to_end(&mut data)
                    .map(NonZero::new);

                match result {
                    Ok(None) => break 'reading_loop,
                    Ok(Some(res)) => {
                        assert_eq!(res.get(), data.len());
                        if reader_tx.send(Ok(DataPacket::NewData(data))).is_err() {
                            flush_in_flight(&mut in_flight, &mut budget_calc);
                            break 'reading_loop;
                        }

                        // Sending new data succeeded. Since this channel is
                        // bounded to 1, that means any previous in-flight NewData
                        // chunk on this channel must already have been received.
                        // Its bytes are therefore safe to return to the budget.
                        if let Some(flushed) = in_flight.replace(res) {
                            budget_calc.add_budget(flushed)
                        }

                        // Charge the newly sent chunk against the budget, then try
                        // to reclaim it immediately by sending FlushBuffer. If that
                        // succeeds, the consumer has already drained NewData.
                        let (flushed_in_flight, new_budget) = budget_calc.consume(res, &reader_tx);

                        if flushed_in_flight {
                            in_flight = None
                        }

                        budget_for_send = new_budget;
                    }
                    Err(err) => {
                        let _ = reader_tx.send(Err(err));
                        flush_in_flight(&mut in_flight, &mut budget_calc);
                        break 'reading_loop;
                    }
                };
            }

            // If the final chunk was not proven drained during the loop, keep its
            // sender around so a later budget calculation can try to FlushBuffer
            // and reclaim those bytes.
            if let Some(in_flight) = in_flight {
                budget_calc.enqueue(in_flight, reader_tx)
            }
        }

        #[cfg(debug_assertions)]
        {
            budget_calc.assert_budget_complete()
        }
    });

    TarFileStream(rx)
}

fn open_tar_gz_archive_inner(path: &Path) -> io::Result<TarFileStream> {
    let file = std::fs::File::open(path)?;
    let gz_reader = flate2::read::MultiGzDecoder::new(file);
    Ok(open_archive_inner(Box::new(gz_reader)))
}

pub fn open_tar_gz_archive<P: AsRef<Path>>(path: P) -> io::Result<TarFileStream> {
    open_tar_gz_archive_inner(path.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Read};

    fn tar_with_files(files: &[(&str, &[u8])]) -> io::Result<Vec<u8>> {
        let mut builder = tar::Builder::new(Vec::new());

        for &(path, data) in files {
            let mut header = tar::Header::new_gnu();
            header.set_path(path)?;
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();

            builder.append(&header, data)?;
        }

        builder.finish()?;
        builder.into_inner()
    }

    #[test]
    fn empty_read_returns_immediately_even_if_sender_is_alive() {
        let (tx, rx) = flume::bounded(1);
        let mut entry = TarFileEntry::create(PathBuf::from("file.txt"), rx);

        let mut empty = [];
        let result = entry.read(&mut empty);
        assert_eq!(result.unwrap(), 0);
        drop(tx);
    }

    #[test]
    fn read_skips_flush_packets() -> io::Result<()> {
        let (tx, rx) = flume::bounded(8);
        let mut entry = TarFileEntry::create(PathBuf::from("file.txt"), rx);

        tx.send(Ok(DataPacket::FlushBuffer)).unwrap();
        tx.send(Ok(DataPacket::FlushBuffer)).unwrap();
        tx.send(Ok(DataPacket::NewData(b"hello".to_vec()))).unwrap();

        let mut buf = [0; 5];
        let read = entry.read(&mut buf)?;

        assert_eq!(read, 5);
        assert_eq!(buf, *b"hello");

        Ok(())
    }

    #[test]
    fn read_continues_from_current_read_before_receiving_more() -> io::Result<()> {
        let (tx, rx) = flume::bounded(8);
        let mut entry = TarFileEntry::create(PathBuf::from("file.txt"), rx);

        tx.send(Ok(DataPacket::NewData(b"abcdef".to_vec())))
            .unwrap();
        tx.send(Ok(DataPacket::NewData(b"gh".to_vec()))).unwrap();

        let mut first = [0; 2];
        let mut second = [0; 4];
        let mut third = [0; 2];

        assert_eq!(entry.read(&mut first)?, 2);
        assert_eq!(&first, b"ab");

        assert_eq!(entry.read(&mut second)?, 4);
        assert_eq!(&second, b"cdef");

        assert_eq!(entry.read(&mut third)?, 2);
        assert_eq!(&third, b"gh");

        Ok(())
    }

    #[test]
    fn read_to_end_preserves_existing_prefix_and_reads_remaining_packets() -> io::Result<()> {
        let (tx, rx) = flume::bounded(8);
        let mut entry = TarFileEntry::create(PathBuf::from("file.txt"), rx);

        tx.send(Ok(DataPacket::NewData(b"abc".to_vec()))).unwrap();
        tx.send(Ok(DataPacket::FlushBuffer)).unwrap();
        tx.send(Ok(DataPacket::NewData(b"def".to_vec()))).unwrap();
        drop(tx);

        let mut one = [0; 1];
        assert_eq!(entry.read(&mut one)?, 1);
        assert_eq!(&one, b"a");

        let mut out = b"prefix:".to_vec();
        let added = entry.read_to_end(&mut out)?;

        assert_eq!(added, 5);
        assert_eq!(*out, *b"prefix:bcdef");

        Ok(())
    }

    #[test]
    fn read_propagates_errors() {
        let (tx, rx) = flume::bounded(8);
        let mut entry = TarFileEntry::create(PathBuf::from("file.txt"), rx);

        tx.send(Err(io::Error::other("boom"))).unwrap();

        let mut buf = [0; 8];
        let err = entry.read(&mut buf).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert_eq!(err.to_string(), "boom");
    }

    #[test]
    fn read_to_end_propagates_errors_after_existing_data() {
        let (tx, rx) = flume::bounded(8);
        let mut entry = TarFileEntry::create(PathBuf::from("file.txt"), rx);

        tx.send(Ok(DataPacket::NewData(b"abc".to_vec()))).unwrap();
        tx.send(Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "truncated",
        )))
        .unwrap();

        let mut out = Vec::new();
        let err = entry.read_to_end(&mut out).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn open_archive_inner_yields_entries_with_paths_and_contents() -> io::Result<()> {
        let tar = tar_with_files(&[("alpha.txt", b"alpha"), ("nested/beta.txt", b"beta beta")])?;

        let rx = open_archive_inner(Box::new(Cursor::new(tar)));
        let mut entries = rx.into_iter();

        let mut first = entries.next().expect("missing first entry")?;
        assert_eq!(first.path(), Path::new("alpha.txt"));

        let mut first_contents = Vec::new();
        first.read_to_end(&mut first_contents)?;
        assert_eq!(&first_contents, b"alpha");

        let mut second = entries.next().expect("missing second entry")?;
        assert_eq!(second.path(), Path::new("nested/beta.txt"));

        let mut second_contents = Vec::new();
        second.read_to_end(&mut second_contents)?;
        assert_eq!(&second_contents, b"beta beta");

        assert!(entries.next().is_none());

        Ok(())
    }

    #[test]
    fn open_archive_inner_reports_invalid_tar_error() {
        let rx = open_archive_inner(Box::new(Cursor::new(b"not a tar archive".to_vec())));
        let mut entries = rx.into_iter();

        let result = entries.next().expect("expected one error");
        assert!(result.is_err());
    }

    #[test]
    fn read_to_end_can_take_first_packet_when_output_is_empty() -> io::Result<()> {
        let (tx, rx) = flume::bounded(8);
        let mut entry = TarFileEntry::create(PathBuf::from("file.txt"), rx);

        let mut first = Vec::with_capacity(64);
        first.extend_from_slice(b"abc");

        tx.send(Ok(DataPacket::NewData(first))).unwrap();
        tx.send(Ok(DataPacket::NewData(b"def".to_vec()))).unwrap();
        drop(tx);

        let mut out = Vec::new();
        let added = entry.read_to_end(&mut out)?;

        assert_eq!(added, 6);
        assert_eq!(out, b"abcdef");

        Ok(())
    }
}
