//! Small, platform-specific I/O primitives shared by the append paths.
//!
//! The engine deliberately uses buffered, ordinary file I/O rather than
//! `O_DIRECT`: reads are served from `mmap`, and writes are sequential appends
//! whose cost is dominated by the filesystem and device, not by the page cache.
//! This module keeps the platform differences in one place:
//!
//! - positioned writes (`pwrite` on Unix, `seek_write` on Windows);
//! - vectored writes for a WAL group, so Unix does not copy the payload into a
//!   second contiguous frame buffer;
//! - best-effort block reservation (`fallocate(FALLOC_FL_KEEP_SIZE)`) so an
//!   append does not pay allocation latency and fragmentation on Linux;
//! - best-effort sequential-read hints for pack scans.
//!
//! Every hint is advisory. A failure never changes the on-disk format or
//! correctness: the caller falls back to the portable buffered path.

use std::fs::File;
use std::io;
#[cfg(not(any(unix, windows)))]
use std::io::Seek as _;
#[cfg(any(unix, not(any(unix, windows))))]
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::FileExt as _;
#[cfg(windows)]
use std::os::windows::fs::FileExt as _;

/// Write all of `buf` at `offset`, without changing the caller-visible file
/// cursor on Unix.
///
/// On Windows `seek_write` is used instead of seek-then-write, which removes
/// the per-flush seek and the need to clone the handle merely to get a mutable
/// `File`.
///
/// # Errors
/// Returns the first I/O error, or [`io::ErrorKind::WriteZero`] if the
/// underlying write accepts no bytes.
#[allow(unused_mut)]
pub(crate) fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        file.write_all_at(buf, offset)
    }
    #[cfg(windows)]
    {
        while !buf.is_empty() {
            match file.seek_write(buf, offset) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "positioned write accepted no bytes",
                    ));
                }
                Ok(written) => {
                    buf = &buf[written..];
                    offset = offset.saturating_add(u64::try_from(written).unwrap_or(u64::MAX));
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        // The portable fallback has to move a cursor, so use a clone. This is
        // the only place that pays for one.
        let mut clone = file.try_clone()?;
        clone.seek(io::SeekFrom::Start(offset))?;
        let _ = offset;
        while !buf.is_empty() {
            match clone.write(buf) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "positioned write accepted no bytes",
                    ));
                }
                Ok(written) => buf = &buf[written..],
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}

/// Write a set of buffers with one system call where the platform supports it.
///
/// Unix uses `writev` through [`Write::write_vectored`]. Windows uses its
/// ordinary write path for each slice, so callers that care about avoiding the
/// extra copies should keep using a single contiguous buffer there. The helper
/// handles short writes and `EINTR` in both cases.
///
/// # Errors
/// Returns the first I/O error, or [`io::ErrorKind::WriteZero`] if no progress
/// is possible.
#[cfg(unix)]
pub(crate) fn write_all_vectored(file: &mut File, bufs: &[io::IoSlice<'_>]) -> io::Result<()> {
    let mut bufs = bufs;
    while !bufs.is_empty() {
        match file.write_vectored(bufs) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "vectored write accepted no bytes",
                ));
            }
            Ok(mut written) => {
                while written > 0 {
                    if written >= bufs[0].len() {
                        written = written.saturating_sub(bufs[0].len());
                        bufs = &bufs[1..];
                    } else {
                        // A partial slice can only be advanced by re-slicing.
                        // `IoSlice` does not expose that, so fall back to one
                        // extra write for the remaining bytes.
                        let remaining = &bufs[0][written..];
                        file.write_all(remaining)?;
                        written = 0;
                        bufs = &bufs[1..];
                    }
                }
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// Reserve blocks for an append without changing the file's logical length.
///
/// Linux uses `fallocate(FALLOC_FL_KEEP_SIZE)`. This is a best-effort latency
/// optimization: unsupported filesystems return an error that callers ignore,
/// and the file's size (and therefore the pack scanner) is unchanged. Disabled
/// by default; set `MTXDB_PREALLOC=1` to enable it on a measured workload.
///
/// # Errors
/// Returns the platform error; callers treat it as a hint failure.
#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
pub(crate) fn reserve_append_space(file: &File, offset: u64, len: u64) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;

    // `FALLOC_FL_KEEP_SIZE` from <linux/falloc.h>.
    const FALLOC_FL_KEEP_SIZE: i32 = 0x01;
    extern "C" {
        fn fallocate(fd: i32, mode: i32, offset: i64, len: i64) -> i32;
    }

    // Reservation is opt-in: it slowed concurrent pack writers on the measured
    // HDD workload. Read once:
    // every flush would otherwise pay for an environment lookup.
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("MTXDB_PREALLOC").as_deref() == Ok("1")) {
        return Ok(());
    }

    let offset = i64::try_from(offset).unwrap_or(i64::MAX);
    let len = i64::try_from(len).unwrap_or(i64::MAX);
    if len <= 0 {
        return Ok(());
    }
    // SAFETY: `file` owns a valid descriptor for the call, and the syscall only
    // reserves blocks for a range; it cannot change the file's logical size
    // because `FALLOC_FL_KEEP_SIZE` is set.
    let result = unsafe { fallocate(file.as_raw_fd(), FALLOC_FL_KEEP_SIZE, offset, len) };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Best-effort block reservation on platforms without a suitable primitive.
#[cfg(not(target_os = "linux"))]
#[allow(
    clippy::unnecessary_wraps,
    reason = "the stub keeps the same fallible signature as the Linux implementation"
)]
pub(crate) fn reserve_append_space(_file: &File, _offset: u64, _len: u64) -> io::Result<()> {
    Ok(())
}

/// Ask the kernel to read a pack file sequentially.
///
/// On Linux this is `posix_fadvise(POSIX_FADV_SEQUENTIAL)`; on Windows the
/// caller should open the file with `FILE_FLAG_SEQUENTIAL_SCAN` instead. Both
/// are hints and failures are ignored.
pub(crate) fn advise_sequential(file: &File) {
    #[cfg(target_os = "linux")]
    #[allow(unsafe_code)]
    {
        use std::os::unix::io::AsRawFd;
        extern "C" {
            fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
        }
        const POSIX_FADV_SEQUENTIAL: i32 = 2;
        // SAFETY: the descriptor is valid for the duration of the call; the
        // function is an advisory hint and does not affect correctness.
        let _ = unsafe { posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_SEQUENTIAL) };
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = file;
    }
}

/// Open a file for pack scanning with the platform's sequential-read hint.
///
/// # Errors
/// Propagates the open error.
pub(crate) fn open_read_sequential(path: &std::path::Path) -> io::Result<File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_SEQUENTIAL_SCAN
        options.custom_flags(0x0800_0000);
    }
    let file = options.open(path)?;
    advise_sequential(&file);
    Ok(file)
}

/// Path to re-execute this test binary.
///
/// Coverage builds (and concurrent rebuilds) relink the on-disk test binary
/// into the same target directory, so the path returned by `current_exe()` can
/// be replaced or briefly absent when the child is spawned. On Linux
/// `/proc/self/exe` names the running image itself, which survives that.
#[cfg(test)]
pub(crate) fn test_binary_path() -> std::path::PathBuf {
    let running = std::path::Path::new("/proc/self/exe");
    if running.exists() {
        return running.to_path_buf();
    }
    std::env::current_exe().unwrap()
}
