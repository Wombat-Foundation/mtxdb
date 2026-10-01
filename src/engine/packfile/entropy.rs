//! Operating-system entropy without a third-party CSPRNG crate.
//!
//! `getrandom` pulls `wasip2` -> `wit-bindgen`, whose manifest requires Cargo's
//! unstabilized `edition2024` feature. Cargo parses every target's dependency
//! graph when it resolves, so merely depending on `getrandom` makes the 1.81
//! MSRV build fail before any mtxdb code compiles. The crate only ever needs
//! 16 bytes to seed a [`PackId`](super::PackId), so the supported platforms are
//! handled directly: `/dev/urandom` on unix and `BCryptGenRandom` on Windows.

use std::io;

/// Fill every byte of `dest` with operating-system cryptographic entropy.
///
/// # Errors
/// Returns the underlying [`io::Error`] if `/dev/urandom` cannot be read.
#[cfg(unix)]
pub(super) fn fill(dest: &mut [u8]) -> io::Result<()> {
    use std::io::Read as _;
    // `/dev/urandom` is the portable unix entropy device: it does not block
    // once the kernel pool is initialized, needs no unsafe FFI, and exists on
    // every unix target mtxdb builds for (Linux, macOS, the BSDs).
    let mut source = std::fs::File::open("/dev/urandom")?;
    source.read_exact(dest)
}

/// Fill every byte of `dest` with operating-system cryptographic entropy.
///
/// # Errors
/// Returns an [`io::Error`] carrying the `BCryptGenRandom` NTSTATUS if the
/// system RNG reports failure.
///
/// # Panics
/// Panics if the destination is longer than a `u32`; callers only ever pass a
/// 16-byte [`PackId`](super::PackId), so this is unreachable in practice.
#[cfg(windows)]
#[allow(unsafe_code)]
pub(super) fn fill(dest: &mut [u8]) -> io::Result<()> {
    use std::ffi::c_void;

    // `bcrypt.dll` has shipped since Windows Vista. Passing a null algorithm
    // handle together with this flag selects the system-preferred RNG, so no
    // algorithm provider has to be opened or freed.
    const BCRYPT_USE_SYSTEM_PREFERRED_RNG: u32 = 0x0000_0002;

    #[link(name = "bcrypt")]
    extern "system" {
        fn BCryptGenRandom(
            algorithm: *mut c_void,
            buffer: *mut u8,
            buffer_len: u32,
            flags: u32,
        ) -> i32;
    }

    // A `PackId` is always 16 bytes, but reject an oversized request rather
    // than silently truncating the ABI length field.
    let buffer_len = u32::try_from(dest.len()).expect("entropy request fits in u32");

    // SAFETY: `dest` is a live, exclusively borrowed slice, so `as_mut_ptr`
    // and `buffer_len` describe a writable buffer of exactly that length. A
    // null algorithm handle plus BCRYPT_USE_SYSTEM_PREFERRED_RNG is the
    // documented "use the system RNG" call, and no pointer is retained after
    // return. This is the only unsafe call on the Windows entropy path.
    let status = unsafe {
        BCryptGenRandom(
            std::ptr::null_mut(),
            dest.as_mut_ptr(),
            buffer_len,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };

    // `BCryptGenRandom` returns an NTSTATUS: >= 0 (STATUS_SUCCESS) on success,
    // negative on failure. Surface the status rather than `GetLastError`,
    // which the NTSTATUS return does not set.
    if status < 0 {
        Err(io::Error::other(format!(
            "BCryptGenRandom failed with NTSTATUS {status:#010x}"
        )))
    } else {
        Ok(())
    }
}

#[cfg(not(any(unix, windows)))]
compile_error!("mtxdb's OS entropy source supports unix and Windows only");
