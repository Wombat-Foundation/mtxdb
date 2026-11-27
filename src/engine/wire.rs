//! Shared wire primitives for fixed-width ordered keys and compact integers.
//!
//! This module deliberately contains data-layout code only. Tree traversal,
//! probing, and I/O stay in their owning modules so a macro expansion never
//! hides a state machine from the debugger or the borrow checker.

use crate::storage::StorageError;

/// One fixed-width field whose big-endian encoding sorts like its value.
pub trait OrderedField: Copy + Eq + Ord {
    /// Encoded width in bytes.
    const LEN: usize;

    /// Append this field to `out` in memcomparable order.
    fn encode_into(&self, out: &mut Vec<u8>);

    /// Decode exactly one field from `bytes`.
    ///
    /// # Errors
    /// Returns [`StorageError::Corrupt`] when the slice is not [`Self::LEN`]
    /// bytes.
    fn decode_from(bytes: &[u8]) -> Result<Self, StorageError>;
}

impl OrderedField for i64 {
    const LEN: usize = 8;

    fn encode_into(&self, out: &mut Vec<u8>) {
        let mut bytes = self.to_be_bytes();
        bytes[0] ^= 0x80;
        out.extend_from_slice(&bytes);
    }

    fn decode_from(bytes: &[u8]) -> Result<Self, StorageError> {
        let mut bytes = <[u8; 8]>::try_from(bytes)
            .map_err(|_| StorageError::Corrupt("ordered i64 field length".to_owned()))?;
        bytes[0] ^= 0x80;
        Ok(Self::from_be_bytes(bytes))
    }
}

macro_rules! impl_unsigned_field {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl OrderedField for $ty {
                const LEN: usize = core::mem::size_of::<$ty>();

                fn encode_into(&self, out: &mut Vec<u8>) {
                    out.extend_from_slice(&self.to_be_bytes());
                }

                fn decode_from(bytes: &[u8]) -> Result<Self, StorageError> {
                    <[u8; core::mem::size_of::<$ty>()]>::try_from(bytes)
                        .map(Self::from_be_bytes)
                        .map_err(|_| {
                            StorageError::Corrupt(concat!(
                                "ordered ",
                                stringify!($ty),
                                " field length"
                            )
                            .to_owned())
                        })
                }
            }
        )+
    };
}

impl_unsigned_field!(u64, u32, u16, u8);

impl<const N: usize> OrderedField for [u8; N] {
    const LEN: usize = N;

    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self);
    }

    fn decode_from(bytes: &[u8]) -> Result<Self, StorageError> {
        <[u8; N]>::try_from(bytes)
            .map_err(|_| StorageError::Corrupt("ordered byte-array field length".to_owned()))
    }
}

/// Define a fixed-width ordered key and its [`crate::timeline::OrderedKey`]
/// implementation from a field list.
///
/// The leading field is the scan prefix; the remaining fields are compared in
/// declaration order. Every field must implement [`OrderedField`].
///
/// ```
/// use mtxdb::define_ordered_key;
/// use mtxdb::storage::NodeId;
///
/// define_ordered_key! {
///     pub struct ExampleKey(*b"EXK1") {
///         prefix scope: [u8; 16],
///         sequence: i64,
///         event: NodeId,
///     }
/// }
/// ```
#[macro_export]
macro_rules! define_ordered_key {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident ( $schema:expr ) {
            prefix $prefix:ident : $prefix_ty:ty,
            $( $field:ident : $field_ty:ty ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        $vis struct $name {
            /// Leading scan-prefix field.
            pub $prefix: $prefix_ty,
            $(
                /// Ordered field.
                pub $field: $field_ty,
            )+
        }

        impl $crate::timeline::OrderedKey for $name {
            type Prefix = $prefix_ty;

            const WIRE_LEN: usize = <$prefix_ty as $crate::wire::OrderedField>::LEN
                $( + <$field_ty as $crate::wire::OrderedField>::LEN )+;
            const SCHEMA_TAG: [u8; 4] = $schema;

            fn prefix(&self) -> Self::Prefix {
                self.$prefix
            }

            fn encode_into(&self, out: &mut Vec<u8>) {
                $crate::wire::OrderedField::encode_into(&self.$prefix, out);
                $(
                    $crate::wire::OrderedField::encode_into(&self.$field, out);
                )+
            }

            fn decode(bytes: &[u8]) -> Result<Self, $crate::storage::StorageError> {
                if bytes.len() != Self::WIRE_LEN {
                    return Err($crate::storage::StorageError::Corrupt(
                        "ordered key length".to_owned(),
                    ));
                }
                let mut offset = 0;
                let $prefix = {
                    let end = offset
                        + <$prefix_ty as $crate::wire::OrderedField>::LEN;
                    let value = <$prefix_ty as $crate::wire::OrderedField>::decode_from(
                        &bytes[offset..end],
                    )?;
                    offset = end;
                    value
                };
                $(
                    let $field = {
                        let end = offset
                            + <$field_ty as $crate::wire::OrderedField>::LEN;
                        let value = <$field_ty as $crate::wire::OrderedField>::decode_from(
                            &bytes[offset..end],
                        )?;
                        offset = end;
                        value
                    };
                )+
                let _ = offset;
                Ok(Self {
                    $prefix,
                    $( $field ),+
                })
            }
        }
    };
}

/// Append a prefix-varint encoding of `value` to `out`.
///
/// The first byte's leading-one count gives the total width, so a decoder can
/// reject a truncated or oversized frame without scanning every byte first.
pub fn encode_prefix_varint(value: u64, out: &mut Vec<u8>) {
    if value < (1 << 56) {
        let mut width = 1;
        while width < 8 && value >= (1_u64 << (7 * width)) {
            width += 1;
        }
        let start = out.len();
        out.extend_from_slice(&value.to_be_bytes()[8 - width..]);
        let prefix = if width == 1 {
            0
        } else {
            (((1_u16 << (width - 1)) - 1) as u8) << (9 - width)
        };
        out[start] |= prefix;
        return;
    }
    out.push(0xFF);
    out.extend_from_slice(&value.to_be_bytes());
}

/// Decode one prefix-varint from the front of `bytes`.
///
/// Returns the value and the number of bytes consumed.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] for an empty, truncated, or non-canonical
/// encoding.
pub fn decode_prefix_varint(bytes: &[u8]) -> Result<(u64, usize), StorageError> {
    let first = *bytes
        .first()
        .ok_or_else(|| StorageError::Corrupt("empty prefix varint".to_owned()))?;
    if first == 0xFF {
        let payload = bytes
            .get(1..9)
            .ok_or_else(|| StorageError::Corrupt("truncated prefix varint".to_owned()))?;
        let value = u64::from_be_bytes(
            <[u8; 8]>::try_from(payload)
                .map_err(|_| StorageError::Corrupt("truncated prefix varint".to_owned()))?,
        );
        if value < (1 << 56) {
            return Err(StorageError::Corrupt(
                "non-canonical prefix varint".to_owned(),
            ));
        }
        return Ok((value, 9));
    }

    let width = usize::try_from(first.leading_ones())
        .map_err(|_| StorageError::Corrupt("prefix varint width".to_owned()))?
        + 1;
    let payload = bytes
        .get(..width)
        .ok_or_else(|| StorageError::Corrupt("truncated prefix varint".to_owned()))?;
    let mask = if width == 8 { 0 } else { 0xFF_u8 >> width };
    let mut value = u64::from(payload[0] & mask);
    for &byte in &payload[1..] {
        value = (value << 8) | u64::from(byte);
    }
    if width > 1 && value < (1_u64 << (7 * (width - 1))) {
        return Err(StorageError::Corrupt(
            "non-canonical prefix varint".to_owned(),
        ));
    }
    Ok((value, width))
}

#[cfg(test)]
mod tests {
    use super::{decode_prefix_varint, encode_prefix_varint};

    #[test]
    fn prefix_varints_round_trip_at_every_width() {
        let values = [
            0,
            1,
            127,
            128,
            16_383,
            16_384,
            2_097_151,
            2_097_152,
            (1 << 28) - 1,
            1 << 28,
            (1 << 56) - 1,
            1 << 56,
            u64::MAX,
        ];
        for value in values {
            let mut encoded = Vec::new();
            encode_prefix_varint(value, &mut encoded);
            assert_eq!(
                decode_prefix_varint(&encoded).unwrap(),
                (value, encoded.len()),
                "value {value}"
            );
        }
    }

    #[test]
    fn prefix_varint_rejects_truncated_and_non_canonical_inputs() {
        assert!(decode_prefix_varint(&[]).is_err());
        assert!(decode_prefix_varint(&[0x80]).is_err());
        assert!(decode_prefix_varint(&[0xFF, 0, 0, 0, 0, 0, 0, 0]).is_err());
        assert!(decode_prefix_varint(&[0x80, 0x00]).is_err());
    }
}
