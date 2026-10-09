use std::path::Path;

use super::bench_cache::mount_fstype;

pub(crate) fn is_ram_backed(path: &Path) -> bool {
    matches!(
        mount_fstype(path).as_deref(),
        Some("tmpfs" | "ramfs" | "devtmpfs" | "hugetlbfs")
    )
}
