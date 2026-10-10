use std::fs;
use std::path::Path;

/// Filesystem type of the mount holding `path`.
pub(crate) fn mount_fstype(path: &Path) -> Option<String> {
    let canonical = path.canonicalize().ok()?;
    let mounts = fs::read_to_string("/proc/mounts").ok()?;
    let mut best: Option<(usize, String)> = None;
    for line in mounts.lines() {
        let mut fields = line.split_whitespace();
        let Some(_dev) = fields.next() else {
            continue;
        };
        let Some(mount_point) = fields.next() else {
            continue;
        };
        let Some(fstype) = fields.next() else {
            continue;
        };
        let mount_point = mount_point
            .replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\");
        let mount_path = Path::new(&mount_point);
        if canonical.starts_with(mount_path) {
            let depth = mount_path.components().count();
            let deeper_or_equal = best.as_ref().is_none_or(|(d, _)| depth >= *d);
            if deeper_or_equal {
                best = Some((depth, fstype.to_owned()));
            }
        }
    }
    best.map(|(_, fstype)| fstype)
}
