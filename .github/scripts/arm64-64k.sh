#!/usr/bin/env bash
# Full-system guest: a real 64 KiB Linux kernel, even without nested KVM.
set -euo pipefail
work="$(mktemp -d "${RUNNER_TEMP:-/tmp}/mtxdb-arm64-XXXXXX")"
vm_pid=''
cleanup() {
    if [[ -n "$vm_pid" ]]; then kill "$vm_pid" 2>/dev/null || true; fi
    if [[ -f "$work/serial.log" ]]; then cp "$work/serial.log" arm64-64k-serial.log; fi
}
trap cleanup EXIT
cp /usr/share/AAVMF/AAVMF_VARS.fd "$work/vars.fd"
ssh-keygen -q -t ed25519 -N '' -f "$work/key"
cat > "$work/user-data" <<CLOUD
#cloud-config
users:
  - name: tester
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $(cat "$work/key.pub")
ssh_pwauth: false
CLOUD
printf 'instance-id: mtxdb-64k\nlocal-hostname: mtxdb-64k\n' > "$work/meta-data"
cloud-localds "$work/seed.img" "$work/user-data" "$work/meta-data"
curl --fail --location --retry 3 -o "$work/base.img" \
    https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-arm64.img
qemu-img create -f qcow2 -F qcow2 -b "$work/base.img" "$work/disk.img" 24G
ssh_args=(-i "$work/key" -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5)
remote() { ssh "${ssh_args[@]}" tester@127.0.0.1 "$@"; }
boot_with_accel() {
    local accel="$1" cpu="$2"
    shift 2
    printf 'QEMU acceleration: -accel %s -cpu %s\n' "$accel" "$cpu"
    qemu-system-aarch64 -machine virt -accel "$accel" -cpu "$cpu" -smp 4 -m 4096 \
        -drive "if=none,file=$work/disk.img,format=qcow2,id=root" \
        -device virtio-blk-pci,drive=root \
        -drive "if=none,file=$work/seed.img,format=raw,id=seed" \
        -device virtio-blk-pci,drive=seed \
        -netdev user,id=net,hostfwd=tcp:127.0.0.1:2222-:22 \
        -device virtio-net-pci,netdev=net -nographic "$@" \
        > "$work/serial.log" 2>&1 &
    vm_pid=$!
    for attempt in {1..180}; do
        if remote true 2>/dev/null; then return; fi
        if ! kill -0 "$vm_pid" 2>/dev/null; then break; fi
        sleep 5
    done
    cat "$work/serial.log"
    return 1
}

boot() {
    # KVM requires nested virtualization exposed by the runner provider. A
    # present but unusable /dev/kvm must still fall back to TCG.
    if [[ -r /dev/kvm && -w /dev/kvm ]]; then
        if boot_with_accel kvm host "$@"; then return; fi
        echo 'KVM boot failed; retrying with TCG' >&2
        if [[ -n "$vm_pid" ]]; then
            kill "$vm_pid" 2>/dev/null || true
            wait "$vm_pid" 2>/dev/null || true
            vm_pid=''
        fi
    fi
    boot_with_accel tcg max "$@"
}
boot -drive "if=pflash,format=raw,readonly=on,file=/usr/share/AAVMF/AAVMF_CODE.fd" \
    -drive "if=pflash,format=raw,file=$work/vars.fd"
remote 'set -e; sudo cloud-init status --wait; sudo apt-get update; sudo apt-get install -y linux-image-generic-64k build-essential curl ca-certificates pkg-config libzstd-dev'
remote 'sudo cp /boot/vmlinuz-*-64k /home/tester/kernel; sudo cp /boot/initrd.img-*-64k /home/tester/initrd; sudo chown tester:tester /home/tester/kernel /home/tester/initrd'
scp -i "$work/key" -P 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null tester@127.0.0.1:kernel "$work/kernel"
scp -i "$work/key" -P 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null tester@127.0.0.1:initrd "$work/initrd"
remote 'sudo poweroff' || true
shutdown_deadline=$((SECONDS + 60))
while kill -0 "$vm_pid" 2>/dev/null && (( SECONDS < shutdown_deadline )); do
    sleep 1
done
if kill -0 "$vm_pid" 2>/dev/null; then
    echo "QEMU did not exit after guest poweroff; stopping it" >&2
    kill "$vm_pid" 2>/dev/null || true
fi
wait "$vm_pid" || true
vm_pid=''
boot -kernel "$work/kernel" -initrd "$work/initrd" \
    -append 'root=LABEL=cloudimg-rootfs rw console=ttyAMA0'
remote 'set -eu; uname -a; test "$(getconf PAGESIZE)" = 65536; getconf PAGESIZE'
# Transfer only tracked source, including the new workflow scripts, without .git or target/.
git archive HEAD | remote 'mkdir -p source; tar -x -C source'
remote 'bash -se' <<'GUEST'
set -euo pipefail
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
export PATH="$HOME/.cargo/bin:$PATH" RUSTUP_TOOLCHAIN=stable RUSTC_WRAPPER=''
cd source
rustc -vV
cargo test --locked --all-features --lib --tests
cargo test --locked --no-default-features --features multi-reader,zstd --lib --tests
GUEST
