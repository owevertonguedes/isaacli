"""Seccomp-BPF filter for the sandboxed command, assembled here in pure Python.

WHY A HAND-ASSEMBLED FILTER AND NOT A LIBRARY OR A BLOB
-------------------------------------------------------
bwrap takes a compiled BPF program on a file descriptor (`--add-seccomp-fd`),
not a rule list. The usual ways to produce one are `libseccomp` through a
Python binding, or a precompiled blob committed to the repo. Neither is used
here:

- The bindings (`python3-seccomp`, `pyseccomp`) are not installed on this
  machine and are not a dependency this project wants for a filter this small;
  requiring them would mean the sandbox quietly loses a layer wherever they are
  missing.
- A committed blob is a few hundred opaque bytes that nobody can review. A
  filter nobody can read is trusted, not verified, and this is the one file
  where that trade is worst.

So the program is assembled below from the instruction list itself. It is
long-winded but every branch is visible, and `build_filter()` is deterministic:
the same source always produces the same bytes.

WHAT IT DENIES, AND WHY THAT LIST
---------------------------------
The starting point is Flatpak's own default filter (`flatpak-run.c`), which is
what confines every Flatpak application on this machine, minus the entries that
only make sense for a GUI sandbox. The families kept:

- NESTED NAMESPACES AND MOUNTS. This is the one that matters most. A process
  inside the jail has zero capabilities, but a NEW user namespace hands it a
  full capability set inside that namespace, which is the standard first step
  of a container escape and reaches a large amount of kernel attack surface.
  `unshare`, `setns` and `clone(CLONE_NEWUSER)` close that door; the mount
  calls (old and new API) close the door next to it.
- KERNEL AND MODULE CONTROL: module loading, `kexec`, `reboot`, `swapon`,
  `acct`, `quotactl`, `syslog`. All of these would already fail without
  capabilities. They stay on the list because "already impossible" is a
  property of today's kernel, not a promise, and denying them costs nothing.
- KERNEL KEYRING (`add_key`, `keyctl`, `request_key`). The keyring is not
  namespaced the way a filesystem is, and it has a long history of use-after-
  free bugs.
- NUMA AND PAGE MIGRATION (`mbind`, `move_pages`, `migrate_pages`, the
  mempolicy pair) plus `userfaultfd` and `bpf`. No agent command needs them and
  each is a recurring source of privilege-escalation bugs.
- PROCESS INSPECTION: `ptrace`, `perf_event_open`, `process_vm_readv/writev`.

WHAT IT DELIBERATELY DOES NOT DENY
----------------------------------
`clone3` stays allowed. It carries its arguments in a struct behind a pointer,
and seccomp cannot follow a pointer, so a filter cannot tell a thread creation
from a new user namespace through it. Denying it outright would break glibc's
thread and process creation on any libc version that does not fall back to
`clone`, which means breaking `python3` and `pytest`. Denying it with a wrong
errno would break them more subtly. This is a real, acknowledged hole in the
`CLONE_NEWUSER` block: the `clone` path is closed, the `clone3` path is not.
It is documented rather than papered over, and it does not make the filter
worthless, because everything else on the list stays denied on both paths.

Being exact about what that costs: `unshare(CLONE_NEWUSER)` is denied and
`clone(CLONE_NEWUSER)` is denied, but `clone3` with the same flag is not
reachable by this filter, so "no nested user namespaces" is a claim this file
does not make. What it does make is the narrower one: two of the three doors
are shut, and every other family on the list is shut on all paths.

`personality` also stays allowed: Flatpak filters it by argument, and this
filter does no argument inspection except on `clone`'s flags.

ARCHITECTURE
------------
Syscall numbers are per-architecture, and the numbers below are x86_64. The
filter therefore starts by checking `seccomp_data.arch` and killing anything
that is not x86_64, which closes re-entry through the i386 compat ABI, where
the same number means a different call.

That check is NOT enough on its own, and getting this wrong is the easy
mistake: x32 code reports `AUDIT_ARCH_X86_64` too, and distinguishes itself by
setting `X32_SYSCALL_BIT` in the syscall number. An x32 `unshare` would arrive
as `0x40000000 + 272`, match none of the numbers below and fall through to
ALLOW, bypassing the whole deny-list. So the number is range-checked against
that bit before anything else is compared.
`build_filter()` returns None on any other host architecture rather than
applying an x86_64 table to it; the caller is expected to say so out loud, the
same way it does when `systemd-run` is missing. A filter applied to the wrong
number table would deny random syscalls and look like it worked.

HOW TO CHECK THIS FILE IS DOING ANYTHING
----------------------------------------
`tests/check_execution.py` section 10c attempts the denied calls from INSIDE
the sandbox and requires them to fail with EPERM, and runs `python3`, `pytest`,
`git` and `sh -c` to require that they still work. It never reads a refusal
message: a message-only test passes identically whether the filter is loaded or
silently absent.
"""
import platform
import struct

# --- BPF instruction encoding (linux/bpf_common.h) -------------------------
BPF_LD = 0x00
BPF_JMP = 0x05
BPF_RET = 0x06
BPF_W = 0x00
BPF_ABS = 0x20
BPF_JEQ = 0x10
BPF_JGE = 0x30
BPF_K = 0x00
BPF_ALU = 0x04
BPF_AND = 0x50

# --- seccomp constants (linux/seccomp.h, linux/audit.h) --------------------
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
AUDIT_ARCH_X86_64 = 0xC000003E
EPERM = 1

# x32 code reports AUDIT_ARCH_X86_64 like everything else here, and marks its
# calls by setting this bit in the syscall number. So the arch check alone does
# NOT keep x32 out: without an explicit test, `unshare` arrives as
# 0x40000000 + 272, matches none of the numbers below, and falls through to
# ALLOW. That is a bypass of the entire deny-list, so any number carrying the
# bit is killed rather than filtered.
X32_SYSCALL_BIT = 0x40000000

# Offsets into struct seccomp_data: nr (int), arch (u32), instruction_pointer
# (u64), args[6] (u64). Little-endian, so the low half of args[0] is at 16.
OFF_NR = 0
OFF_ARCH = 4
OFF_ARG0_LOW = 16

CLONE_NEWUSER = 0x10000000

# x86_64 syscall numbers, taken from this machine's <sys/syscall.h> rather than
# from memory. Grouped as in the docstring; the comment on each group is the
# reason it is here, and an entry without a reason does not belong.
DENIED_SYSCALLS = {
    # Nested namespaces and mounts: the container-escape path.
    "unshare": 272,
    "setns": 308,
    "mount": 165,
    "umount2": 166,
    "pivot_root": 155,
    "chroot": 161,
    # The newer mount API, which reaches the same place by another door.
    "open_tree": 428,
    "move_mount": 429,
    "fsopen": 430,
    "fsconfig": 431,
    "fsmount": 432,
    "fspick": 433,
    "mount_setattr": 442,
    # Process inspection and profiling.
    "ptrace": 101,
    "perf_event_open": 298,
    "process_vm_readv": 310,
    "process_vm_writev": 311,
    # Kernel and module control.
    "init_module": 175,
    "finit_module": 313,
    "delete_module": 176,
    "kexec_load": 246,
    "kexec_file_load": 320,
    "reboot": 169,
    "swapon": 167,
    "swapoff": 168,
    "syslog": 103,
    "acct": 163,
    "quotactl": 179,
    "uselib": 134,
    "modify_ldt": 154,
    # Kernel keyring.
    "add_key": 248,
    "keyctl": 250,
    "request_key": 249,
    # NUMA, page migration, and two recurring escalation primitives.
    "mbind": 237,
    "get_mempolicy": 239,
    "set_mempolicy": 238,
    "move_pages": 279,
    "migrate_pages": 256,
    "bpf": 321,
    "userfaultfd": 323,
}

# Filtered by argument rather than outright: threads and subprocesses go
# through clone too, so only the CLONE_NEWUSER flag is refused. See the
# docstring on why clone3 cannot get the same treatment.
NR_CLONE = 56

SUPPORTED_MACHINE = "x86_64"


def _stmt(code, k):
    """A non-branching instruction: struct sock_filter{u16 code; u8 jt, jf; u32 k}."""
    return struct.pack("=HBBI", code, 0, 0, k)


def _jump(code, k, jt, jf):
    """A conditional jump. jt/jf count instructions to SKIP, so 0 means fall through."""
    return struct.pack("=HBBI", code, jt, jf, k)


def build_filter():
    """Assemble the BPF program, or return None on a non-x86_64 host.

    None is not an error to swallow: the numbers above are x86_64's, and
    applying them elsewhere would deny whatever those numbers happen to mean
    there. The caller says so in the command output instead of pretending the
    layer is present.
    """
    if platform.machine() != SUPPORTED_MACHINE:
        return None

    program = [
        # Refuse to guess: anything not x86_64 (including the i386 and x32
        # compat ABIs, where these numbers mean other calls) dies here rather
        # than falling through the number checks below.
        _stmt(BPF_LD | BPF_W | BPF_ABS, OFF_ARCH),
        _jump(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        _stmt(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        _stmt(BPF_LD | BPF_W | BPF_ABS, OFF_NR),
        # x32 shares this arch value, so it has to be excluded by syscall
        # number instead. Without this, every deny below is one bit away from
        # being skipped. See X32_SYSCALL_BIT.
        _jump(BPF_JMP | BPF_JGE | BPF_K, X32_SYSCALL_BIT, 0, 1),
        _stmt(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
    ]

    # One pair of instructions per syscall, so every jump offset stays 0 or 1.
    # A single shared "deny" target would be cheaper in instructions and would
    # break the moment the list outgrew the 8-bit jump offset.
    for _name, number in sorted(DENIED_SYSCALLS.items(), key=lambda item: item[1]):
        program.append(_jump(BPF_JMP | BPF_JEQ | BPF_K, number, 0, 1))
        program.append(_stmt(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM))

    # clone last, because reading its flags clobbers the accumulator that holds
    # the syscall number, and nothing may be checked after that.
    program += [
        _jump(BPF_JMP | BPF_JEQ | BPF_K, NR_CLONE, 0, 4),
        _stmt(BPF_LD | BPF_W | BPF_ABS, OFF_ARG0_LOW),
        _stmt(BPF_ALU | BPF_AND | BPF_K, CLONE_NEWUSER),
        _jump(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, 0),
        _stmt(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
        _stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    ]
    return b"".join(program)
