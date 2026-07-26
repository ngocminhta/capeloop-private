"""Small cross-platform advisory file-lock primitives.

The research core has no third-party runtime dependencies. POSIX platforms
use ``flock`` and Windows uses a one-byte ``msvcrt`` range lock. Callers own
the file descriptor, must open it read/write so an empty lock file can receive
the Windows marker byte, and are responsible for closing it.
"""

from __future__ import annotations

import errno
import os

try:  # pragma: no cover - exactly one platform branch imports at runtime
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    _fcntl = None

try:  # pragma: no cover - exactly one platform branch imports at runtime
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    _msvcrt = None


_LOCK_CONTENTION_ERRNOS = {
    errno.EACCES,
    errno.EAGAIN,
    getattr(errno, "EDEADLK", errno.EAGAIN),
}
_WINDOWS_LOCK_CONTENTION_CODES = {33, 36}


def lock_backend() -> str:
    """Return the active standard-library lock backend."""

    if _fcntl is not None:
        return "fcntl-flock"
    if _msvcrt is not None:
        return "msvcrt-byte-range"
    return "unavailable"


def _is_contention(error: OSError) -> bool:
    return (
        error.errno in _LOCK_CONTENTION_ERRNOS
        or getattr(error, "winerror", None)
        in _WINDOWS_LOCK_CONTENTION_CODES
    )


def _ensure_lock_byte(descriptor: int) -> None:
    """Give every lock file a portable byte-zero range."""

    if os.fstat(descriptor).st_size == 0:
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, b"\0")
        os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)


def try_file_lock(descriptor: int, *, shared: bool = False) -> bool:
    """Attempt a nonblocking advisory lock.

    Returns ``False`` only for ordinary lock contention. Other operating-system
    errors are propagated so callers do not mistake an unusable lock file for
    a busy collection.
    """

    _ensure_lock_byte(descriptor)
    if _fcntl is not None:
        operation = _fcntl.LOCK_SH if shared else _fcntl.LOCK_EX
        try:
            _fcntl.flock(descriptor, operation | _fcntl.LOCK_NB)
        except OSError as exc:
            if _is_contention(exc):
                return False
            raise
        return True
    if _msvcrt is not None:  # pragma: no cover - exercised on Windows
        operation = (
            _msvcrt.LK_NBRLCK if shared else _msvcrt.LK_NBLCK
        )
        try:
            _msvcrt.locking(descriptor, operation, 1)
        except OSError as exc:
            if _is_contention(exc):
                return False
            raise
        return True
    raise RuntimeError(
        "this Python runtime provides no supported advisory file-lock backend"
    )


def unlock_file(descriptor: int) -> None:
    """Release a lock acquired by :func:`try_file_lock`."""

    if _fcntl is not None:
        _fcntl.flock(descriptor, _fcntl.LOCK_UN)
        return
    if _msvcrt is not None:  # pragma: no cover - exercised on Windows
        os.lseek(descriptor, 0, os.SEEK_SET)
        _msvcrt.locking(descriptor, _msvcrt.LK_UNLCK, 1)
        return
    raise RuntimeError(
        "this Python runtime provides no supported advisory file-lock backend"
    )
