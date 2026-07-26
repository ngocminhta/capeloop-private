from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import errno
import os
import unittest

import cape_loop.file_lock as file_lock
from cape_loop.file_lock import (
    lock_backend,
    try_file_lock,
    unlock_file,
)


class _FakeMsvcrt:
    LK_NBRLCK = 1
    LK_NBLCK = 2
    LK_UNLCK = 3

    def __init__(self, *, contend: bool = False) -> None:
        self.contend = contend
        self.calls: list[tuple[int, int, int, int]] = []

    def locking(self, descriptor: int, operation: int, count: int) -> None:
        self.calls.append(
            (
                descriptor,
                operation,
                count,
                os.lseek(descriptor, 0, os.SEEK_CUR),
            )
        )
        if self.contend and operation != self.LK_UNLCK:
            raise OSError(errno.EACCES, "simulated lock contention")


class FileLockTests(unittest.TestCase):
    def test_exclusive_lock_rejects_a_second_holder(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "collection.lock"
            first = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            second = os.open(path, os.O_RDWR)
            try:
                self.assertTrue(try_file_lock(first))
                self.assertFalse(try_file_lock(second, shared=True))
                unlock_file(first)
                self.assertTrue(try_file_lock(second, shared=True))
                unlock_file(second)
            finally:
                os.close(first)
                os.close(second)

    def test_shared_locks_exclude_an_exclusive_holder(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "collection.lock"
            first = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            second = os.open(path, os.O_RDWR)
            third = os.open(path, os.O_RDWR)
            try:
                self.assertTrue(try_file_lock(first, shared=True))
                self.assertTrue(try_file_lock(second, shared=True))
                self.assertFalse(try_file_lock(third))
                unlock_file(second)
                unlock_file(first)
                self.assertTrue(try_file_lock(third))
                unlock_file(third)
            finally:
                os.close(first)
                os.close(second)
                os.close(third)

    def test_supported_runtime_has_a_backend(self) -> None:
        self.assertIn(
            lock_backend(),
            {"fcntl-flock", "msvcrt-byte-range"},
        )

    def test_windows_backend_marks_empty_file_and_seeks_before_calls(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "collection.lock"
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            fake = _FakeMsvcrt()
            try:
                with (
                    patch.object(file_lock, "_fcntl", None),
                    patch.object(file_lock, "_msvcrt", fake),
                ):
                    self.assertTrue(
                        file_lock.try_file_lock(
                            descriptor,
                            shared=True,
                        )
                    )
                    file_lock.unlock_file(descriptor)
                self.assertEqual(path.read_bytes(), b"\0")
                self.assertEqual(
                    [call[1:] for call in fake.calls],
                    [
                        (fake.LK_NBRLCK, 1, 0),
                        (fake.LK_UNLCK, 1, 0),
                    ],
                )
            finally:
                os.close(descriptor)

    def test_windows_backend_reports_contention_and_uses_exclusive_mode(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "collection.lock"
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            fake = _FakeMsvcrt(contend=True)
            try:
                with (
                    patch.object(file_lock, "_fcntl", None),
                    patch.object(file_lock, "_msvcrt", fake),
                ):
                    self.assertFalse(file_lock.try_file_lock(descriptor))
                self.assertEqual(
                    [call[1:] for call in fake.calls],
                    [(fake.LK_NBLCK, 1, 0)],
                )
            finally:
                os.close(descriptor)


if __name__ == "__main__":
    unittest.main()
