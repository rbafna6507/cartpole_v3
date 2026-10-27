"""Bounded on-disk diagnostics and camera liveness for unattended viewing."""
from pathlib import Path
import shutil
import threading
import time


class JsonlLog:
    """Rolling diagnostics that never make tracking depend on writable storage.

    Successful writes reach the OS without Python buffering (not fsync). Failed
    writes are counted, not replayed; recording retries after five seconds.
    """
    def __init__(self, path, max_bytes=8 * 1024 * 1024, backups=3,
                 min_free_bytes=256 * 1024 * 1024, retry_seconds=5):
        if max_bytes < 1 or backups < 0 or min_free_bytes < 0 or retry_seconds < 0:
            raise ValueError("Invalid recording limits")
        self.path = Path(path)
        self.max_bytes, self.backups = max_bytes, backups
        self.min_free_bytes, self.retry_seconds = min_free_bytes, retry_seconds
        self.lock = threading.Lock()
        self.file = None
        self.size = 0
        self.error = self.last_error = None
        self.written = self.dropped = self.rotations = 0
        self.first_write_t = self.last_write_t = None
        self.next_retry = self.next_space_check = 0
        self.closed = False
        self.repair_offset = None

    def _close_file(self):
        file, self.file = self.file, None
        if file is not None:
            file.close()

    def _failed(self, error):
        self.error = self.last_error = str(error)
        self.next_retry = time.monotonic() + self.retry_seconds
        try:
            self._close_file()
        except OSError:
            pass

    def write(self, text):
        data = text.encode("utf-8")
        with self.lock:
            now = time.monotonic()
            if self.closed or now < self.next_retry:
                self.dropped += 1
                return False
            try:
                if len(data) > self.max_bytes:
                    raise ValueError("Telemetry record exceeds log limit")
                if now >= self.next_space_check:
                    # Checking once per second bounds overhead on the tracking path.
                    if shutil.disk_usage(self.path.parent).free < self.min_free_bytes:
                        raise OSError("Recording paused: free space below reserve")
                    self.next_space_check = now + 1
                if self.file is None:
                    self.file = self.path.open("a+b", buffering=0)
                    if self.repair_offset is not None:
                        self.file.truncate(self.repair_offset)
                        self.repair_offset = None
                    self.size = self.path.stat().st_size
                if self.size + len(data) > self.max_bytes:
                    self._close_file()
                    for i in range(self.backups, 0, -1):
                        src = self.path if i == 1 else Path(f"{self.path}.{i - 1}")
                        if src.exists():
                            src.replace(Path(f"{self.path}.{i}"))
                    self.file = self.path.open("wb", buffering=0)
                    self.size = 0
                    self.rotations += 1
                # A partial OS write must not leave a broken JSON line before recovery.
                self.repair_offset = self.size
                if self.file.write(data) != len(data):
                    raise OSError("Incomplete telemetry write")
                self.repair_offset = None
                self.size += len(data)
                self.written += 1
                self.first_write_t = self.first_write_t if self.first_write_t is not None else now
                self.last_write_t = now
                self.error = None
                return True
            except (OSError, ValueError) as error:
                self.dropped += 1
                self._failed(error)
                return False

    def flush(self):
        with self.lock:
            try:
                if self.file is not None:
                    self.file.flush()
            except OSError as error:
                self._failed(error)

    def close(self):
        with self.lock:
            self.closed = True
            try:
                self._close_file()
            except OSError as error:
                self._failed(error)

    def report(self):
        with self.lock:
            return dict(error=self.error, last_error=self.last_error,
                        written_records=self.written, dropped_records=self.dropped,
                        first_write_monotonic=self.first_write_t, last_write_monotonic=self.last_write_t,
                        rotations=self.rotations, retention='rolling',
                        max_bytes=self.max_bytes * (self.backups + 1),
                        min_free_bytes=self.min_free_bytes, closed=self.closed)

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def prune_sessions(directory, keep=4):
    """Only directories marked by this runtime are eligible for deletion."""
    sessions = [p for p in Path(directory).iterdir()
                if p.is_dir() and not p.is_symlink() and (p / ".cartpole-vision-session").is_file()]
    for path in sorted(sessions, key=lambda p: p.stat().st_mtime, reverse=True)[keep:]:
        shutil.rmtree(path)


def camera_failure(report, elapsed, timeout):
    if report.get("error"): return f"Camera failed: {report['error']}"
    age = report.get("latest_age_s")
    if age is None and elapsed > timeout: return "Camera produced no frame before timeout"
    if age is not None and age > timeout: return f"Camera stalled for {age:.2f} seconds"
    return None
