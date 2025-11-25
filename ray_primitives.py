import os
import time
from typing import Optional, Dict, Any
import ray


def pin_to_core(core_id: Optional[int] = None) -> None:
    """
    Pin current process to a single CPU core if possible.
    Supports Linux via sched_setaffinity. On macOS, CPU affinity is not supported
    by the standard library and this function will log a warning.
    
    Useful on Raspberry Pi 5 to isolate Minion, EnvRunner, and Learner on separate cores.
    
    Args:
        core_id: CPU core ID to pin to (0-indexed). If None, no pinning is performed.
    """
    if core_id is None:
        return
    
    core_id = int(core_id)
    import logging
    logger = logging.getLogger('MyRLApp.pin_to_core')
    
    # Linux: use sched_setaffinity
    if hasattr(os, 'sched_setaffinity'):
        try:
            os.sched_setaffinity(0, {core_id})
            logger.info(f"Successfully pinned process {os.getpid()} to CPU core {core_id}")
            return
        except Exception as e:
            logger.warning(f"Failed to pin process to core {core_id}: {e}")
            return
    
    # macOS: CPU affinity not supported via standard Python APIs
    # The macOS kernel does support thread affinity, but it requires complex
    # Mach thread APIs that are not easily accessible from Python.
    # For macOS, consider using external tools like 'taskset' or 'cpuset' if needed.
    if os.uname().sysname == 'Darwin':
        logger.warning(
            f"CPU pinning requested for core {core_id}, but macOS does not support "
            f"process-level CPU affinity via Python's standard library. "
            f"On macOS, the OS scheduler will handle CPU assignment automatically. "
            f"For true CPU pinning on macOS, you would need to use external tools or "
            f"implement Mach thread APIs (not recommended)."
        )
        return
    
    # Other platforms
    logger.warning(f"CPU pinning not supported on platform {os.uname().sysname}")


@ray.remote
class RingBufferActor:
    """
    Lock-free, overwrite-on-full ring buffer that stores Ray ObjectRefs.
    Designed for one or more producers (Minion(s)) and a single consumer (EnvRunner),
    but works with multiple consumers if each consumer drains in-order.
    """

    def __init__(self, capacity: int, cpu_core: Optional[int] = None):
        pin_to_core(cpu_core)
        assert capacity >= 2, "capacity must be >= 2"
        self.capacity = int(capacity)
        self.buf = [None] * self.capacity
        self.read_idx = 0
        self.write_idx = 0
        self.full = False
        self._puts = 0
        self._drops = 0

    # --- helpers ---
    def _size(self) -> int:
        if self.full:
            return self.capacity
        if self.write_idx >= self.read_idx:
            return self.write_idx - self.read_idx
        return self.capacity - (self.read_idx - self.write_idx)

    def empty(self) -> bool:
        return (not self.full) and (self.read_idx == self.write_idx)

    def size(self) -> int:
        return self._size()

    def stats(self) -> Dict[str, Any]:
        return {
            "size": self._size(),
            "capacity": self.capacity,
            "read_idx": self.read_idx,
            "write_idx": self.write_idx,
            "full": self.full,
            "puts": self._puts,
            "drops": self._drops,
        }

    # --- API ---
    def put(self, ref) -> bool:
        """
        Insert an ObjectRef. If the buffer is full, drop the oldest (overwrite).
        Returns True if an old element was dropped (overwritten).
        """
        dropped = False
        self.buf[self.write_idx] = ref
        self.write_idx = (self.write_idx + 1) % self.capacity
        self._puts += 1

        if self.full:
            # Overwrite oldest; move read pointer forward.
            self.read_idx = self.write_idx
            self._drops += 1
            dropped = True
        elif self.write_idx == self.read_idx:
            self.full = True
        return dropped

    def get(self, timeout_ms: Optional[int] = None):
        """
        Pop the next ObjectRef in FIFO order.
        If empty, block up to timeout_ms, else return None.
        """
        if timeout_ms is None:
            timeout_ms = 0
        if timeout_ms < 0:
            timeout_ms = 0

        # quick path
        if self.empty():
            if timeout_ms == 0:
                return None
            # Spin-wait with short sleep to reduce CPU
            deadline = time.time() + (timeout_ms / 1000.0)
            while self.empty() and time.time() < deadline:
                time.sleep(0.001)  # 1ms
            if self.empty():
                return None

        ref = self.buf[self.read_idx]
        self.full = False
        self.read_idx = (self.read_idx + 1) % self.capacity
        return ref

    def clear(self) -> None:
        self.buf = [None] * self.capacity
        self.read_idx = 0
        self.write_idx = 0
        self.full = False
        self._puts = 0
        self._drops = 0


@ray.remote
class WeightsHubActor:
    """
    Lightweight weights broadcaster.
    Learner calls set(weights_np_dict) after each update to publish a new ObjectRef.
    Consumers call get() to retrieve (ref, version).
    """

    def __init__(self, cpu_core: Optional[int] = None):
        pin_to_core(cpu_core)
        self._ref = None
        self._version = -1

    def set(self, weights_np_dict) -> int:
        # Store as a single Ray object for zero-copy on-node broadcast
        self._ref = ray.put(weights_np_dict)
        self._version += 1
        return self._version

    def get(self):
        # Returns (ObjectRef, version)
        return self._ref, self._version

    def version(self) -> int:
        return self._version