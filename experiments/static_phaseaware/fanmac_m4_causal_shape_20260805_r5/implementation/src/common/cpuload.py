"""cpuload.py — controlled CPU memory-bandwidth load for Exp 2 (shared-bus contention).

Compiles and drives `stream_load.c` (native STREAM-style triad, P-core biased).
Intensity is a duty-cycle percent on a SINGLE P-core (one M4 P-core already saturates
~92 GB/s, so duty% is the clean graded knob and one thread leaves the other P-cores for
GPU dispatch). Each run reports the aggregate CPU->DRAM bandwidth it actually achieved
(GB/s) -- the load axis Exp2 plots decode throughput against.
"""
import os
import subprocess

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "stream_load.c")
_BIN = os.path.join(_DIR, "stream_load")


MODES = {"stream": 0, "memcpy": 1, "scan": 2, "random": 3}   # CPU access patterns (P2)


def ensure_built():
    """Compile stream_load.c if the binary is missing or stale. Returns the binary path."""
    if os.path.exists(_BIN) and os.path.getmtime(_BIN) >= os.path.getmtime(_SRC):
        return _BIN
    subprocess.run(["clang", "-O3", "-pthread", _SRC, "-o", _BIN], check=True)
    return _BIN


class CpuBandwidthLoad:
    """A native STREAM-style load at `intensity_pct` duty cycle on `threads` P-core(s).

    Lifecycle: start() launches it (non-blocking), it self-terminates after `seconds`,
    wait() collects the achieved aggregate GB/s. intensity_pct<=0 is a no-op (baseline).

        load = CpuBandwidthLoad(intensity_pct=50, seconds=3.5).start()
        ... measure GPU decode while it runs ...
        gbps = load.wait()
    """

    def __init__(self, intensity_pct, threads=1, mb=96, seconds=3.5, mode="stream"):
        self.intensity = float(intensity_pct)   # duty cycle %, 0 = no load
        self.threads = int(threads)
        self.mb = int(mb)
        self.seconds = float(seconds)
        self.mode = MODES[mode] if isinstance(mode, str) else int(mode)  # access pattern
        self.proc = None
        self.gbps = 0.0
        self.raw = ""

    def start(self):
        if self.intensity <= 0 or self.threads <= 0:
            return self
        ensure_built()
        self.proc = subprocess.Popen(
            [_BIN, f"{self.seconds}", f"{self.threads}", f"{self.mb}", f"{self.intensity}",
             f"{self.mode}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return self

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def wait(self):
        """Block until the load finishes; parse and return aggregate GB/s."""
        if self.proc is None:
            return 0.0
        self.raw, _ = self.proc.communicate()
        for line in (self.raw or "").splitlines():
            if line.startswith("GBPS"):
                self.gbps = float(line.split()[1])
        return self.gbps

    def stop(self):
        if self.running():
            self.proc.terminate()
            try:
                self.raw, _ = self.proc.communicate(timeout=5)
            except Exception:
                self.proc.kill()
                self.raw, _ = self.proc.communicate()
            for line in (self.raw or "").splitlines():
                if line.startswith("GBPS"):
                    self.gbps = float(line.split()[1])
        return self.gbps

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
