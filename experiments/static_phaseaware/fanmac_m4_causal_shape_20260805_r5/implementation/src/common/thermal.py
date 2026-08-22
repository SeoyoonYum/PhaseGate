"""Thermal / power controls. See RESEARCH.md §6: trap #2 (fanless throttle), trap #4 (power).

NOTE (measured 2026-06): `pmset -g therm` does NOT report CPU_Speed_Limit on Apple Silicon
(that field is Intel-only). The working no-sudo throttle signal is the measurement spread
throttle_ratio() = max/median of a burst. Full GPU clock/temperature traces require
powermetrics (sudo) -> PowerMetricsLogger (opt-in).
"""
import re
import subprocess
import sys

import numpy as np


def power_state():
    """(is_on_ac, raw_line) from `pmset -g batt`."""
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True,
                             timeout=5).stdout
        line = next((l for l in out.splitlines() if "%" in l), out.strip())
        on_ac = ("AC Power" in out) or ("charging" in line.lower()) or ("charged" in line.lower())
        return (on_ac and "discharging" not in line.lower()), line.strip()
    except Exception:
        return True, "pmset unavailable"


def assert_power(allow_battery=False):
    """Trap #4: refuse to run on battery unless explicitly allowed (charger fixed)."""
    ok, line = power_state()
    print(f"[power] {line}")
    if not ok and not allow_battery:
        print("[ABORT] On battery (control #4: charger fixed). Plug in or pass --allow-battery.",
              file=sys.stderr)
        sys.exit(2)
    if not ok:
        print("[WARN] On battery (--allow-battery): numbers are NOT publication-grade.")
    return ok


def cpu_speed_limit():
    """CPU_Speed_Limit from `pmset -g therm` (100=unthrottled). Returns None on Apple
    Silicon (Intel-only field) -> use throttle_ratio() as the throttle signal instead."""
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True,
                             timeout=5).stdout
        m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def throttle_ratio(times):
    """max/median of a measurement burst. > ~1.10 on the fanless M4 flags thermal/clock
    throttling (trap #2's working no-sudo GPU-inclusive signal)."""
    a = np.asarray(times, float)
    med = float(np.median(a))
    return (float(a.max()) / med) if med else float("nan")


def top_mem_procs(n=5):
    try:
        out = subprocess.run(["ps", "-Ao", "rss,comm"], capture_output=True, text=True,
                             timeout=5).stdout.splitlines()[1:]
        rows = []
        for l in out:
            l = l.strip()
            if not l:
                continue
            rss, _, comm = l.partition(" ")
            try:
                rows.append((int(rss), comm.strip()))
            except ValueError:
                pass
        rows.sort(reverse=True)
        return [(c, r / 1024.0) for r, c in rows[:n]]
    except Exception:
        return []


class PowerMetricsLogger:
    """Optional GPU/CPU clock + thermal-pressure trace via `sudo powermetrics`.
    Needs sudo (prompts, or a passwordless sudoers entry) -> opt-in. No-op fallback if
    unavailable; throttle_ratio() remains the no-sudo signal. Primarily for Exp2."""

    def __init__(self, out_path, interval_ms=500, samplers="gpu_power,cpu_power,thermal"):
        self.out_path = out_path
        self.interval_ms = interval_ms
        self.samplers = samplers
        self._proc = None
        self._fh = None

    def start(self):
        cmd = ["sudo", "-n", "powermetrics", "--samplers", self.samplers,
               "-i", str(self.interval_ms)]
        try:
            self._fh = open(self.out_path, "w")
            self._proc = subprocess.Popen(cmd, stdout=self._fh, stderr=subprocess.STDOUT)
            return True
        except Exception as e:
            print(f"[thermal] powermetrics unavailable ({e}); using throttle_ratio() instead")
            self._proc = None
            return False

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        if self._fh is not None:
            self._fh.close()
