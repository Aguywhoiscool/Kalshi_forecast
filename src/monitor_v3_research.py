"""Read-only terminal progress display for the offline v3 research jobs.

Run: python -m src.monitor_v3_research --run-dir artifacts/v3_20261104_overnight
The bar measures completed epochs; the heartbeat animation is not batch progress.
"""
import argparse
import csv
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import time


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def log_tail(path, size=24000):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - size))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def cpu_seconds(pid):
    """Read target process CPU time without changing or signalling it."""
    if os.name != "nt" or not pid:
        return None
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None
    try:
        created, exited, system, user = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                      ctypes.byref(system), ctypes.byref(user)):
            return None
        if exited.dwHighDateTime or exited.dwLowDateTime:
            return None
        return (((system.dwHighDateTime << 32) | system.dwLowDateTime)
                + ((user.dwHighDateTime << 32) | user.dwLowDateTime)) / 10_000_000
    finally:
        kernel.CloseHandle(handle)


def epoch_progress(run, task, log):
    completed = 0
    try:
        with (run / task / "epochs.csv").open(newline="", encoding="utf-8") as stream:
            completed = sum(1 for row in csv.DictReader(stream) if row.get("epoch", "").isdigit())
    except OSError:
        pass
    epochs = re.findall(r"(?:^|\n)Epoch (\d+)/(\d+)", log_tail(log))
    current, maximum = map(int, epochs[-1]) if epochs else (completed + 1, 8)
    return completed, current, maximum


def render(primary, tick):
    state = read_json(primary / "status.json")
    quarter = primary / "learning_curve_25pct"
    quarter_state = read_json(quarter / "status.json")
    primary_log = primary / "run.log"
    if not primary_log.exists() and primary.with_suffix(".log").exists():
        primary_log = primary.with_suffix(".log")
    active, label, log = primary, "100% sampled data", primary_log
    if state.get("stage") == "complete" and quarter_state:
        active, state, label, log = quarter, quarter_state, "25% learning curve", primary / "learning_curve.log"
    stage = state.get("stage", "waiting for status")
    process = read_json(active / "process.json")
    pid = process.get("pid")
    cpu = cpu_seconds(pid)
    heartbeat = "|/-\\"[tick % 4]
    if stage.startswith("training_"):
        task = stage.removeprefix("training_")
        done, current, maximum = epoch_progress(active, task, log)
        width = 24
        filled = min(width, int(width * min(done, maximum) / maximum))
        bar = "#" * filled + "-" * (width - filled)
        message = f"[{bar}] {done}/{maximum} epochs complete | epoch {current} | {task} | {label}"
    else:
        message = f"{stage} | {label}"
    cpu_text = f"CPU {cpu:.1f}s" if cpu is not None else "process unavailable"
    age = ""
    try:
        changed = datetime.fromisoformat(state["utc"])
        elapsed = max(0, int((datetime.now(timezone.utc) - changed).total_seconds()))
        age = f" | phase {elapsed // 60:02d}:{elapsed % 60:02d}"
    except (KeyError, ValueError):
        pass
    message = f"{heartbeat} monitor | " + message + f" | PID {pid or '?'} {cpu_text}{age}"
    finished = stage == "failed" or (state.get("stage") == "complete" and active == quarter)
    if active == primary and stage == "complete" and not quarter_state:
        finished = True
    return message, finished


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="Print one status line and exit")
    args = parser.parse_args()
    print("Epoch bar = completed epochs / configured maximum; early stopping may finish sooner.")
    print("Batches inside the current epoch are unknown. 'monitor' animation is a heartbeat only.")
    print("Ctrl+C stops this display; training keeps running.")
    previous_width = 0
    tick = 0
    try:
        while True:
            message, finished = render(args.run_dir, tick)
            width = max(30, shutil.get_terminal_size((150, 24)).columns - 1)
            message = message[:width]
            print("\r" + message.ljust(previous_width), end="", flush=True)
            previous_width = len(message)
            if args.once or finished:
                print()
                break
            tick += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDisplay stopped. Research jobs were not changed.")


if __name__ == "__main__":
    main()
