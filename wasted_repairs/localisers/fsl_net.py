"""FSL-Net (Barrabes et al., ICML 2025, arXiv:2506.09101), the pretrained feature-shift localisation network.

The model runs in its own environment (scripts/setup_fslnet.sh) through a persistent worker process
(fslnet_worker.py), so it is loaded once per run. Score of feature j = the model's probability that j shifted; the
authors flag a feature above 0.5 (scored as FSL@0.5). Windows shorter than MIN_W rows are not scored. Inputs are cast
to float32; the full reference is passed; the model has no randomness, so the seed is unused."""
import atexit
import os
import select
import struct
import subprocess
import time

import numpy as np

from .. import config

KIND = "score"
MIN_W = 20

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fslnet_worker.py")
_LOAD_TIMEOUT = 300.0
_CALL_TIMEOUT = 300.0
_proc = None
_owner = None
_dead = False


class _WorkerError(RuntimeError):
    pass


def _read_exact(fd, k, timeout):
    buf = bytearray()
    end = time.monotonic() + timeout
    while len(buf) < k:
        left = end - time.monotonic()
        if left <= 0:
            raise _WorkerError("timeout")
        ready, _, _ = select.select([fd], [], [], left)
        if not ready:
            raise _WorkerError("timeout")
        chunk = os.read(fd, k - len(buf))
        if not chunk:
            raise _WorkerError("worker closed its pipe")
        buf.extend(chunk)
    return bytes(buf)


def _stop():
    global _proc
    p, _proc = _proc, None
    if p is None or _owner != os.getpid():
        return
    try:
        p.stdin.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        p.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            p.kill(); p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass


def _start():
    global _proc, _owner
    env = dict(os.environ)
    env.setdefault("FSLNET_THREADS", "1")
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = env["FSLNET_THREADS"]
    env.pop("PYTHONPATH", None); env.pop("PYTHONHOME", None)
    p = subprocess.Popen([config.FSLNET_PYTHON, "-s", _WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, env=env, cwd=os.path.expanduser("~"), bufsize=0, close_fds=True)
    _proc, _owner = p, os.getpid()
    head = _read_exact(p.stdout.fileno(), 8, _LOAD_TIMEOUT)
    if head[:4] != b"FSLR" or struct.unpack("<i", head[4:])[0] != 0:
        raise _WorkerError("worker did not load the model")


def _ensure():
    global _proc
    if _proc is not None and _owner != os.getpid():
        _proc = None
    if _proc is None or _proc.poll() is not None:
        _proc = None
        _start()


def _ask(ref32, win32):
    _ensure()
    n, d = ref32.shape
    w = win32.shape[0]
    msg = b"FSLQ" + struct.pack("<iii", n, w, d) + ref32.tobytes() + win32.tobytes()
    try:
        _proc.stdin.write(msg); _proc.stdin.flush()
    except (BrokenPipeError, OSError, ValueError) as e:
        raise _WorkerError(repr(e))
    fd = _proc.stdout.fileno()
    head = _read_exact(fd, 12, _CALL_TIMEOUT)
    if head[:4] != b"FSLA":
        raise _WorkerError("bad reply")
    status, dd = struct.unpack("<ii", head[4:])
    if dd != d:
        raise _WorkerError("bad reply length")
    out = np.frombuffer(_read_exact(fd, 8 * d, _CALL_TIMEOUT), dtype="<f8").astype(float)
    return status, out


def run(ref, win, seed):
    global _dead
    ref = np.asarray(ref); win = np.asarray(win)
    if win.ndim == 1:
        win = win[None, :]
    d = ref.shape[1]
    nan = np.full(d, np.nan)
    if win.shape[0] < MIN_W or win.shape[1] != d or ref.shape[0] < 2:
        return nan
    if _dead or not (np.isfinite(ref).all() and np.isfinite(win).all()):
        return nan
    ref32 = np.ascontiguousarray(ref, dtype="<f4"); win32 = np.ascontiguousarray(win, dtype="<f4")
    for attempt in (0, 1):
        try:
            status, out = _ask(ref32, win32)
        except _WorkerError:
            _stop()
            if attempt == 1:
                return nan
            continue
        except (OSError, FileNotFoundError):
            _dead = True
            return nan
        if status != 0 or not np.isfinite(out).all():
            return nan
        return out
    return nan


atexit.register(_stop)
