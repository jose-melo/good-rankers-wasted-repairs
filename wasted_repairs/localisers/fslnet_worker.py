"""FSL-Net worker, run with the FSL-Net environment's Python (see scripts/setup_fslnet.sh).

Loads the authors' pretrained model once (FSLNet.from_pretrained) and answers requests on a binary pipe:
  worker -> parent, once:   b"FSLR" + int32 status (0 = model loaded)
  parent -> worker:         b"FSLQ" + int32 n, w, d + n*d float32 (reference) + w*d float32 (window)
  worker -> parent:         b"FSLA" + int32 status + int32 d + d float64 (per-feature shift probabilities)
"""
import os
import struct
import sys

_out = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)
_in = os.fdopen(os.dup(0), "rb", buffering=0)

_threads = int(os.environ.get("FSLNET_THREADS", "1"))
os.environ.setdefault("OMP_NUM_THREADS", str(_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_threads))

import numpy as np   # noqa: E402
import torch         # noqa: E402


def _read_exact(k):
    buf = bytearray()
    while len(buf) < k:
        chunk = _in.read(k - len(buf))
        if not chunk:
            raise EOFError
        buf.extend(chunk)
    return bytes(buf)


def _write_all(b):
    view = memoryview(b)
    while len(view):
        k = _out.write(view)
        view = view[k:]


def main():
    torch.set_num_threads(_threads)
    device = os.environ.get("FSLNET_DEVICE", "cpu")
    try:
        from fslnet.fslnet import FSLNet
        model = FSLNet.from_pretrained(device=device)
        model.eval()
    except Exception as e:  # noqa: BLE001
        print("fslnet_worker: load failed: %r" % (e,), file=sys.stderr, flush=True)
        _write_all(b"FSLR" + struct.pack("<i", 1))
        return 1
    _write_all(b"FSLR" + struct.pack("<i", 0))
    while True:
        try:
            head = _read_exact(16)
        except EOFError:
            return 0
        if head[:4] != b"FSLQ":
            return 2
        n, w, d = struct.unpack("<iii", head[4:])
        try:
            ref = np.frombuffer(_read_exact(4 * n * d), dtype="<f4").reshape(n, d)
            win = np.frombuffer(_read_exact(4 * w * d), dtype="<f4").reshape(w, d)
        except EOFError:
            return 0
        try:
            r = torch.tensor(ref, dtype=torch.float32).to(device)
            q = torch.tensor(win, dtype=torch.float32).to(device)
            with torch.no_grad():
                soft, _ = model(r, q)
            p = soft.detach().to("cpu").numpy().astype("<f8").reshape(-1)
            if p.shape[0] != d:
                raise ValueError("output length %d != d %d" % (p.shape[0], d))
            status = 0
        except Exception as e:  # noqa: BLE001
            print("fslnet_worker: call failed: %r" % (e,), file=sys.stderr, flush=True)
            p = np.full(d, np.nan, dtype="<f8")
            status = 1
        _write_all(b"FSLA" + struct.pack("<ii", status, d) + p.tobytes())


if __name__ == "__main__":
    sys.exit(main())
