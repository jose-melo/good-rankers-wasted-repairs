"""Build data/cats/cats_thin.npz from the CATS files data.csv and metadata.csv (Zenodo 10.5281/zenodo.8338435):
all 17 channels, one row every 10 s, the label column and the anomaly segments (inclusive 1 Hz row indices)."""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wasted_repairs import config  # noqa: E402

THIN = 10
CHANNELS = ["aimp", "amud", "arnd", "asin1", "asin2", "adbr", "adfl",
            "bed1", "bed2", "bfo1", "bfo2", "bso1", "bso2", "bso3", "ced1", "cfo1", "cso1"]
MD5 = {"data.csv": "92050471389956f07603b83f7dbf6d0e", "metadata.csv": "cb52d1d73bf4bb54ebbfb466b94618de"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    raw = config.CATS
    for fn, want in MD5.items():
        assert md5(os.path.join(raw, fn)) == want, f"md5 mismatch for {fn}"
    df = pd.read_csv(os.path.join(raw, "data.csv"))
    assert list(df.columns) == ["timestamp"] + CHANNELS + ["y", "category"], df.columns
    ts = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    sec = ((ts - ts.iloc[0]).dt.total_seconds()).to_numpy().astype(np.int64)
    assert np.all(np.diff(sec) == 1), "data.csv is not a regular 1 Hz series"
    X = df[CHANNELS].to_numpy(dtype=np.float64)
    yb = df["y"].to_numpy() != 0
    d = np.diff(np.r_[0, yb.astype(np.int8), 0])
    lo = np.flatnonzero(d == 1); hi = np.flatnonzero(d == -1) - 1
    m = pd.read_csv(os.path.join(raw, "metadata.csv"))
    m["start_time"] = pd.to_datetime(m["start_time"]); m["end_time"] = pd.to_datetime(m["end_time"])
    m = m.sort_values("start_time").reset_index(drop=True)
    ms = ((m.start_time - ts.iloc[0]).dt.total_seconds()).to_numpy().astype(np.int64)
    me = ((m.end_time - ts.iloc[0]).dt.total_seconds()).to_numpy().astype(np.int64)
    seg_lo, seg_hi = (lo, hi) if len(lo) == len(ms) else (ms, me - 1)
    idx = np.arange(0, len(df), THIN)
    out = os.path.join(raw, "cats_thin.npz")
    np.savez(out, X=np.ascontiguousarray(X[idx]), row=idx.astype(np.int64), y=yb[idx].astype(np.int8),
             names=np.array(CHANNELS), seg_lo=seg_lo.astype(np.int64), seg_hi=seg_hi.astype(np.int64),
             thin=np.int64(THIN))
    print("wrote", out, len(idx), "rows,", len(seg_lo), "anomaly segments")


if __name__ == "__main__":
    main()
