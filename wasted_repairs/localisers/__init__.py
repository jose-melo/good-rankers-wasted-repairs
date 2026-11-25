"""Localisers evaluated in the paper. KS and the conditional test (COND) are computed by the harness itself."""
import importlib

import numpy as np

# name -> (module, kind). kind "p": p-values, Holm at 0.10; "score": thresholded scores.
PLUGINS = {"ARM": "arm", "CLF": "classifier", "FSL": "fsl_net"}
MODULES = {"LOO": "loo", "KS_B": "block", "COND_B": "block"}
BUILTIN = ("KS", "COND")
ALL = BUILTIN + ("ARM", "CLF", "FSL", "LOO", "KS_B", "COND_B")


def kind(name):
    if name in BUILTIN or name in MODULES:
        return "p"
    return importlib.import_module(f".{PLUGINS[name]}", __name__).KIND


class Plugins:
    """Localisers that see only the reference rows and the window (ARM, classifier, FSL-Net)."""

    def __init__(self, ctx):
        self.d, self.seed = int(ctx.d), int(ctx.seed)
        self.ref = np.vstack([ctx.Xf, ctx.Xc])[:, :self.d]
        self.mods = [(nm, importlib.import_module(f".{PLUGINS[nm]}", __name__)) for nm in ctx.entrants]

    def window(self, W, twin, consecutive, resid):
        out = {}
        for nm, p in self.mods:
            v = np.full(self.d, np.nan)
            if len(W) >= getattr(p, "MIN_W", 1):
                try:
                    v = np.asarray(p.run(self.ref, W[:, :self.d], self.seed), float)
                except Exception as e:  # noqa: BLE001
                    print("localiser", nm, "failed:", repr(e)[:200], flush=True)
            out[nm] = v
        return out


def build(names, make_ctx):
    """Group the requested localisers by module and fit each module once. make_ctx(entrants) -> context."""
    groups = {}
    for nm in names:
        if nm in BUILTIN:
            continue
        key = "plugins" if nm in PLUGINS else MODULES[nm]
        groups.setdefault(key, []).append(nm)
    objs = []
    for key, ents in groups.items():
        ctx = make_ctx(ents)
        objs.append(Plugins(ctx) if key == "plugins" else importlib.import_module(f".{key}", __name__).fit(ctx))
    return objs
