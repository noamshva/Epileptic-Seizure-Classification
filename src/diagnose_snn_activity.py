"""
Diagnostic: run ONE window through the SNN with a Monitor on every layer and
report total spikes per layer. Wherever the count drops to zero is where the
signal dies — tells us whether a silent output layer is an under-training
problem (trainable weights too small) or a propagation problem (thresholds /
fixed weights too weak).

Usage (on the remote, same BIN_FACTOR as training):
    BIN_FACTOR=8 python -u src/diagnose_snn_activity.py
    BIN_FACTOR=8 python -u src/diagnose_snn_activity.py --weights output_data/trained_models/snn_weights.pt
"""
import sys
import types
import argparse

if "torch._six" not in sys.modules:
    _shim = types.ModuleType("torch._six")
    _shim.container_abcs = ()
    _shim.string_classes = (str,)
    _shim.int_classes = (int,)
    sys.modules["torch._six"] = _shim

import importlib.util as ilu
from pathlib import Path

import numpy as np
import torch
from bindsnet.network.monitors import Monitor

_SCRIPT_DIR = Path(__file__).resolve().parent
_spec = ilu.spec_from_file_location("train_chbmit", _SCRIPT_DIR / "3.run_train_chbmit.py")
_t = ilu.module_from_spec(_spec)
_spec.loader.exec_module(_t)


def layer_order(name: str) -> tuple:
    """Sort key so the report reads input → output."""
    if name.endswith("-resonators"):
        return (0, name)
    if name.endswith("-band"):
        return (1, name)
    if name.endswith("-topography"):
        return (2, name)
    return ({"input_layer": 3, "hidden1_layer": 4,
             "hidden2_layer": 5, "output_layer": 6}.get(name, 9), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="path to snn_weights.pt; if omitted, uses fresh random/init weights")
    ap.add_argument("--n-windows", type=int, default=2,
                    help="how many windows to probe (1 seizure + 1 non-seizure by default)")
    args = ap.parse_args()

    device = _t.DEVICE
    snn = _t.build_snn()
    snn.to(device)
    snn.train(mode=False)

    if args.weights:
        weights = torch.load(args.weights, map_location=device)
        loaded = 0
        for (src, tgt), conn in snn.connections.items():
            key = f"{src}__{tgt}"
            if key in weights and weights[key].shape == conn.w.shape:
                conn.w.copy_(weights[key].to(conn.w.device, conn.w.dtype))
                loaded += 1
        print(f"Loaded weights into {loaded}/{len(snn.connections)} connections from {args.weights}")
    else:
        print("Using fresh (untrained) weights")

    # Attach a spike monitor to every layer
    windows = list(_t.enumerate_windows(["chb01"], _t.WINDOW_SECONDS, _t.WINDOW_OVERLAP))
    seiz = next(w for w in windows if w["window_label"] == "seizure")
    non = next(w for w in windows if w["window_label"] == "non_seizure")
    probe = [seiz, non][: args.n_windows]

    cache = _t._MinuteCache(capacity=2)

    for win in probe:
        inputs, label = _t._build_window_inputs(win, cache)
        T = next(iter(inputs.values())).shape[0]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        monitors = {}
        for lname, layer in snn.layers.items():
            m = Monitor(layer, state_vars=("s",), time=T)
            snn.add_monitor(m, name=f"diag_{lname}")
            monitors[lname] = m

        print(f"\n=== window @ {win['abs_start_s']:.1f}s  label={'seizure' if label==1 else 'non_seizure'}  "
              f"(BIN_FACTOR={_t.BIN_FACTOR}, T={T}) ===")
        snn.run(inputs=inputs, time=T)

        # Aggregate spikes per layer, grouped for readability
        rows = []
        for lname, m in monitors.items():
            s = m.get("s")                      # [T, *, n]
            total = int(s.sum().item())
            n_neurons = snn.layers[lname].n
            rows.append((lname, n_neurons, total))

        # Summarize the resonator / band / topography groups (many layers) +
        # the unique classifier layers individually.
        def group_sum(suffix):
            return sum(t for (n, _, t) in rows if n.endswith(suffix))
        def group_layers(suffix):
            return [(n, neu, t) for (n, neu, t) in rows if n.endswith(suffix)]

        print(f"  [input ] resonators  total spikes = {group_sum('-resonators'):>10d}")
        print(f"  [agg   ] band        total spikes = {group_sum('-band'):>10d}")
        print(f"  [topo  ] topography  total spikes = {group_sum('-topography'):>10d}")
        for lname in ("input_layer", "hidden1_layer", "hidden2_layer", "output_layer"):
            t = next(t for (n, _, t) in rows if n == lname)
            neu = snn.layers[lname].n
            print(f"  [{lname:13s}] n={neu:<4d}  total spikes = {t:>10d}")

        # Per-output-neuron detail
        out_s = monitors["output_layer"].get("s")
        per_neuron = out_s.sum(dim=tuple(range(out_s.ndim - 1))).long().tolist()
        print(f"  output per-neuron spike counts = {per_neuron}  (idx0=non_seizure, idx1=seizure)")

        for lname in list(snn.monitors.keys()):
            if lname.startswith("diag_"):
                del snn.monitors[lname]
        snn.reset_state_variables()


if __name__ == "__main__":
    main()
