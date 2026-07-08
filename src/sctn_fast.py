"""Numba-JIT fast path for the SCTN resonator used by the encoder.

The `sctn` package ships with numba disabled (numba_helper._optimize=True routes
@jitclass/@njit to no-op decorators), so the resonator runs as pure interpreted
Python at ~400 s per resonator-minute. This module reimplements the exact
forward pass of `sctn.resonator.resonator()` (a fixed 5-neuron topology:
delta-sigma input neuron 0 -> chain 1->2->3->4 with a 1-step-delayed feedback
4->1) as a single @njit kernel, producing bit-identical spike trains.

Bit-identity is enforced by construction (float64 throughout, same op order,
same int16 input quantization, same IDENTITY/leakage/reset sequence) and verified
against the pure-Python output and the spikes already on disk.

Parameters are extracted from a genuinely-constructed resonator object, so the
weights/thetas/leakage/topology can never drift from the library.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from sctn.resonator import resonator as _resonator


# --------------------------------------------------------------------------
# Single-neuron update: integrate -> clip -> IDENTITY activation -> leakage ->
# optional reset. Mirrors SCTNeuron._kernel + _activation_function_identity +
# _log_cycle reset, in that exact order.
# --------------------------------------------------------------------------
@njit(cache=True)
def _neuron(m, rgv, timer, ws, theta, LF, LP, ident, minclip, maxclip, reset_flag, reset_to):
    # integrate
    if LF < 3:
        m = m + ws + theta
    else:
        lf = 2.0 ** (LF - 3)
        m = m + ws * lf + theta * lf
    # clip
    if m < minclip:
        m = minclip
    elif m > maxclip:
        m = maxclip
    # IDENTITY activation
    c = m + ident
    if m > ident:
        emit = 1
        rgv = float(ident)
    elif m < -ident:
        emit = 0
        rgv = float(ident)
    else:
        rgv = float(int(rgv + c + 1.0))       # int() truncates toward zero
        if rgv >= 65536.0:
            rgv = rgv - 65536.0
            emit = 1
        else:
            emit = 0
    # leakage
    if timer >= LP:
        if m < 0.0:
            m = m + (-m) / (2.0 ** LF)
        else:
            m = m - (m / (2.0 ** LF))
        timer = 0
    else:
        timer = timer + 1
    # reset
    if reset_flag == 1 and emit > 0:
        m = reset_to
    return m, rgv, timer, emit


@njit(cache=True)
def _run_stream(signal, amplitude,
                LF, LP, theta, ident, minclip, maxclip, reset_flag, reset_to,
                w1_fwd, w1_fb, w2, w3, w4,
                m, rgv, timer, s4_prev):
    """Process one signal chunk, mutating state arrays; return (spikes, s4_prev)."""
    n = signal.shape[0]
    out = np.empty(n, dtype=np.int8)
    for t in range(n):
        # neuron 0 (delta-sigma): membrane set directly from int16-quantized input
        p = np.int16(signal[t] * amplitude)
        m0 = float(p)
        m0, rgv[0], timer[0], s0 = _neuron(
            m0, rgv[0], timer[0], 0.0, theta[0], LF[0], LP[0],
            ident[0], minclip[0], maxclip[0], reset_flag[0], reset_to[0])
        # neuron 1: forward from n0 (this step) + feedback from n4 (previous step)
        ws1 = s0 * w1_fwd + s4_prev * w1_fb
        m[1], rgv[1], timer[1], s1 = _neuron(
            m[1], rgv[1], timer[1], ws1, theta[1], LF[1], LP[1],
            ident[1], minclip[1], maxclip[1], reset_flag[1], reset_to[1])
        # neuron 2
        m[2], rgv[2], timer[2], s2 = _neuron(
            m[2], rgv[2], timer[2], s1 * w2, theta[2], LF[2], LP[2],
            ident[2], minclip[2], maxclip[2], reset_flag[2], reset_to[2])
        # neuron 3
        m[3], rgv[3], timer[3], s3 = _neuron(
            m[3], rgv[3], timer[3], s2 * w3, theta[3], LF[3], LP[3],
            ident[3], minclip[3], maxclip[3], reset_flag[3], reset_to[3])
        # neuron 4 (output)
        m[4], rgv[4], timer[4], s4 = _neuron(
            m[4], rgv[4], timer[4], s3 * w4, theta[4], LF[4], LP[4],
            ident[4], minclip[4], maxclip[4], reset_flag[4], reset_to[4])
        out[t] = s4
        s4_prev = s4
    return out, s4_prev


_PARAM_CACHE: dict = {}


def _extract(f0, clk):
    key = (f0, clk)
    if key in _PARAM_CACHE:
        return _PARAM_CACHE[key]
    net = _resonator(freq0=f0, clk_freq=clk)
    amp = float(net.amplitude[0])
    neurons = [net.layers_neurons[i].neurons[0] for i in range(5)]
    LF = np.array([int(n.leakage_factor) for n in neurons], dtype=np.int64)
    LP = np.array([int(n.leakage_period) for n in neurons], dtype=np.int64)
    theta = np.array([float(n.theta) for n in neurons], dtype=np.float64)
    ident = np.array([float(n.identity_const) for n in neurons], dtype=np.float64)
    minclip = np.array([float(n.min_clip) for n in neurons], dtype=np.float64)
    maxclip = np.array([float(n.max_clip) for n in neurons], dtype=np.float64)
    reset_flag = np.array([1 if n.membrane_should_reset else 0 for n in neurons], dtype=np.int64)
    reset_to = np.array([float(n.reset_to) for n in neurons], dtype=np.float64)
    w1 = neurons[1].synapses_weights
    w1_fwd, w1_fb = float(w1[0]), float(w1[1])
    w2 = float(neurons[2].synapses_weights[0])
    w3 = float(neurons[3].synapses_weights[0])
    w4 = float(neurons[4].synapses_weights[0])
    params = dict(amp=amp, LF=LF, LP=LP, theta=theta, ident=ident,
                  minclip=minclip, maxclip=maxclip, reset_flag=reset_flag, reset_to=reset_to,
                  w1_fwd=w1_fwd, w1_fb=w1_fb, w2=w2, w3=w3, w4=w4)
    _PARAM_CACHE[key] = params
    return params


def _fresh_state():
    return (np.zeros(5, dtype=np.float64),   # membrane
            np.zeros(5, dtype=np.float64),   # rand_gauss_var
            np.zeros(5, dtype=np.int64))     # leakage_timer


def _stream(signal, p, m, rgv, timer, s4_prev):
    return _run_stream(
        np.ascontiguousarray(signal, dtype=np.float64), p["amp"],
        p["LF"], p["LP"], p["theta"], p["ident"], p["minclip"], p["maxclip"],
        p["reset_flag"], p["reset_to"],
        p["w1_fwd"], p["w1_fb"], p["w2"], p["w3"], p["w4"],
        m, rgv, timer, s4_prev)


def resonate_no_warmup(f0, clk, signal):
    """Neuron-4 spike train for `signal` starting from zero state (no warmup).
    Matches: net.input_full_data(signal); last_neuron.out_spikes(False, len(signal))."""
    p = _extract(f0, clk)
    m, rgv, timer = _fresh_state()
    out, _ = _stream(signal, p, m, rgv, timer, 0)
    return out


def resonate_minutes(f0, clk, minute_signals, warmup_seconds=5):
    """Warm up with `warmup_seconds` of zeros, then process each minute signal in
    order carrying state (exactly like _run_one_resonator). Returns list of int8
    spike arrays, one per minute."""
    p = _extract(f0, clk)
    m, rgv, timer = _fresh_state()
    s4_prev = 0
    if warmup_seconds:
        warm = np.zeros(int(clk * warmup_seconds), dtype=np.float64)
        _, s4_prev = _stream(warm, p, m, rgv, timer, s4_prev)
    outs = []
    for sig in minute_signals:
        out, s4_prev = _stream(sig, p, m, rgv, timer, s4_prev)
        outs.append(out)
    return outs
