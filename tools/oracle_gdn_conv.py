#!/usr/bin/env python3
"""Oracle: lab streaming causal conv vs HF F.conv1d; GDN step vs HF recurrent."""
import math
import numpy as np

def silu(x):
    return x / (1.0 + np.exp(-x))

def softplus(x):
    # stable
    return np.where(x > 20, x, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0))

def lab_stream_conv(xs, w, flip=False):
    """xs: [T,C], w: [C,K] with w[:,0]=oldest tap (HF). history starts 0."""
    T, C = xs.shape
    K = w.shape[1]
    past = K - 1
    hist = np.zeros((C, past), np.float64)
    outs = np.zeros_like(xs, dtype=np.float64)
    for t in range(T):
        x = xs[t]
        for c in range(C):
            if not flip:
                y = w[c, past] * x[c]
                for j in range(past):
                    y += w[c, j] * hist[c, j]
            else:
                y = w[c, 0] * x[c]
                for j in range(past):
                    y += w[c, past - j] * hist[c, j]
            outs[t, c] = y
            if past:
                hist[c, :-1] = hist[c, 1:]
                hist[c, -1] = x[c]
    return outs

def hf_causal_conv(xs, w):
    """xs [T,C], w [C,K]. Pad K-1 left, F.conv1d groups=C, take :T."""
    T, C = xs.shape
    K = w.shape[1]
    # depthwise: for each c, conv1d
    pad = np.zeros((past := K - 1, C), np.float64)
    xpad = np.concatenate([pad, xs], axis=0)  # [T+past, C]
    outs = np.zeros_like(xs, dtype=np.float64)
    for c in range(C):
        for t in range(T):
            window = xpad[t:t+K, c]  # oldest..current
            outs[t, c] = np.dot(w[c], window)
    return outs

def lab_delta_step(S, q, k, v, decay, beta):
    """S [kd, vd], decay-first."""
    S = S * decay
    kv = S.T @ k  # [vd]
    S = S + np.outer(k, beta * (v - kv))
    o = S.T @ q
    return S, o

def fused_delta_step(S, q, k, v, decay, beta):
    """Old fused: kv from undecayed S."""
    kv = S.T @ k
    S = decay * S + np.outer(k, beta * (v - kv))
    o = S.T @ q
    return S, o

def hf_recurrent(qs, ks, vs, gs_log, betas):
    """qs/ks [T,kd], vs [T,vd], gs_log [T] (log-decay), betas [T]."""
    T, kd = qs.shape
    vd = vs.shape[1]
    S = np.zeros((kd, vd), np.float64)
    outs = np.zeros((T, vd), np.float64)
    for t in range(T):
        S = S * np.exp(gs_log[t])
        kv = S.T @ ks[t]
        delta = (vs[t] - kv) * betas[t]
        S = S + np.outer(ks[t], delta)
        outs[t] = S.T @ qs[t]
    return outs, S

def main():
    rng = np.random.default_rng(0)
    T, C, K = 16, 8, 4
    xs = rng.normal(size=(T, C))
    w = rng.normal(size=(C, K))
    lab = lab_stream_conv(xs, w, flip=False)
    hf = hf_causal_conv(xs, w)
    flip = lab_stream_conv(xs, w, flip=True)
    print('conv lab vs hf maxabs', np.max(np.abs(lab - hf)))
    print('conv flip vs hf maxabs', np.max(np.abs(flip - hf)))
    assert np.max(np.abs(lab - hf)) < 1e-12

    # GDN: show fused vs decay-first diverge; decay-first matches HF
    T, kd, vd = 20, 8, 8
    q = rng.normal(size=(T, kd)); k = rng.normal(size=(T, kd)); v = rng.normal(size=(T, vd))
    # L2 + qscale
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-6)
    k = k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-6)
    q = q / math.sqrt(kd)
    a_log = rng.uniform(0.01, 2.0, size=T)  # positive A
    ssm_a = -np.exp(np.log(a_log))  # -exp(A_log) with A_log=log(a_log)
    # simpler: ssm_a negative
    ssm_a = -np.exp(rng.uniform(-2, 2, size=1)[0]) * np.ones(T)
    alpha = rng.normal(size=T); dt = rng.normal(size=T)
    sp = softplus(alpha + dt)
    g_log = ssm_a * sp  # already log-decay (negative * positive)
    beta = 1/(1+np.exp(-rng.normal(size=T)))

    outs_hf, _ = hf_recurrent(q, k, v, g_log, beta)
    S = np.zeros((kd, vd)); outs_lab = []
    for t in range(T):
        S, o = lab_delta_step(S, q[t], k[t], v[t], np.exp(g_log[t]), beta[t])
        outs_lab.append(o)
    outs_lab = np.stack(outs_lab)
    S2 = np.zeros((kd, vd)); outs_fused = []
    for t in range(T):
        S2, o = fused_delta_step(S2, q[t], k[t], v[t], np.exp(g_log[t]), beta[t])
        outs_fused.append(o)
    outs_fused = np.stack(outs_fused)
    print('delta decay-first vs hf maxabs', np.max(np.abs(outs_lab - outs_hf)))
    print('delta fused vs hf maxabs', np.max(np.abs(outs_fused - outs_hf)))
    # When decay≈1, fused≈decay-first
    print('fused vs decay-first maxabs', np.max(np.abs(outs_fused - outs_lab)))

    # If decays near 1, fused matches
    g1 = np.zeros(T)  # log-decay 0 => decay 1
    outs_hf1, _ = hf_recurrent(q, k, v, g1, beta)
    S = np.zeros((kd, vd)); outs_f = []
    for t in range(T):
        S, o = fused_delta_step(S, q[t], k[t], v[t], 1.0, beta[t]); outs_f.append(o)
    print('when decay=1 fused vs hf', np.max(np.abs(np.stack(outs_f)-outs_hf1)))

if __name__ == '__main__':
    main()
