"""Schwarzschild null geodesics; G=c=M=E=1. Based on the supplied RK4.

Batch rays use independent radius-dependent affine steps, not error-controlled
steps. Termination is interpolated to r=2 or the emitting celestial sphere.
"""
import numpy as np

BC = np.sqrt(27.0)
OBSERVER = 40.0
SKY = 400.0


def derivatives(s, b):
    r, v, p = s.T
    return np.column_stack((v, b*b/r**3*(1-3/r), b/r**2))


def rk4(s, h, b):
    h = np.asarray(h)[..., None]
    k1 = derivatives(s, b)
    k2 = derivatives(s+h*k1/2, b)
    k3 = derivatives(s+h*k2/2, b)
    k4 = derivatives(s+h*k3, b)
    return s+h*(k1+2*k2+2*k3+k4)/6


def trace_batch(b, r0=OBSERVER, sky=SKY, scale=1.0):
    b = np.asarray(b, dtype=float)
    radicand = 1-b*b/r0**2*(1-2/r0)
    if np.any(radicand < 0):
        raise ValueError('Initial direction not accessible to a local observer')
    s = np.column_stack((np.full(b.size, r0), -np.sqrt(radicand), np.zeros(b.size)))
    status = np.zeros(b.size, dtype=int)  # 0 unresolved, 1 captured, 2 background
    error = np.zeros(b.size)
    for _ in range(30000):
        ix = np.flatnonzero(status == 0)
        if not len(ix):
            break
        old = s[ix].copy()
        step = scale*np.minimum(0.7, 0.006*old[:, 0])
        new = rk4(old, step, b[ix])
        q = new[:, 1]**2+b[ix]**2/new[:, 0]**2*(1-2/new[:, 0])
        error[ix] = np.maximum(error[ix], abs(q-1))
        capture = new[:, 0] <= 2
        escape = (new[:, 0] >= sky) & (new[:, 1] > 0)
        done = capture | escape
        target = np.where(capture, 2., sky)
        frac = np.zeros(len(ix))
        frac[done] = (target[done]-old[done, 0])/(new[done, 0]-old[done, 0])
        new[done] = old[done]+frac[done, None]*(new[done]-old[done])
        s[ix] = new
        status[ix[capture]] = 1
        status[ix[escape]] = 2
    return s, status, error


def build_lut():
    # Extra logarithmic samples resolve multiple turns close to the separatrix.
    near = np.geomspace(2e-7, .2, 1800)
    b = np.unique(np.r_[np.linspace(0, 24, 6001), BC-near, BC+near])
    s, status, error = trace_batch(b)
    assert np.all(status), 'Unresolved rays must not be rendered as a shadow'
    return {'b': b.tolist(), 'phi': s[:, 2].tolist(), 'status': status.tolist(),
            'observer': OBSERVER, 'sky': SKY, 'max_error': float(error.max()),
            'captured': int((status == 1).sum()), 'escaped': int((status == 2).sum())}
