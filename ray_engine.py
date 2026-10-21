"""Schwarzschild null geodesics, G=c=M=E=1, with finite-screen events.

The original differential equations, RK4 and inward initial conditions are
unchanged. Each accepted RK4 interval is a Cartesian line segment for event
localisation. Screen, horizon and outer-boundary events compete by segment
fraction; only the first event terminates a ray.
"""
import numpy as np

BC = np.sqrt(27.0)
OBSERVER = 40.0                 # Retain the legacy engine/LUT defaults.
SKY = 400.0                    # Outer computational boundary, not an emitter.
SCENE_OBSERVER = 30.0
SCENE_BOUNDARY = 120.0
SCREEN_X = -19.0
SCREEN_HALF_SIZE = 24.0
UNRESOLVED, CAPTURED, SCREEN_HIT, MISSED, NUMERICAL_FAILURE = range(5)
STATUS_LABELS = {UNRESOLVED: '未判定', CAPTURED: '捕獲',
                 SCREEN_HIT: 'スクリーン到達', MISSED: 'スクリーン未到達',
                 NUMERICAL_FAILURE: '数値エラー'}


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


def cartesian(s, psi=0.0):
    """Convert (r, dr/dlambda, phi) in the ray's orbital plane to xyz."""
    s = np.asarray(s)
    transverse = s[..., 0]*np.sin(s[..., 2])
    return np.stack((s[..., 0]*np.cos(s[..., 2]),
                     transverse*np.cos(psi), transverse*np.sin(psi)), -1)


def _sphere_fraction(a, d, radius, outward):
    """First inward / outward sphere crossing of a+t*d, 0 <= t <= 1."""
    aa = np.einsum('ij,ij->i', d, d)
    bb = np.einsum('ij,ij->i', a, d)
    cc = np.einsum('ij,ij->i', a, a)-radius*radius
    disc = bb*bb-aa*cc
    valid = (aa > 0) & (disc >= 0)
    # Stable quadratic roots also work for short segments near a boundary.
    root = np.sqrt(np.maximum(disc, 0))
    q = -bb-np.copysign(root, bb)
    r1 = np.divide(q, aa, out=np.full_like(aa, np.inf), where=aa > 0)
    r2 = np.divide(cc, q, out=np.zeros_like(aa), where=q != 0)
    t = np.maximum(r1, r2) if outward else np.minimum(r1, r2)
    return np.where(valid & (t >= 0) & (t <= 1), t, np.inf)


def segment_events(old, new, outer, screen_x=SCREEN_X,
                   half_size=SCREEN_HALF_SIZE):
    """Return status, fraction, endpoint for Cartesian segments (N, 3).

    A plane crossing outside the rectangle is ignored, allowing later valid
    crossings. Edges are included. On ties: horizon, then screen, then outer.
    screen_x=None is an explicit legacy celestial-sphere mode used by build_lut.
    """
    old, new = np.asarray(old, float), np.asarray(new, float)
    d = new-old
    capture = _sphere_fraction(old, d, 2.0, False)
    escape = _sphere_fraction(old, d, outer, True)
    hit = np.full(len(old), np.inf)
    if screen_x is not None:
        dx = d[:, 0]
        t = np.divide(screen_x-old[:, 0], dx, out=np.full(len(old), np.inf),
                      where=dx != 0)
        # A ray already on the screen is an immediate hit if inside its edges.
        t[(dx == 0) & (old[:, 0] == screen_x)] = 0.0
        crossing = np.flatnonzero((t >= 0) & (t <= 1))
        yz = old[crossing, 1:]+t[crossing, None]*d[crossing, 1:]
        inside = np.all(np.abs(yz) <= half_size, axis=1)
        ix = crossing[inside]
        hit[ix] = t[ix]
    status = np.full(len(old), UNRESOLVED, dtype=int)
    t = np.minimum(np.minimum(capture, hit), escape)
    done = np.isfinite(t)
    status[done & (t == escape)] = MISSED if screen_x is not None else SCREEN_HIT
    status[done & (t == hit)] = SCREEN_HIT
    status[done & (t == capture)] = CAPTURED
    fraction = np.where(done, t, 1.0)
    point = old+fraction[:, None]*d
    if screen_x is not None:
        point[status == SCREEN_HIT, 0] = screen_x
    return status, fraction, point


def constraint_error(s, b):
    r, v, _ = np.asarray(s).T
    return np.abs(v*v+b*b/r**2*(1-2/r)-1)


def _trace(b, r0, sky, scale, psi, screen_x, half_size, max_steps, record):
    b = np.atleast_1d(np.asarray(b, dtype=float))
    if b.ndim != 1 or not np.all(np.isfinite(b)):
        raise ValueError('b must be a finite one-dimensional array')
    psi = np.broadcast_to(np.asarray(psi, float), b.shape)
    if not np.all(np.isfinite(psi)):
        raise ValueError('psi must be finite and broadcastable to b')
    if not (np.isfinite(r0) and np.isfinite(sky) and 2 < r0 < sky):
        raise ValueError('Require 2 < observer radius < outer boundary')
    if not np.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('scale must be in (0, 1]; reduce it for convergence checks')
    if (not isinstance(max_steps, (int, np.integer))) or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')
    if screen_x is not None and not np.isfinite(screen_x):
        raise ValueError('screen_x must be finite, or None for legacy sphere mode')
    if not np.isfinite(half_size) or half_size <= 0:
        raise ValueError('half_size must be finite and positive')
    radicand = 1-b*b/r0**2*(1-2/r0)
    if np.any(radicand < 0):
        raise ValueError('Initial direction not accessible to a local observer')
    s = np.column_stack((np.full(b.size, r0), -np.sqrt(radicand), np.zeros(b.size)))
    status = np.zeros(b.size, dtype=int)
    error = np.zeros(b.size)
    if record and b.size != 1:
        raise ValueError('History is only recorded for a single ray')
    history = [cartesian(s, psi)[0]] if record else None
    for _ in range(max_steps):
        ix = np.flatnonzero(status == UNRESOLVED)
        if not len(ix):
            break
        old = s[ix].copy()
        step = scale*np.minimum(0.7, 0.006*old[:, 0])
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            new = rk4(old, step, b[ix])
            residual = constraint_error(new, b[ix])
        good = np.all(np.isfinite(new), axis=1) & np.isfinite(residual) & (new[:, 0] > 0)
        status[ix[~good]] = NUMERICAL_FAILURE
        error[ix[~good]] = np.inf
        ix, old, new = ix[good], old[good], new[good]
        if not len(ix):
            continue
        # This diagnostic measures RK4 grid states, before chord interpolation.
        error[ix] = np.maximum(error[ix], residual[good])
        old_xyz = cartesian(old, psi[ix])
        new_xyz = cartesian(new, psi[ix])
        event, fraction, point = segment_events(old_xyz, new_xyz, sky,
                                                 screen_x, half_size)
        done = event != UNRESOLVED
        if np.any(done):
            interp = old[done]+fraction[done, None]*(new[done]-old[done])
            xyz = point[done]
            w = xyz[:, 1]*np.cos(psi[ix[done]])+xyz[:, 2]*np.sin(psi[ix[done]])
            interp[:, 0] = np.linalg.norm(xyz, axis=1)
            raw_phi = np.arctan2(w, xyz[:, 0])
            # Retain accumulated turns, including negative b, across atan2's cut.
            interp[:, 2] += np.arctan2(np.sin(raw_phi-interp[:, 2]),
                                       np.cos(raw_phi-interp[:, 2]))
            new[done] = interp
        s[ix] = new
        status[ix] = event
        if record:
            history.append(point[0].copy())
    return s, status, error, history


def trace_batch(b, r0=OBSERVER, sky=SKY, scale=1.0, *, psi=0.0,
                screen_x=SCREEN_X, half_size=SCREEN_HALF_SIZE, max_steps=30000):
    """Return final (r, v, phi), status, maximum RK4 constraint residual.

    psi specifies orbital-plane orientation and may vary per ray. Status 3 is
    assigned only upon reaching the outer boundary without a screen hit; step
    exhaustion remains status 0 and a numerical failure is status 4. Terminal
    chord interpolation has its own O(h^2) error, separate from the RK4 residual.
    """
    s, status, error, _ = _trace(b, r0, sky, scale, psi, screen_x, half_size,
                                max_steps, False)
    return s, status, error


def trace_path(b, psi=0.0, r0=SCENE_OBSERVER, sky=SCENE_BOUNDARY, scale=1.0,
               *, screen_x=SCREEN_X, half_size=SCREEN_HALF_SIZE, max_steps=30000):
    """The same engine/events as trace_batch, with an exact terminal xyz point."""
    _, status, error, history = _trace([b], r0, sky, scale, psi, screen_x,
                                      half_size, max_steps, True)
    return np.asarray(history), int(status[0]), float(error[0])


def build_lut():
    """Legacy spherical-sky LUT; it cannot describe a finite square screen."""
    near = np.geomspace(2e-7, .2, 1800)
    b = np.unique(np.r_[np.linspace(0, 24, 6001), BC-near, BC+near])
    s, status, error = trace_batch(b, screen_x=None)
    assert np.all(np.isin(status, [CAPTURED, SCREEN_HIT])), 'Unresolved LUT rays'
    return {'b': b.tolist(), 'phi': s[:, 2].tolist(), 'status': status.tolist(),
            'observer': OBSERVER, 'sky': SKY, 'max_error': float(error.max()),
            'captured': int((status == CAPTURED).sum()),
            'escaped': int((status == SCREEN_HIT).sum())}
