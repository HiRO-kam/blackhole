"""Finite static observer, spherical emitting background, G=c=M=E=1."""
import numpy as np
from ray_engine import trace_batch
from geodesic import rk4_step


def trajectory(b, psi=0., r0=30.):
    s = np.array([r0, -np.sqrt(1-b*b/r0**2*(1-2/r0)), 0.])
    history = [s.copy()]
    status = '未判定'
    for _ in range(18000):
        old = s.copy()
        s = rk4_step(s, min(.15, .008*s[0]), b, 1.)
        target = 2. if s[0] <= 2 else r0
        if s[0] <= 2 or (s[0] >= r0 and s[1] > 0):
            t = (target-old[0])/(s[0]-old[0])
            s = old+t*(s-old)
            history.append(s.copy())
            status = '捕獲' if target == 2 else '脱出'
            break
        history.append(s.copy())
    a = np.array(history)
    r, p = a[:, 0], a[:, 2]
    return np.array([r*np.cos(p), r*np.sin(p)*np.cos(psi), r*np.sin(p)*np.sin(psi)]).T, status


def background(d, rotation=0., grid=True):
    lon = np.arctan2(d[..., 2], d[..., 0])+rotation
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    u, v = lon/(2*np.pi) % 1, .5-lat/np.pi
    if grid:
        c = np.stack((.25+.45*u, .25+.4*v, .4+.25*np.sin(lon)**2), -1)
        lines = (np.minimum(u*24 % 1, 1-u*24 % 1)<.035) | (np.minimum(v*12 % 1, 1-v*12 % 1)<.035)
        c[lines] = [.95, .96, 1.]
    else:
        glow = np.exp(-((lat-.28*np.sin(lon*2))/.13)**2)
        c = np.stack((.015+.10*glow, .025+.09*glow, .06+.18*glow), -1)
        # Deterministic celestial texture, identical for lensed/reference images.
        ix, iy = np.floor(u*900), np.floor(v*450)
        h = np.mod(np.sin(ix*127.1+iy*311.7)*43758.5453, 1)
        star = h > .994
        c[star] = (.65+.35*h[star])[:, None]
    return np.uint8(np.clip(c, 0, 1)*255)


def render_image(n=256, fov=45., rotation=0., grid=True, progress=None):
    r0=30.
    q = (np.arange(n)+.5-n/2)/(n/2)*np.tan(np.deg2rad(fov/2))
    yy, zz = np.meshgrid(q, -q)
    rho = np.hypot(yy, zz)
    b = r0*rho/np.sqrt(1+rho*rho)/np.sqrt(1-2/r0)
    # Only exact duplicate radii share work; no interpolation across the shadow edge.
    unique, inv = np.unique(b, return_inverse=True)
    phi = np.empty(len(unique)); status = np.empty(len(unique), int)
    for start in range(0, len(unique), 2048):
        end = min(start+2048, len(unique))
        s, st, _ = trace_batch(unique[start:end], r0=r0, sky=120.)
        phi[start:end], status[start:end] = s[:, 2], st
        if progress: progress(end/len(unique))
    p, st = phi[inv].reshape(n,n), status[inv].reshape(n,n)
    uy = np.divide(yy, rho, out=np.zeros_like(yy), where=rho>0)
    uz = np.divide(zz, rho, out=np.zeros_like(zz), where=rho>0)
    d = np.stack((np.cos(p), np.sin(p)*uy, np.sin(p)*uz), -1)
    image = background(d, rotation, grid)
    image[st==1] = 0
    image[st==0] = [255, 0, 180]  # Never silently turn unfinished rays black.
    # Straight rays intersect the same celestial sphere (finite observer).
    direction = np.stack((-np.ones_like(yy), yy, zz), -1)
    direction /= np.linalg.norm(direction, axis=-1)[...,None]
    dx=direction[...,0]
    distance=-r0*dx+np.sqrt((r0*dx)**2+120**2-r0**2)
    reference=direction*distance[...,None]; reference[...,0]+=r0
    reference=background(reference/120., rotation, grid)
    return image, reference, {'rays': n*n, 'unique':len(unique), 'unresolved':int((st==0).sum()), 'fov':fov, 'n':n}
