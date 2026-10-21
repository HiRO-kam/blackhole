"""Finite planar emitting screen; static observer, G=c=M=E=1."""
import time
import numpy as np
from ray_engine import (
    trace_batch, trace_path, constraint_error, STATUS_LABELS,
    SCENE_OBSERVER, SCENE_BOUNDARY, SCREEN_X, SCREEN_HALF_SIZE,
    UNRESOLVED, CAPTURED, SCREEN_HIT, MISSED, NUMERICAL_FAILURE,
)

MISSED_COLOR = np.array([28, 36, 48], dtype=np.uint8)
UNRESOLVED_COLOR = np.array([255, 0, 180], dtype=np.uint8)
FAILURE_COLOR = np.array([255, 64, 64], dtype=np.uint8)


def trajectory(b, psi=0., r0=SCENE_OBSERVER, sky=SCENE_BOUNDARY,
               *, scale=1.0, max_steps=30000):
    path, status, _ = trace_path(b, psi, r0, sky, scale, max_steps=max_steps)
    return path, STATUS_LABELS[status]


def screen_background(y, z, rotation=0., grid=True, *, show_lines=True):
    """RGB texture of the physical screen at Cartesian (SCREEN_X, y, z).

    Rotation turns the texture within the fixed rectangle, not the ray or plane.
    This sampler is shared by the generated image, flat reference and 3D plane.
    """
    y, z = np.broadcast_arrays(np.asarray(y, float), np.asarray(z, float))
    c, s = np.cos(rotation), np.sin(rotation)
    u = .5+(c*y+s*z)/(2*SCREEN_HALF_SIZE)
    v = .5-(-s*y+c*z)/(2*SCREEN_HALF_SIZE)
    if grid:
        color = np.stack((.25+.45*np.clip(u, 0, 1),
                           .25+.40*np.clip(v, 0, 1),
                           .4+.25*np.sin(2*np.pi*u)**2), -1)
        lines = ((np.minimum(u*12 % 1, 1-u*12 % 1) < .035) |
                 (np.minimum(v*12 % 1, 1-v*12 % 1) < .035))
        if show_lines:
            color = np.where(lines[..., None], [.95, .96, 1.], color)
    else:
        glow = np.exp(-((v-.5-.15*np.sin(4*np.pi*u))/.065)**2)
        color = np.stack((.015+.10*glow, .025+.09*glow, .06+.18*glow), -1)
        ix, iy = np.floor(u*900), np.floor(v*450)
        h = np.mod(np.sin(ix*127.1+iy*311.7)*43758.5453, 1)
        color = np.where((h > .994)[..., None], (.65+.35*h)[..., None], color)
    return np.uint8(np.clip(color, 0, 1)*255)


def screen_grid_lines(rotation=0.):
    """Texture-grid centre lines clipped to the physical square, for 3D display.

    Drawing these as lines keeps the Tk animation light and avoids losing thin
    grid stripes when the surface texture is sampled with a coarse mesh.
    """
    c, s = np.cos(rotation), np.sin(rotation)
    limit = SCREEN_HALF_SIZE
    spacing = 2*limit/12
    result = []
    for a, b in [(c, s), (s, -c)]:
        for level in np.arange(-2*limit, 2*limit+spacing/2, spacing):
            points = []
            for edge in [-limit, limit]:
                if abs(b) > 1e-14:
                    z = (level-a*edge)/b
                    if abs(z) <= limit+1e-12:
                        points.append([edge, float(np.clip(z, -limit, limit))])
                if abs(a) > 1e-14:
                    y = (level-b*edge)/a
                    if abs(y) <= limit+1e-12:
                        points.append([float(np.clip(y, -limit, limit)), edge])
            if len(points) >= 2:
                points = np.unique(points, axis=0)
                if len(points) >= 2:
                    result.append(points[[0, -1]])
    return result


def background(points, rotation=0., grid=True):
    """Compatibility name: points now mean physical screen xyz, not sky units."""
    points = np.asarray(points)
    return screen_background(points[..., 1], points[..., 2], rotation, grid)


def camera_ray(y, z, r0=SCENE_OBSERVER):
    """Camera-plane slopes -> original finite static observer's (b, psi)."""
    rho = np.hypot(y, z)
    b = r0*rho/np.sqrt(1+rho*rho)/np.sqrt(1-2/r0)
    return b, np.arctan2(z, y)


def pixel_ray(x, y, n, fov, r0=SCENE_OBSERVER):
    """Matplotlib pixel coordinates have integer centres, like render_image."""
    t = np.tan(np.deg2rad(fov/2))
    yy = (np.asarray(x)+.5-n/2)/(n/2)*t
    zz = -(np.asarray(y)+.5-n/2)/(n/2)*t
    return camera_ray(yy, zz, r0)


def render_image(n=256, fov=45., rotation=0., grid=True, progress=None,
                 *, scale=1.0, max_steps=30000, batch_size=2048):
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive integer')
    if not np.isfinite(fov) or not 0 < fov < 180:
        raise ValueError('fov must be between 0 and 180 degrees')
    if not np.isfinite(rotation):
        raise ValueError('rotation must be finite')
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    start_time = time.perf_counter()
    r0 = SCENE_OBSERVER
    q = (np.arange(n)+.5-n/2)/(n/2)*np.tan(np.deg2rad(fov/2))
    yy, zz = np.meshgrid(q, -q)
    rho = np.hypot(yy, zz)
    b, _ = camera_ray(yy, zz, r0)
    uy = np.divide(yy, rho, out=np.zeros_like(yy), where=rho > 0)
    uz = np.divide(zz, rho, out=np.zeros_like(zz), where=rho > 0)
    # The square has reflection/quarter-turn symmetry, not axial symmetry.
    # Sharing requires BOTH identical b AND identical aperture orientation.
    # No interpolation in b or across the shadow/screen edges is performed.
    aperture = np.maximum(np.abs(uy), np.abs(uz))
    aperture[rho == 0] = 1.
    unique, inv = np.unique(np.column_stack((b.ravel(), aperture.ravel())),
                             axis=0, return_inverse=True)
    states = np.empty((len(unique), 3))
    status = np.empty(len(unique), int)
    max_error = 0.
    for start in range(0, len(unique), batch_size):
        end = min(start+batch_size, len(unique))
        representative_psi = np.arccos(np.clip(unique[start:end, 1], 0, 1))
        s, st, errors = trace_batch(unique[start:end, 0], r0=r0,
                                     sky=SCENE_BOUNDARY, scale=scale,
                                     psi=representative_psi, max_steps=max_steps)
        states[start:end], status[start:end] = s, st
        max_error = max(max_error, float(errors.max()))
        if progress:
            progress(end/len(unique))
    st = status[inv].reshape(n, n)
    final = states[inv].reshape(n, n, 3)
    w = final[..., 0]*np.sin(final[..., 2])
    image = np.empty((n, n, 3), dtype=np.uint8)
    image[:] = MISSED_COLOR
    hit = st == SCREEN_HIT
    # Sample the physical hit point only. Missed/unresolved rays have no texel.
    image[hit] = screen_background((w*uy)[hit], (w*uz)[hit], rotation, grid)
    image[st == CAPTURED] = 0
    image[st == UNRESOLVED] = UNRESOLVED_COLOR
    image[st == NUMERICAL_FAILURE] = FAILURE_COLOR
    # Gravity-free rays intersect the very same rectangle in physical units.
    ref_y, ref_z = (r0-SCREEN_X)*yy, (r0-SCREEN_X)*zz
    ref_hit = (np.abs(ref_y) <= SCREEN_HALF_SIZE) & (np.abs(ref_z) <= SCREEN_HALF_SIZE)
    reference = np.empty_like(image)
    reference[:] = MISSED_COLOR
    reference[ref_hit] = screen_background(ref_y[ref_hit], ref_z[ref_hit], rotation, grid)
    meta = {'rays': n*n, 'unique': len(unique), 'fov': fov, 'n': n,
            'rotation': rotation, 'grid': grid, 'scale': scale,
            'captured': int((st == CAPTURED).sum()),
            'screen_hits': int(hit.sum()), 'missed': int((st == MISSED).sum()),
            'unresolved': int((st == UNRESOLVED).sum()),
            'numerical_failures': int((st == NUMERICAL_FAILURE).sum()),
            'max_error': max_error,
            'max_endpoint_error': float(constraint_error(states, unique[:, 0]).max()),
            'elapsed_seconds': time.perf_counter()-start_time,
            'observer': r0, 'outer': SCENE_BOUNDARY,
            'screen_x': SCREEN_X, 'screen_half_size': SCREEN_HALF_SIZE}
    return image, reference, meta
