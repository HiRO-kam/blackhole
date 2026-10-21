"""Reproducible numerical/GUI validation and PNG/JPEG examples.

python validate_screen.py --scipy --original-dir PATH
SciPy is optional and is used only for the independent DOP853 reference.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import platform
import time
import numpy as np
from PIL import Image
from ray_engine import (trace_batch, cartesian, constraint_error, BC,
                        CAPTURED, SCREEN_HIT, MISSED, derivatives, rk4)
from imaging import render_image, pixel_ray, trajectory


def reference_ray(b, psi):
    from scipy.integrate import solve_ivp

    def rhs(_, s):
        r, v, phi = s
        return [v, b*b/r**3*(1-3/r), b/r**2]

    def horizon(_, s):
        return s[0]-2

    def outer(_, s):
        return s[0]-120

    def plane(_, s):
        return s[0]*np.cos(s[2])+19

    horizon.terminal, horizon.direction = True, -1
    outer.terminal, outer.direction = True, 1
    initial = [30., -np.sqrt(1-b*b/30**2*(1-2/30)), 0.]
    sol = solve_ivp(rhs, [0., 500.], initial, method='DOP853',
                    rtol=1e-12, atol=1e-13, max_step=.2,
                    events=[horizon, outer, plane])
    assert sol.success, sol.message
    for state in sol.y_events[2]:
        xyz = cartesian(state, psi)
        if np.all(np.abs(xyz[1:]) <= 12.):
            return SCREEN_HIT, xyz
    if len(sol.t_events[0]):
        return CAPTURED, cartesian(sol.y_events[0][0], psi)
    assert len(sol.t_events[1]), 'Reference did not reach a terminating event'
    return MISSED, cartesian(sol.y_events[1][0], psi)


def validate_original(directory):
    path = Path(directory)/'ray_engine.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    import inspect
    result = {}
    for fn in [derivatives, rk4]:
        original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn.__name__)
        current = ast.parse(inspect.getsource(fn)).body[0]
        result[fn.__name__+'_ast_unchanged'] = ast.dump(original) == ast.dump(current)
        assert result[fn.__name__+'_ast_unchanged']
    spec = importlib.util.spec_from_file_location('original_ray_engine', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    b = np.array([0., 4.5, 5.2, 6., 7., 12.])
    s, st, err = trace_batch(b, r0=30, sky=120, screen_x=None)
    old_s, old_st, old_err = module.trace_batch(b, r0=30, sky=120)
    np.testing.assert_array_equal(st, old_st)
    # Before event interpolation, the integral and RK4 residual are identical.
    np.testing.assert_allclose(err, old_err, atol=0, rtol=0)
    result['legacy_statuses_unchanged'] = True
    result['rk4_residuals_bit_identical'] = True
    result['legacy_terminal_coordinate_difference'] = float(np.max(np.linalg.norm(cartesian(s)-cartesian(old_s), axis=1)))
    return result


def gui_smoke(out):
    print('GUI: importing Tk', flush=True)
    import tkinter as tk
    from types import SimpleNamespace
    from unittest.mock import patch
    from ray_tracing_3d import BlackHoleApp

    print('GUI: creating hidden root', flush=True)
    root = tk.Tk()
    root.withdraw()
    callbacks = []
    root.report_callback_exception = lambda *args: callbacks.append(str(args))
    print('GUI: constructing app', flush=True)
    app = BlackHoleApp(root)
    print('GUI: app constructed', flush=True)
    app.launch(True)
    assert [s for _, s, _ in app.paths] == ['捕獲', 'スクリーン未到達', 'スクリーン到達']
    app.toggle()
    app.reset_view()
    app.res.set('128')
    app.generate()
    print('GUI: generating', flush=True)
    def wait_for_generation():
        app.playing = False
        if not app.busy:
            root.quit()
        else:
            root.after(10, wait_for_generation)
    root.after(10, wait_for_generation)
    timeout = root.after(40000, root.quit)
    root.mainloop()
    root.after_cancel(timeout)
    assert not app.busy, 'GUI generation timeout'
    assert app.image is not None, app.message.get()
    assert app.image.shape == (128, 128, 3)
    print('GUI: generation complete', flush=True)
    app.pick(SimpleNamespace(inaxes=app.result_ax, xdata=78.3, ydata=62.2))
    expected_b, expected_psi = pixel_ray(78, 62, 128, 45.)
    path, label = trajectory(float(expected_b), float(expected_psi))
    assert app.paths[0][1] == label
    np.testing.assert_allclose(app.paths[0][0][-1], path[-1], atol=1e-10)
    app.frame = 90
    app.tick()
    np.testing.assert_array_equal(np.array(app.lines[0].get_data_3d()).T[-1], app.paths[0][0][-1])
    # Save a comparison scene with one captured, one missed and one hit ray.
    app.launch(True)
    app.frame = 90
    app.tick()
    app.fig.savefig(out/'gui_preview.png', dpi=150, facecolor=app.fig.get_facecolor())
    with patch('ray_tracing_3d.filedialog.asksaveasfilename', return_value=str(out/'gui_saved.jpg')):
        app.save()
    assert Image.open(out/'gui_saved.jpg').size == (128, 128)
    assert not callbacks, callbacks
    root.destroy()
    return {'tk_creation': True, 'launch_compare': True, 'pause_resume': True,
            'reset_view': True, 'background_thread_generation': True,
            'pixel_selection_consistent': True, 'animation_terminal_point': True,
            'jpeg_save': True, 'callback_errors': callbacks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scipy', action='store_true')
    parser.add_argument('--original-dir')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[128, 256, 512])
    args = parser.parse_args()
    out = Path(__file__).resolve().parent/'validation'
    out.mkdir(exist_ok=True)
    report = {'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                               'platform': platform.platform()}}
    if args.original_dir:
        report['original_comparison'] = validate_original(args.original_dir)
        print('Original equations, RK4 and pre-interpolation residuals match.', flush=True)
    bs = [0., 4.5, 5.19, BC-2e-7, BC+2e-7, BC+5e-6, BC+8e-6, 5.2, 5.21, 5.3, 5.5, 6., 6.5,
          7., 7., 8., 10., 12., 13., 14., 15., 20.]
    b = np.asarray(bs)
    psi = np.arange(len(b)) % 2*np.pi/4
    references = [reference_ray(bj, pj) for bj, pj in zip(b, psi)] if args.scipy else None
    convergence = []
    for scale in [1., .5, .25]:
        s, st, err = trace_batch(b, r0=30, sky=120, psi=psi, scale=scale)
        if convergence:
            np.testing.assert_array_equal(st, convergence[0]['statuses'])
        row = {'scale': scale, 'statuses': st.tolist(), 'max_rk4_error': float(err.max()),
               'max_terminal_state_constraint_error': float(constraint_error(s, b).max())}
        if references:
            np.testing.assert_array_equal(st, [r[0] for r in references])
            distance = np.linalg.norm(cartesian(s, psi)-np.array([r[1] for r in references]), axis=1)
            hit = st == SCREEN_HIT
            near = np.abs(b-BC) < 1e-5
            row.update({'dop853_status_agreement': True,
                        'max_position_error': float(distance.max()),
                        'max_screen_hit_error': float(distance[hit].max()),
                        'max_position_error_away_from_critical': float(distance[~near].max()),
                        'position_errors': distance.tolist()})
        convergence.append(row)
        print('Convergence:', row, flush=True)
    report['rays'] = [{'b': float(bj), 'psi': float(pj)} for bj, pj in zip(b, psi)]
    report['convergence'] = convergence
    if references:
        assert convergence[-1]['max_screen_hit_error'] < convergence[0]['max_screen_hit_error']
        assert convergence[-1]['max_position_error'] < convergence[0]['max_position_error']
        assert convergence[-1]['max_screen_hit_error'] < 1e-6
    timings = []
    for n in args.resolutions:
        image, reference, meta = render_image(n)
        assert meta['unresolved'] == meta['numerical_failures'] == 0
        assert meta['max_error'] < 1e-7
        Image.fromarray(image).save(out/f'image_{n}.png')
        Image.fromarray(reference).save(out/f'reference_{n}.png')
        timings.append(meta)
        print('Image:', meta, flush=True)
    report['images'] = timings
    # Quantify rendering changes when halving h on the same camera grid.
    coarse, _, cm = render_image(128)
    fine, _, fm = render_image(128, scale=.5)
    report['image_step_halving_128'] = {
        'changed_rgb_pixels': int(np.any(coarse != fine, axis=-1).sum()),
        'counts_default': {k: cm[k] for k in ['captured', 'screen_hits', 'missed']},
        'counts_half': {k: fm[k] for k in ['captured', 'screen_hits', 'missed']}}
    (out/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.gui:
        report['gui'] = gui_smoke(out)
        print('Tk GUI smoke checks passed.', flush=True)
    (out/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', out/'validation.json', flush=True)


if __name__ == '__main__':
    main()
