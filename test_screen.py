"""Run: python -m unittest -v test_screen (NumPy only; no GUI needed)."""
import unittest
from unittest.mock import patch
import numpy as np
import ray_engine as engine
from ray_engine import (trace_batch, trace_path, cartesian, segment_events,
                        CAPTURED, SCREEN_HIT, MISSED, UNRESOLVED, NUMERICAL_FAILURE)
from imaging import (render_image, trajectory, screen_background, screen_grid_lines, pixel_ray,
                     MISSED_COLOR, UNRESOLVED_COLOR, FAILURE_COLOR)


class EventTests(unittest.TestCase):
    def event(self, start, end, outer=120., **kwargs):
        st, t, point = segment_events(np.array([start], float), np.array([end], float),
                                      outer, **kwargs)
        return st[0], t[0], point[0]

    def test_screen_before_outer_in_same_segment(self):
        st, t, point = self.event([-18, 0, 0], [-22, 0, 0], outer=20.)
        self.assertEqual(st, SCREEN_HIT)
        self.assertEqual(t, .25)
        np.testing.assert_array_equal(point, [-19, 0, 0])

    def test_outer_before_screen_in_same_segment(self):
        st, _, point = self.event([-18, 0, 0], [-22, 0, 0], outer=18.5)
        self.assertEqual(st, MISSED)
        np.testing.assert_array_equal(point, [-18.5, 0, 0])

    def test_horizon_before_screen_even_when_end_is_outside_horizon(self):
        st, _, point = self.event([3, 0, 0], [-20, 0, 0])
        self.assertEqual(st, CAPTURED)
        np.testing.assert_allclose(point, [2, 0, 0], atol=1e-14)

    def test_screen_edges_included_and_outside_ignored(self):
        self.assertEqual(self.event([-18, 12, -12], [-20, 12, -12])[0], SCREEN_HIT)
        self.assertEqual(self.event([-18, 12.000001, 0], [-20, 12.000001, 0])[0], UNRESOLVED)

    def test_outside_crossing_does_not_prevent_later_inside_crossing(self):
        self.assertEqual(self.event([-18, 13, 0], [-20, 13, 0])[0], UNRESOLVED)
        self.assertEqual(self.event([-20, 11, 0], [-18, 11, 0])[0], SCREEN_HIT)

    def test_interpolated_transverse_coordinates(self):
        st, t, point = self.event([-18, 0, 0], [-22, 8, -4])
        self.assertEqual(st, SCREEN_HIT)
        self.assertEqual(t, .25)
        np.testing.assert_array_equal(point, [-19, 2, -1])

    def test_screen_wins_outer_tie(self):
        self.assertEqual(self.event([-18, 0, 0], [-20, 0, 0], outer=19.)[0], SCREEN_HIT)

    def test_screen_before_horizon_in_same_segment(self):
        self.assertEqual(self.event([3, 0, 0], [1, 0, 0], screen_x=2.5)[0], SCREEN_HIT)

    def test_horizon_wins_screen_tie(self):
        self.assertEqual(self.event([3, 0, 0], [1, 0, 0], screen_x=2.)[0], CAPTURED)

    def test_parallel_and_on_screen(self):
        self.assertEqual(self.event([-18, 0, 0], [-18, 1, 0])[0], UNRESOLVED)
        st, t, _ = self.event([-19, 0, 0], [-19, 1, 0])
        self.assertEqual((st, t), (SCREEN_HIT, 0.))


class RayTests(unittest.TestCase):
    def test_radial_ray(self):
        path, st, error = trace_path(0.)
        self.assertEqual(st, CAPTURED)
        self.assertEqual(error, 0.)
        np.testing.assert_allclose(path[-1], [2, 0, 0], atol=1e-14)

    def test_square_depends_on_orientation(self):
        _, st, _ = trace_batch([7., 7.], r0=30, sky=120, psi=[0., np.pi/4])
        np.testing.assert_array_equal(st, [MISSED, SCREEN_HIT])

    def test_path_and_batch_share_endpoints_and_status(self):
        b = np.array([4.5, 5.2, 6., 7., 10., 15.])
        psi = np.array([0., .2, .4, np.pi/4, -.8, 1.1])
        s, st, error = trace_batch(b, r0=30, sky=120, psi=psi)
        for j in range(len(b)):
            path, code, err = trace_path(b[j], psi[j])
            self.assertEqual(code, st[j])
            self.assertAlmostEqual(err, error[j], places=14)
            np.testing.assert_allclose(path[-1], cartesian(s[j], psi[j]), atol=2e-12)
            events, _, _ = segment_events(path[:-1], path[1:], 120.)
            self.assertTrue(np.all(events[:-1] == UNRESOLVED))
            # Rounding may place the last point a few ulps inside a sphere.
            if code == SCREEN_HIT:
                self.assertEqual(path[-1, 0], -19.)
                self.assertLessEqual(np.max(np.abs(path[-1, 1:])), 12.)
            else:
                self.assertAlmostEqual(np.linalg.norm(path[-1]), 2. if code == CAPTURED else 120., places=10)

    def test_critical_rays_remain_classified_on_step_halving(self):
        b = engine.BC+np.array([-2e-7, 2e-7, -.01, .01])
        s1, st1, _ = trace_batch(b, r0=30, sky=120)
        _, st2, _ = trace_batch(b, r0=30, sky=120, scale=.5)
        np.testing.assert_array_equal(st1, st2)
        self.assertTrue(np.all(np.isin(st1, [CAPTURED, SCREEN_HIT, MISSED])))
        self.assertGreater(s1[1, 2], 4*np.pi)

    def test_exhaustion_is_not_a_miss(self):
        _, st, _ = trace_batch([0., 10.], r0=30, sky=120, max_steps=1)
        np.testing.assert_array_equal(st, [UNRESOLVED, UNRESOLVED])

    def test_numerical_failure_is_not_a_miss(self):
        with patch.object(engine, 'rk4', return_value=np.full((1, 3), np.nan)):
            _, st, error = trace_batch([10.], r0=30, sky=120)
        self.assertEqual(st[0], NUMERICAL_FAILURE)
        self.assertTrue(np.isinf(error[0]))

    def test_explicit_legacy_sphere_mode(self):
        s, st, _ = trace_batch([0., 6., 10.], r0=30, sky=120, screen_x=None)
        np.testing.assert_array_equal(st, [CAPTURED, SCREEN_HIT, SCREEN_HIT])
        np.testing.assert_allclose(s[:, 0], [2., 120., 120.], atol=1e-12)

    def test_invalid_initial_direction_rejected(self):
        with self.assertRaises(ValueError):
            trace_batch([100.], r0=30, sky=120)


class ImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image, cls.reference, cls.meta = render_image(33, rotation=.31)

    def test_all_pixels_accounted_for(self):
        meta = self.meta
        self.assertEqual(sum(meta[k] for k in ['captured', 'screen_hits', 'missed',
                                              'unresolved', 'numerical_failures']), 33**2)
        self.assertEqual(meta['unresolved'], 0)
        self.assertEqual(meta['numerical_failures'], 0)
        self.assertGreater(meta['captured'], 0)
        self.assertGreater(meta['screen_hits'], 0)
        self.assertGreater(meta['missed'], 0)

    def test_selected_pixels_equal_full_per_direction_traces(self):
        rows, cols = np.meshgrid(np.arange(0, 33, 4), np.arange(0, 33, 4))
        rows, cols = rows.ravel(), cols.ravel()
        b, psi = pixel_ray(cols, rows, 33, 45.)
        s, st, _ = trace_batch(b, r0=30, sky=120, psi=psi)
        xyz = cartesian(s, psi)
        expected = np.empty((len(b), 3), np.uint8)
        expected[:] = MISSED_COLOR
        hit = st == SCREEN_HIT
        expected[hit] = screen_background(xyz[hit, 1], xyz[hit, 2], .31)
        expected[st == CAPTURED] = 0
        np.testing.assert_array_equal(self.image[rows, cols], expected)
        _, label = trajectory(float(b[0]), float(psi[0]))
        self.assertEqual(label, engine.STATUS_LABELS[st[0]])

    def test_reference_is_finite_plane_with_correct_orientation(self):
        q = (np.arange(33)+.5-33/2)/(33/2)*np.tan(np.deg2rad(22.5))
        yy, zz = np.meshgrid(q*49, -q*49)
        inside = (np.abs(yy) <= 12) & (np.abs(zz) <= 12)
        np.testing.assert_array_equal(self.reference[inside], screen_background(yy[inside], zz[inside], .31))
        np.testing.assert_array_equal(self.reference[~inside], np.tile(MISSED_COLOR, ((~inside).sum(), 1)))
        np.testing.assert_array_equal(self.image[16, 16], [0, 0, 0])

    def test_unfinished_pixels_are_magenta(self):
        image, _, meta = render_image(8, max_steps=1)
        self.assertTrue(np.all(image == UNRESOLVED_COLOR))
        self.assertEqual(meta['unresolved'], 64)
        self.assertEqual(meta['missed'], 0)

    def test_failed_pixels_are_red(self):
        def broken(s, *_):
            return np.full_like(s, np.nan)
        with patch.object(engine, 'rk4', side_effect=broken):
            image, _, meta = render_image(8)
        self.assertTrue(np.all(image == FAILURE_COLOR))
        self.assertEqual(meta['numerical_failures'], 64)

    def test_both_textures_support_scalar_and_array_sampling(self):
        for grid in [True, False]:
            rgb = screen_background(1., 2., .7, grid)
            array = screen_background([1.], [2.], .7, grid)
            self.assertEqual(rgb.shape, (3,))
            np.testing.assert_array_equal(array[0], rgb)

    def test_3d_grid_matches_rotated_texture(self):
        for rotation in [0., .31, 1.7]:
            for line in screen_grid_lines(rotation):
                self.assertTrue(np.all(np.abs(line) <= 12.))
                y, z = line.mean(axis=0)
                np.testing.assert_array_equal(screen_background(y, z, rotation), [242, 244, 255])


if __name__ == '__main__':
    unittest.main()
