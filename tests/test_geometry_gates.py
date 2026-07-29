"""Geometry vs analytic answers; gates on deliberately broken scenes."""

import numpy as np
import pytest

from figlib.geometry import exp_partial_sums
from figlib.scene import Curve, MathLabel, Scene
from figlib.style import DEFAULT_STYLE, Role
from figlib.gates import mechanical, numerical


class TestExpPartialSums:
    def test_converges_to_exp_on_imaginary_axis(self):
        theta = 2.0
        pts = exp_partial_sums(1j * theta, n_terms=25)
        target = np.exp(1j * theta)
        assert pts[-1, 0] == pytest.approx(target.real, abs=1e-9)
        assert pts[-1, 1] == pytest.approx(target.imag, abs=1e-9)

    def test_starts_at_origin_first_step_is_one(self):
        pts = exp_partial_sums(1j * 2.0, n_terms=5)
        assert pts[0] == pytest.approx([0.0, 0.0])
        assert pts[1] == pytest.approx([1.0, 0.0])

    def test_real_argument_stays_on_real_axis(self):
        pts = exp_partial_sums(2.0 + 0j, n_terms=10)
        assert np.max(np.abs(pts[:, 1])) == 0.0

    def test_consecutive_steps_turn_by_right_angle_for_imaginary_z(self):
        # step_k = (iθ)^k/k!; arg ratio between consecutive steps is i·θ/(k+1) → 90°
        pts = exp_partial_sums(1j * 2.0, n_terms=6)
        steps = np.diff(pts, axis=0)
        z = steps[:, 0] + 1j * steps[:, 1]
        angles = np.angle(z[1:] / z[:-1])
        assert np.allclose(np.abs(angles), np.pi / 2, atol=1e-12)


class TestMechanicalGate:
    def test_overlapping_labels_flagged(self):
        scene = Scene(items=[
            Curve(np.array([[0.0, 0.0], [1.0, 1.0]]), role=Role.CONTENT),
            MathLabel(r"e^{i\theta}", (0.5, 0.5)),
            MathLabel(r"e^{i\phi}", (0.5, 0.5)),
        ])
        diags = mechanical(scene, DEFAULT_STYLE)
        assert any(d.kind == "label-collision" for d in diags)

    def test_clipped_label_flagged(self):
        scene = Scene(
            items=[
                Curve(np.array([[0.0, 0.0], [1.0, 1.0]]), role=Role.CONTENT),
                MathLabel(r"\int_0^\infty e^{-x^2}dx", (1.0, 1.0), ha="left"),
            ],
            xlim=(0, 1), ylim=(0, 1),
        )
        diags = mechanical(scene, DEFAULT_STYLE)
        assert any(d.kind == "clipped" for d in diags)

    def test_clean_scene_passes(self):
        scene = Scene(items=[
            Curve(np.array([[0.0, 0.0], [1.0, 1.0]]), role=Role.CONTENT),
            MathLabel(r"x", (0.2, 0.6)),
            MathLabel(r"y", (0.7, 0.2)),
        ])
        assert mechanical(scene, DEFAULT_STYLE) == []


class TestNumericalGate:
    def test_passing_assertions(self):
        assert numerical(lambda: None) == []

    def test_failing_assertion_captured(self):
        def bad():
            assert 1 == 2, "fixed point is not fixed"
        diags = numerical(bad)
        assert len(diags) == 1 and diags[0].kind == "numerical"
        assert "fixed point" in diags[0].detail


class TestDiffusion:
    def setup_method(self):
        from figlib.geometry import Mixture1D
        self.m0 = Mixture1D(np.array([0.5, 0.5]), np.array([-1.8, 1.5]), np.array([0.12, 0.12]))

    def test_marginal_at_large_t_is_standard_normal(self):
        from figlib.geometry import ou_marginal
        mT = ou_marginal(self.m0, 6.0)
        assert np.allclose(mT.means, 0, atol=1e-2)
        assert np.allclose(mT.sigmas, 1, atol=1e-2)

    def test_score_of_single_gaussian_is_linear(self):
        from figlib.geometry import Mixture1D, mixture_score
        g = Mixture1D(np.array([1.0]), np.array([0.5]), np.array([2.0]))
        x = np.array([-1.0, 0.0, 3.0])
        assert np.allclose(mixture_score(g, x), (0.5 - x) / 4.0)

    def test_ode_transport_reproduces_data_distribution(self):
        from figlib.geometry import ou_marginal, pf_ode_paths, sample_mixture
        rng = np.random.default_rng(0)
        T = 2.5
        x_T = sample_mixture(ou_marginal(self.m0, T), 800, rng)
        _, paths = pf_ode_paths(self.m0, x_T, T, 200)
        end = paths[:, -1]
        left = end < 0
        assert abs(left.mean() - 0.5) < 0.05
        assert abs(end[left].mean() - (-1.8)) < 0.08
        assert abs(end[~left].mean() - 1.5) < 0.08

    def test_sde_transport_reproduces_data_distribution(self):
        from figlib.geometry import ou_marginal, reverse_sde_paths, sample_mixture
        rng = np.random.default_rng(1)
        T = 2.5
        x_T = sample_mixture(ou_marginal(self.m0, T), 1500, rng)
        _, paths = reverse_sde_paths(self.m0, x_T, T, 400, rng)
        end = paths[:, -1]
        left = end < 0
        assert abs(left.mean() - 0.5) < 0.06
        assert abs(end[left].mean() - (-1.8)) < 0.1
        assert abs(end[~left].mean() - 1.5) < 0.1
