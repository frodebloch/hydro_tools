"""Synthetic regression test for the per-seed brucon loader's
roll/pitch/heave WF posterior plumbing
(``scripts/p7_brucon_validation/live_cell_per_seed_pwq30.build_live_sigma_posterior``).

The brucon-validation script is not on the regular cqa import path,
but the loader function is the single point that turns brucon raw
samples into a ``LiveSigmaPosterior`` for the operator panel.
A regression here would silently revert the gangway bar to its
horizontal-3DOF lower-bound mode; this test pins:

  - ``samples_wf_{heave,roll,pitch}`` are consumed and produce
    finite ``posterior_wf_{heave,roll,pitch}``.
  - The horizontal posteriors are unchanged whether or not the
    new sample channels are present (backward compatibility).
  - When the new channels are absent the posteriors stay None
    (so any downstream caller falls back to horizontal_3dof
    coverage).
  - The pinned posterior sigma_median for a synthetic 1-deg-RMS
    pitch / 0.5-deg-RMS roll / 0.3-m-RMS heave input matches the
    sample std up to the InvGamma posterior contraction (~few %
    on a 60-s window at 1 Hz).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO_ROOT / "scripts" / "p7_brucon_validation"
sys.path.insert(0, str(SCRIPT_DIR))

import live_cell_per_seed_pwq30 as lc                       # noqa: E402


def _synthetic_d(*, dt=1.0, n=120, sig_lf=0.20, sig_wf=0.20,
                 sig_yaw=math.radians(0.5), sig_heave=0.30,
                 sig_roll=math.radians(0.5), sig_pitch=math.radians(1.0),
                 with_vertical=True):
    """Build a minimal ``d`` dict matching ``load_seed``'s contract.

    Only the fields ``build_live_sigma_posterior`` reads are populated.
    Samples are zero-mean Gaussian with the requested per-channel std
    (so ``sigma_median`` from the posterior should land within a few
    percent of the input std after the InvGamma update).
    """
    rng = np.random.default_rng(0)
    out = dict(
        win_dt=dt,
        samples_lf_x=rng.standard_normal(n) * sig_lf,
        samples_lf_y=rng.standard_normal(n) * sig_lf,
        samples_lf_yaw=rng.standard_normal(n) * sig_yaw,
        samples_wf_x=rng.standard_normal(n) * sig_wf,
        samples_wf_y=rng.standard_normal(n) * sig_wf,
        samples_wf_yaw=rng.standard_normal(n) * sig_yaw,
    )
    if with_vertical:
        out["samples_wf_heave"] = rng.standard_normal(n) * sig_heave
        out["samples_wf_roll"] = rng.standard_normal(n) * sig_roll
        out["samples_wf_pitch"] = rng.standard_normal(n) * sig_pitch
    return out


def test_loader_attaches_vertical_posteriors_when_samples_present():
    d = _synthetic_d(with_vertical=True)
    post = lc.build_live_sigma_posterior(d, sigma_R_b_hat_m=0.05)

    # New WF posteriors are present and finite.
    assert post.posterior_wf_heave is not None
    assert post.posterior_wf_roll is not None
    assert post.posterior_wf_pitch is not None
    assert math.isfinite(post.posterior_wf_heave.sigma_median)
    assert math.isfinite(post.posterior_wf_roll.sigma_median)
    assert math.isfinite(post.posterior_wf_pitch.sigma_median)

    # The horizontal posteriors are unaffected by the vertical
    # plumbing (regression fence: a future refactor must not silently
    # cross-contaminate the bands).
    assert math.isfinite(post.posterior_wf_x.sigma_median)
    assert math.isfinite(post.posterior_wf_y.sigma_median)
    assert math.isfinite(post.posterior_wf_yaw.sigma_median)
    assert math.isfinite(post.posterior_lf_x.sigma_median)
    assert math.isfinite(post.posterior_lf_y.sigma_median)
    assert math.isfinite(post.posterior_lf_yaw.sigma_median)


def test_loader_omits_vertical_posteriors_when_samples_absent():
    """Backward compatibility: a ``d`` dict without the new sample
    channels (e.g. an older calibration npz pickled before the
    extension) must still produce a valid LiveSigmaPosterior with the
    vertical fields left as None, so downstream consumers fall back
    to horizontal_3dof coverage."""
    d = _synthetic_d(with_vertical=False)
    post = lc.build_live_sigma_posterior(d, sigma_R_b_hat_m=0.05)
    assert post.posterior_wf_heave is None
    assert post.posterior_wf_roll is None
    assert post.posterior_wf_pitch is None


def test_vertical_posterior_sigma_recovers_input_std():
    """With a 60-s 1-Hz window of zero-mean Gaussian samples (n=120,
    well above the prior strength n0=2), the posterior median must
    track the input std within ~10 % per channel.

    This pins the unit handling end-to-end: a regression that, e.g.,
    fed degrees instead of radians into the estimator would inflate
    the posterior by 57x and trip this test immediately."""
    sig_pitch = math.radians(1.0)
    sig_roll = math.radians(0.5)
    sig_heave = 0.30
    d = _synthetic_d(sig_pitch=sig_pitch, sig_roll=sig_roll,
                     sig_heave=sig_heave, with_vertical=True)
    post = lc.build_live_sigma_posterior(d, sigma_R_b_hat_m=0.0)
    assert post.posterior_wf_pitch.sigma_median == pytest.approx(
        sig_pitch, rel=0.10)
    assert post.posterior_wf_roll.sigma_median == pytest.approx(
        sig_roll, rel=0.15)
    assert post.posterior_wf_heave.sigma_median == pytest.approx(
        sig_heave, rel=0.10)


def test_loader_horizontal_posteriors_unchanged_by_vertical_plumbing():
    """The horizontal posteriors must be bit-identical whether or not
    the vertical channels are present. This is a regression fence
    against a future refactor accidentally cross-feeding samples."""
    d_h = _synthetic_d(with_vertical=False)
    d_f = _synthetic_d(with_vertical=True)
    p_h = lc.build_live_sigma_posterior(d_h, sigma_R_b_hat_m=0.05)
    p_f = lc.build_live_sigma_posterior(d_f, sigma_R_b_hat_m=0.05)
    for attr in ("posterior_wf_x", "posterior_wf_y", "posterior_wf_yaw",
                 "posterior_lf_x", "posterior_lf_y", "posterior_lf_yaw"):
        assert getattr(p_h, attr).sigma_median == pytest.approx(
            getattr(p_f, attr).sigma_median, abs=1e-12)
