"""Observer-augmented closed-loop linearisation (3-DOF, brucon-aligned).

Extends the 12-state augmented system in `cqa.transient` with the
brucon Fossen passive observer (per DOF: ŷ_LF, ν̂) and Saelid/Jensen
2nd-order wave filter (per DOF: ξ_w, η̂_w), giving a 24-state
linearisation that closes the residual σ_y_LF gap to brucon (sandbox
0.67 m vs brucon 0.69 m at the §12.20 test sea state, vs 0.46 m for
the 12-state aug and 0.25 m for the bare 6-state PD model).

Optionally adds 3 more PI integrator states (one per DOF) → 27-state.

State ordering
--------------
    idx  0..2  : eta = [n, e, psi]              -- true position
    idx  3..5  : nu  = [u, v, r]                -- true velocity
    idx  6..8  : b_hat                          -- observer bias estimate
    idx  9..11 : tau_thr                        -- 1st-order thruster output
    idx 12..14 : eta_hat_LF                     -- LF position estimate
    idx 15..17 : nu_hat                         -- LF velocity estimate
    idx 18..20 : xi_w                           -- wave-filter integrator
    idx 21..23 : eta_hat_w                      -- wave-filter output
    [optional]
    idx 24..26 : I                              -- PI integrator on ŷ_LF

Brucon equations (per DOF)
--------------------------
    e        = y_meas − ŷ_LF − η̂_w              (wave-corrected innovation)
    y_meas   = η_true + η_w_true                 (sensor sees LF + WF)
    ŷ_LF_dot = ν̂ + ω_c · e
    ν̂_dot    = (1/M)·(b̂ + u_cmd − D·ν̂) + K_a1·e
    b̂_dot    = -(1/T_b)·b̂ + K_b1·e
    ξ_w_dot  = η̂_w + k1_f · e
    η̂_w_dot  = -ω_w² · ξ_w − 2 ζ_n ω_w · η̂_w + k2_f · e
    τ_thr_dot = (1/T_thr)·(u_cmd − τ_thr)
    u_cmd    = -Kp·ŷ_LF − Kd·ν̂ − b̂   [− Ki·I if integrator enabled]
    M·ν_dot  = -D·ν + τ_thr + F_drift            (vessel)
    η_dot    = ν

with k1_f = -2 (1 - ζ_n) ω_c / ω_w, k2_f = 2 (1 - ζ_n) ω_w, and
ω_w = 2π / Tp, ζ_n = `wave_filter_zeta_n(Tp)`.

The wave-correction-free linearisation here drives the system with
F_drift only (B_w injected into ν); the true HF motion η_w_true
appears as an exogenous input to the innovation chain (B_wf), and
when integrating against an LF-only PSD (slow drift) is set to zero.
For combined LF+WF analysis, see `state_psd_with_wave_input` below.

Validation
----------
Sway-only block (all surge/yaw rows/cols zeroed by setting their
controller and observer gains to zero) reproduces the per-DOF
sandbox σ_y_LF at HS=4.20 m, Tp=10.22 s, β=90° to within 5 %.
See `tests/test_observer.py`.

References
----------
- §12.20 of analysis.md (sandbox closure of σ_y_LF gap).
- ~/src/brucon/build/bin/config_csov/observer.prototxt (per-DOF gains).
- `scripts/p7_brucon_validation/sandbox_passive_observer.py` (per-DOF
  scalar reference implementation).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .vessel import LinearVesselModel
from .controller import LinearDpController
from .config import ObserverParams, wave_filter_zeta_n


@dataclass
class ObserverAugmentedSystem:
    """24- or 27-state observer-augmented linear closed-loop system.

    Attributes
    ----------
    A : (n, n) state-transition matrix.
    B_w : (n, 3) disturbance-force input (slow-drift / wind / current
        force; enters via M⁻¹ on the ν rows only).
    B_wf : (n, 3) true wave-frequency motion input (η_w_true; enters
        the innovation chain only — observer/wave-filter/bias rows).
        For LF-only analyses set the wave-frequency excitation to zero
        and integrate against B_w only.
    M, Kp, Kd : retained for projections / clipping logic.
    Ki : 3x3 (zero unless `include_integrator=True`).
    T_b, T_thr : scalar time constants (kept for diagnostic).
    n_state : 24 (no integrator) or 27 (with integrator).
    include_integrator : whether the PI integrator block is present.
    """

    A: np.ndarray
    B_w: np.ndarray
    B_wf: np.ndarray
    M: np.ndarray
    Kp: np.ndarray
    Kd: np.ndarray
    Ki: np.ndarray
    T_b: float
    T_thr: float
    n_state: int
    include_integrator: bool


def _slc(start: int) -> slice:
    return slice(start, start + 3)


# Block start indices (constant; integrator block appended at index 24)
_IDX_ETA = 0
_IDX_NU = 3
_IDX_BHAT = 6
_IDX_TAU = 9
_IDX_YHAT = 12
_IDX_VHAT = 15
_IDX_XIW = 18
_IDX_EWHAT = 21
_IDX_INT = 24


def build_observer_augmented_system(
    vessel: LinearVesselModel,
    controller: LinearDpController,
    observer: ObserverParams,
    Tp: float,
    T_thr: float = 5.0,
    include_integrator: bool = True,
    Ki_factor: float = 0.1,
) -> ObserverAugmentedSystem:
    """Assemble the observer-augmented A, B_w, B_wf for a given Tp.

    Parameters
    ----------
    vessel, controller : per `cqa.vessel` / `cqa.controller`. The
        controller's `Kp`, `Kd` are used directly; the observer model's
        velocity equation also uses them via `u_cmd`.
    observer : per-DOF observer gains and ω_c (`cqa.config.ObserverParams`).
    Tp : sea-state peak period [s] used to build ω_w = 2π/Tp and the
        gain-scheduled ζ_n via `wave_filter_zeta_n`. Brucon applies the
        same Tp to all 4 DOFs (PITCH-driven estimator); cqa follows.
    T_thr : 1st-order thruster lag time constant [s]. CSOV default 5 s
        per `cqa.config.ControllerParams.thruster_time_constant_s`.
    include_integrator : if True, append 3 PI integrator states with
        I_dot = ŷ_LF and contribution -Ki·I to u_cmd.
    Ki_factor : per-DOF Ki = Ki_factor · ω_n · Kp (brucon convention is
        0.1 · ω_n · Kp; sandbox uses the same).

    Returns
    -------
    ObserverAugmentedSystem
    """
    M = vessel.M
    D = vessel.D
    Minv = np.linalg.inv(M)
    Kp = controller.Kp
    Kd = controller.Kd

    Ka1 = np.diag(observer.K_a1)
    Kb1 = np.diag(observer.K_b1)
    omega_c = observer.omega_c
    T_b = observer.bias_time_constant_s

    omega_w = 2.0 * np.pi / Tp
    zeta_n = wave_filter_zeta_n(Tp)
    k1f = -2.0 * (1.0 - zeta_n) * omega_c / omega_w
    k2f = 2.0 * (1.0 - zeta_n) * omega_w

    # Integrator gains (per DOF, diagonal). Active only when
    # include_integrator=True; otherwise Ki returned for diagnostic.
    omega_n = np.diag(np.array([
        controller.Kp[0, 0] / max(M[0, 0], 1e-12),
        controller.Kp[1, 1] / max(M[1, 1], 1e-12),
        controller.Kp[2, 2] / max(M[2, 2], 1e-12),
    ]))
    omega_n_diag = np.sqrt(np.maximum(np.diag(omega_n), 0.0))
    Ki_vec = Ki_factor * omega_n_diag * np.diag(Kp)
    Ki = np.diag(Ki_vec)

    n_state = 27 if include_integrator else 24
    A = np.zeros((n_state, n_state))
    B_w = np.zeros((n_state, 3))
    B_wf = np.zeros((n_state, 3))

    I3 = np.eye(3)

    # -- vessel: η_dot = ν, M ν_dot = -D ν + τ_thr (+ F_drift via B_w) --
    A[_slc(_IDX_ETA), _slc(_IDX_NU)] = I3
    A[_slc(_IDX_NU), _slc(_IDX_NU)] = -Minv @ D
    A[_slc(_IDX_NU), _slc(_IDX_TAU)] = Minv
    B_w[_slc(_IDX_NU), :] = Minv

    # -- innovation chain helpers --
    # e_i = y_meas_i − ŷ_LF_i − η̂_w_i  with y_meas = η_true + η_w_true
    # For any block whose RHS contains  G·e :
    #   row += G·η_true (state cols 0..2)
    #   row -= G·ŷ_LF   (state cols 12..14)
    #   row -= G·η̂_w   (state cols 21..23)
    #   B_wf[row, :] += G                         (true HF input)
    def _add_innov(rows: slice, G: np.ndarray) -> None:
        A[rows, _slc(_IDX_ETA)] += G
        A[rows, _slc(_IDX_YHAT)] += -G
        A[rows, _slc(_IDX_EWHAT)] += -G
        B_wf[rows, :] += G

    # -- u_cmd builder (writes -Kp·ŷ_LF − Kd·ν̂ − b̂ [− Ki·I] into a row block,
    # scaled by `scale`) --
    def _add_u_cmd(rows: slice, scale: float | np.ndarray) -> None:
        if np.isscalar(scale):
            s = float(scale)
            A[rows, _slc(_IDX_YHAT)] += -s * Kp
            A[rows, _slc(_IDX_VHAT)] += -s * Kd
            A[rows, _slc(_IDX_BHAT)] += -s * I3
            if include_integrator:
                A[rows, _slc(_IDX_INT)] += -s * Ki
        else:
            S = scale  # 3x3 matrix
            A[rows, _slc(_IDX_YHAT)] += -S @ Kp
            A[rows, _slc(_IDX_VHAT)] += -S @ Kd
            A[rows, _slc(_IDX_BHAT)] += -S
            if include_integrator:
                A[rows, _slc(_IDX_INT)] += -S @ Ki

    # -- bias estimate: b̂_dot = -(1/T_b) b̂ + K_b1 · e --
    A[_slc(_IDX_BHAT), _slc(_IDX_BHAT)] += -(1.0 / T_b) * I3
    _add_innov(_slc(_IDX_BHAT), Kb1)

    # -- thruster lag: τ_thr_dot = (1/T_thr) · (u_cmd − τ_thr) --
    A[_slc(_IDX_TAU), _slc(_IDX_TAU)] += -(1.0 / T_thr) * I3
    _add_u_cmd(_slc(_IDX_TAU), 1.0 / T_thr)

    # -- LF observer position: ŷ_LF_dot = ν̂ + ω_c · e --
    A[_slc(_IDX_YHAT), _slc(_IDX_VHAT)] += I3
    _add_innov(_slc(_IDX_YHAT), omega_c * I3)

    # -- LF observer velocity: ν̂_dot = (1/M)(b̂ + u_cmd - D·ν̂) + K_a1·e --
    A[_slc(_IDX_VHAT), _slc(_IDX_BHAT)] += Minv
    A[_slc(_IDX_VHAT), _slc(_IDX_VHAT)] += -Minv @ D
    _add_u_cmd(_slc(_IDX_VHAT), Minv)
    _add_innov(_slc(_IDX_VHAT), Ka1)

    # -- wave-filter ξ_w: ξ_w_dot = η̂_w + k1_f · e --
    A[_slc(_IDX_XIW), _slc(_IDX_EWHAT)] += I3
    _add_innov(_slc(_IDX_XIW), k1f * I3)

    # -- wave-filter η̂_w: η̂_w_dot = -ω_w² ξ_w − 2ζ_n ω_w η̂_w + k2_f · e --
    A[_slc(_IDX_EWHAT), _slc(_IDX_XIW)] += -(omega_w ** 2) * I3
    A[_slc(_IDX_EWHAT), _slc(_IDX_EWHAT)] += -2.0 * zeta_n * omega_w * I3
    _add_innov(_slc(_IDX_EWHAT), k2f * I3)

    # -- integrator: I_dot = ŷ_LF --
    if include_integrator:
        A[_slc(_IDX_INT), _slc(_IDX_YHAT)] += I3

    return ObserverAugmentedSystem(
        A=A,
        B_w=B_w,
        B_wf=B_wf,
        M=M,
        Kp=Kp,
        Kd=Kd,
        Ki=Ki,
        T_b=T_b,
        T_thr=T_thr,
        n_state=n_state,
        include_integrator=include_integrator,
    )


def position_state_indices(
    aug: ObserverAugmentedSystem | None = None,
) -> dict[str, slice]:
    """Convenience map of state-block names to slice objects.

    If ``aug`` is provided and ``aug.include_integrator`` is False,
    the ``I`` entry is omitted (the underlying state vector is only
    24-dim, so ``slice(24, 27)`` would be out of range).
    """
    blocks = {
        "eta": _slc(_IDX_ETA),
        "nu": _slc(_IDX_NU),
        "b_hat": _slc(_IDX_BHAT),
        "tau_thr": _slc(_IDX_TAU),
        "eta_hat_LF": _slc(_IDX_YHAT),
        "nu_hat": _slc(_IDX_VHAT),
        "xi_w": _slc(_IDX_XIW),
        "eta_hat_w": _slc(_IDX_EWHAT),
    }
    if aug is None or aug.include_integrator:
        blocks["I"] = _slc(_IDX_INT)
    return blocks
