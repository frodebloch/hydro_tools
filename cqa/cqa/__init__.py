"""cqa: combined capability + excursion analysis prototype.

P1 scope: linearised closed-loop covariance prediction of vessel
position/heading excursion under DP, intact state only, building an
excursion polar vs. relative weather direction.

See analysis.md in the parent directory for the full feasibility study.
"""

from .config import CqaConfig, OperationalLimits, ControllerParams, csov_default_config
from .vessel import LinearVesselModel, WindForceModel
from .controller import LinearDpController
from .psd import (
    npd_wind_gust_force_psd,
    slow_drift_force_psd_newman,
    current_variability_force_psd,
)
from .closed_loop import (
    ClosedLoop,
    lyapunov_position_covariance,
    state_covariance_freqdomain,
    position_psd,
    axis_psd,
)
from .excursion import excursion_polar, ExcursionResult
from .transient import (
    AugmentedSystem,
    WcfdiScenario,
    TransientResult,
    build_augmented_system,
    wcfdi_transient,
)
from .gangway import (
    GangwayJointState,
    OperabilityResult,
    rotation_centre_body,
    telescope_direction_body,
    tip_body,
    tip_world,
    telescope_sensitivity,
    telescope_sensitivity_6dof,
    telescope_std_dev,
    telescope_velocity_std_dev,
    evaluate_operability,
)
from .wcfdi_mc import (
    WcfdiMcResult,
    wcfdi_mc,
    starting_state_sensitivity,
)
from .wcfdi_self_mc import (
    WcfdiSelfMcResult,
    WcfdiSelfMcMatrix,
    WcfdiSelfMcMatrixCell,
    wcfdi_self_mc,
    wcfdi_self_mc_matrix,
)
from .operator_view import (
    OperatorSummary,
    summarise_for_operator,
    plot_operator_summary,
    IntactPriorSummary,
    summarise_intact_prior,
    plot_intact_prior,
)
from .rao import (
    RaoTable,
    load_pdstrip_rao,
    evaluate_rao,
    evaluate_drift,
)
from .wave_response import (
    WaveLengthResult,
    cqa_theta_rel_to_pdstrip_beta_deg,
    sigma_L_wave,
    sigma_L_wave_multimodal,
)
from .drift import (
    mean_drift_force_pdstrip,
    slow_drift_force_psd_newman_pdstrip,
)
from .sea_spreading import (
    SeaSpreading,
    cos_2s_norm_const,
    spreading_quadrature,
)
from .extreme_value import (
    RiceExceedanceResult,
    spectral_moments,
    zero_upcrossing_rate,
    variance_decorrelation_time_from_psd,
    vanmarcke_bandwidth_q,
    clh_epsilon,
    p_exceed_rice,
    p_exceed_rice_multiband,
    p_exceed_from_psd,
    inverse_rice,
    inverse_rice_multiband,
    predictive_running_max_quantile,
)
from .online_estimator import (
    SigmaPosterior,
    PosteriorHealth,
    RadialPosterior,
    ValidityBadge,
    BayesianSigmaEstimator,
    combine_radial_posterior,
    compose_validity_badge,
    closed_loop_decorrelation_time,
)
from .sea_state_relations import (
    WindSeaState,
    pm_hs_from_vw,
    pm_tp_from_vw,
    pm_sea_state,
)
from .operability_polar import (
    OperabilityPolar,
    WcfdiOperabilityOverlay,
    operability_polar,
    plot_operability_polar,
    wcfdi_operability_overlay,
)
from .time_series_realisation import (
    realise_vector_force_time_series,
    integrate_closed_loop_response,
    realise_wave_motion_6dof,
    radial_position_time_series,
    base_position_xy_time_series,
    telescope_length_deviation_time_series,
)
from .signal_processing import bandsplit_lowpass

__all__ = [
    "CqaConfig",
    "OperationalLimits",
    "ControllerParams",
    "csov_default_config",
    "LinearVesselModel",
    "WindForceModel",
    "LinearDpController",
    "npd_wind_gust_force_psd",
    "slow_drift_force_psd_newman",
    "current_variability_force_psd",
    "ClosedLoop",
    "lyapunov_position_covariance",
    "state_covariance_freqdomain",
    "position_psd",
    "axis_psd",
    "excursion_polar",
    "ExcursionResult",
    "AugmentedSystem",
    "WcfdiScenario",
    "TransientResult",
    "build_augmented_system",
    "wcfdi_transient",
    "GangwayJointState",
    "OperabilityResult",
    "rotation_centre_body",
    "telescope_direction_body",
    "tip_body",
    "tip_world",
    "telescope_sensitivity",
    "telescope_sensitivity_6dof",
    "telescope_std_dev",
    "telescope_velocity_std_dev",
    "evaluate_operability",
    "WcfdiMcResult",
    "wcfdi_mc",
    "starting_state_sensitivity",
    "WcfdiSelfMcResult",
    "WcfdiSelfMcMatrix",
    "WcfdiSelfMcMatrixCell",
    "wcfdi_self_mc",
    "wcfdi_self_mc_matrix",
    "OperatorSummary",
    "summarise_for_operator",
    "plot_operator_summary",
    "IntactPriorSummary",
    "summarise_intact_prior",
    "plot_intact_prior",
    "RaoTable",
    "load_pdstrip_rao",
    "evaluate_rao",
    "evaluate_drift",
    "WaveLengthResult",
    "cqa_theta_rel_to_pdstrip_beta_deg",
    "sigma_L_wave",
    "sigma_L_wave_multimodal",
    "mean_drift_force_pdstrip",
    "slow_drift_force_psd_newman_pdstrip",
    "SeaSpreading",
    "cos_2s_norm_const",
    "spreading_quadrature",
    "RiceExceedanceResult",
    "spectral_moments",
    "zero_upcrossing_rate",
    "variance_decorrelation_time_from_psd",
    "vanmarcke_bandwidth_q",
    "clh_epsilon",
    "p_exceed_rice",
    "p_exceed_rice_multiband",
    "p_exceed_from_psd",
    "inverse_rice",
    "inverse_rice_multiband",
    "predictive_running_max_quantile",
    "SigmaPosterior",
    "BayesianSigmaEstimator",
    "PosteriorHealth",
    "RadialPosterior",
    "combine_radial_posterior",
    "ValidityBadge",
    "compose_validity_badge",
    "closed_loop_decorrelation_time",
    "WindSeaState",
    "pm_hs_from_vw",
    "pm_tp_from_vw",
    "pm_sea_state",
    "OperabilityPolar",
    "operability_polar",
    "plot_operability_polar",
    "WcfdiOperabilityOverlay",
    "wcfdi_operability_overlay",
    "realise_vector_force_time_series",
    "integrate_closed_loop_response",
    "realise_wave_motion_6dof",
    "radial_position_time_series",
    "base_position_xy_time_series",
    "telescope_length_deviation_time_series",
    "bandsplit_lowpass",
]
