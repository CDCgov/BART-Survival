"""
bart_survival
=============
Bayesian Additive Regression Trees (BART) for Survival Analysis.

Provides high-performance MCMC sampling with DART sparsity selection,
discrete-time probit hazard models, out-of-sample survival curves,
and causal partial dependence (G-computation) auditing.
"""

from bart_survival.version import __version__
from bart_survival import simulation
from bart_survival import surv_bart
from bart_survival.surv_bart import (
    # Primary estimator
    SurvBART,
    BartSurvModel,
    # Data transformation
    format_survival_data,
    get_surv_pre_train,
    format_prediction_data,
    get_posterior_test,
    # Survival curves & metrics
    compute_survival_curves,
    get_sv_prob,
    compute_survival_quantiles,
    get_sv_mean_quant,
    # Partial dependence & Causal G-Computation
    compute_pdp,
    get_pdp,
    compute_causal_pdp,
    get_causal_pdp,
    compute_pdp_difference,
    pdp_diff_metric,
    compute_hazard_ratio,
    pdp_rr_metric,
)
from bart_survival.simulation import (
    simulate_confounded_survival_data,
    simulate_survival,
    get_x_matrix,
)
from bart_survival.bart_adj import (
    FastSurvMHBART,
    predict_out_of_sample,
)

__all__ = [
    # Metadata & modules
    "__version__",
    "simulation",
    "surv_bart",
    # Simulation & Benchmarking
    "simulate_confounded_survival_data",
    "simulate_survival",
    "get_x_matrix",
    # Primary estimator
    "SurvBART",
    "BartSurvModel",
    # Data transformation
    "format_survival_data",
    "get_surv_pre_train",
    "format_prediction_data",
    "get_posterior_test",
    # Survival curves & metrics
    "compute_survival_curves",
    "get_sv_prob",
    "compute_survival_quantiles",
    "get_sv_mean_quant",
    # Partial dependence & Causal G-Computation
    "compute_pdp",
    "get_pdp",
    "compute_causal_pdp",
    "get_causal_pdp",
    "compute_pdp_difference",
    "pdp_diff_metric",
    "compute_hazard_ratio",
    "pdp_rr_metric",
    # Engine
    "FastSurvMHBART",
    "predict_out_of_sample",
]