"""Unit tests for Causal G-Computation and Support Violation Auditing.
"""

import numpy as np
import pandas as pd
from bart_survival import (
    simulate_confounded_survival_data,
    compute_causal_pdp,
    get_causal_pdp,
    compute_pdp,
    get_pdp,
    compute_pdp_difference,
    compute_hazard_ratio,
    compute_shift_pdp,
    compute_hazard_and_risk_difference,
)


def test_causal_pdp_support_audit():
    """Verify that structural positivity violations are audited and flagged correctly."""
    df, meta = simulate_confounded_survival_data(
        N=150, 
        treatment_hr=2.0, 
        include_support_violation=True, 
        n_noise_features=3, 
        seed=42
    )
    X_df = df[meta["feature_columns"]]

    # Run Causal PDP with support audit
    out_sk, c_idx, flagged = compute_causal_pdp(
        X_df, 
        primary_col="inpatient", 
        values=[0, 1], 
        audit_support=True
    )

    # 1. 'los' should be identified as violating support, and baseline confounders should NOT be flagged
    assert "los" in flagged
    assert "age" not in flagged
    assert "biomarker" not in flagged

    # 2. Output matrix dimensions: 2 counterfactual cohorts stacked
    n_obs = len(df)
    assert out_sk.shape[0] == 2 * n_obs
    assert out_sk.shape[1] == len(meta["feature_columns"])

    # 3. Structural validity: Outpatients (val=0) must have los == 0
    los_idx = meta["feature_columns"].index("los")
    inpatient_idx = meta["feature_columns"].index("inpatient")
    
    outpat_rows = out_sk[:n_obs]
    inpat_rows = out_sk[n_obs:]

    assert np.all(outpat_rows[:, inpatient_idx] == 0.0)
    assert np.all(outpat_rows[:, los_idx] == 0.0), "All counterfactual outpatients must have LOS=0"

    assert np.all(inpat_rows[:, inpatient_idx] == 1.0)
    assert np.all(inpat_rows[:, los_idx] > 0.0), "Counterfactual inpatients must have positive donor LOS"


def test_compute_pdp_dispatch():
    """Verify compute_pdp dispatches appropriately to causal vs marginal implementations."""
    df, meta = simulate_confounded_survival_data(N=60, seed=1)
    X_df = df[meta["feature_columns"]]

    # 1. Causal dispatch
    res_causal = compute_pdp(X_df, var_col=["inpatient"], values=[[0, 1]], method="causal")
    assert len(res_causal) == 3  # out_sk, c_idx, flagged
    out_sk, c_idx, flagged = res_causal
    assert "los" in flagged

    # 2. Marginal dispatch
    res_marginal = compute_pdp(X_df, var_col=["inpatient"], values=[[0, 1]], method="marginal")
    assert len(res_marginal) == 2  # out_sk, c_idx
    out_sk_m, c_idx_m = res_marginal
    assert out_sk_m.shape[0] == 2 * len(df)


def test_pdp_metrics():
    """Verify compute_pdp_difference and compute_hazard_ratio calculations."""
    n_draws = 50
    n_obs = 30
    n_times = 5
    rng = np.random.default_rng(99)

    prob = rng.uniform(0.05, 0.2, size=(n_draws, 2 * n_obs, n_times))
    # Make group 2 higher risk
    prob[:, n_obs:, :] = prob[:, n_obs:, :] * 1.5
    sv = np.cumprod(1.0 - prob, axis=2)

    pdp_dict = {"prob": prob, "sv": sv}
    
    # Difference metric
    diff_res = compute_pdp_difference(pdp_dict, idx=n_obs)
    assert "diff_m" in diff_res and "diff_q" in diff_res
    assert len(diff_res["diff_m"]) == n_times
    # Group 1 has higher survival than group 2 -> difference > 0
    assert np.all(diff_res["diff_m"] > 0)

    # Hazard Ratio metric
    hr_res = compute_hazard_ratio(pdp_dict, idx=n_obs)
    assert "rr_m" in hr_res and "rr_q" in hr_res
    assert len(hr_res["rr_m"]) == n_times
    # Group 2 has 1.5x risk -> RR > 1.0
    assert np.all(hr_res["rr_m"] > 1.0)


def test_continuous_shift_pdp():
    """Verify compute_shift_pdp generates correct natural course vs shifted counterfactual data."""
    df, meta = simulate_confounded_survival_data(N=40, seed=12)
    X_df = df[meta["feature_columns"]]

    out_sk, c_idx = compute_shift_pdp(X_df, primary_col="age", delta=5.0)
    assert out_sk.shape == (2 * len(df), len(meta["feature_columns"]))
    age_idx = meta["feature_columns"].index("age")

    # Arm 0: natural course
    arm0_age = out_sk[:len(df), age_idx]
    assert np.allclose(arm0_age, X_df["age"].values)

    # Arm 1: shifted by +5.0
    arm1_age = out_sk[len(df):, age_idx]
    assert np.allclose(arm1_age, X_df["age"].values + 5.0)


def test_hazard_and_risk_difference_metrics():
    """Verify compute_hazard_and_risk_difference with weighted and unweighted HRs and RDs."""
    n_draws = 40
    n_obs = 30
    n_times = 4
    rng = np.random.default_rng(42)

    # Arm 0 has baseline hazard ~0.1
    h0 = rng.uniform(0.08, 0.12, size=(n_draws, n_obs, n_times))
    # Arm 1 has higher hazard ~0.2 (HR ~ 2.0)
    h1 = rng.uniform(0.18, 0.22, size=(n_draws, n_obs, n_times))

    prob = np.concatenate([h0, h1], axis=1)  # shape: (n_draws, 2 * n_obs, n_times)
    sv = np.cumprod(1.0 - prob, axis=2)
    pdp_dict = {"prob": prob, "sv": sv}

    event_weights = np.array([0.4, 0.3, 0.2, 0.1])
    res = compute_hazard_and_risk_difference(
        pdp_val=pdp_dict,
        idx=n_obs,
        event_weights=event_weights
    )

    # 1. Summary Hazard Ratios (both weighted and unweighted should be ~2.0)
    assert "hazard_ratio_weighted_mean" in res
    assert "hazard_ratio_unweighted_mean" in res
    assert 1.5 < res["hazard_ratio_weighted_mean"] < 2.5
    assert 1.5 < res["hazard_ratio_unweighted_mean"] < 2.5
    assert len(res["hazard_ratio_weighted_ci"]) == 2

    # 2. Discrete risk ratio (raw, not averaged)
    assert "discrete_risk_ratio_mean" in res
    assert len(res["discrete_risk_ratio_mean"]) == n_times
    assert np.all(res["discrete_risk_ratio_mean"] > 1.0)

    # 3. Risk Difference (1 over 0: excess risk > 0)
    assert "risk_difference_mean" in res
    assert len(res["risk_difference_mean"]) == n_times
    assert np.all(res["risk_difference_mean"] > 0.0)
    assert "risk_difference_weighted_mean" in res
    assert res["risk_difference_weighted_mean"] > 0.0

    # 4. Survival difference (S1 - S0 < 0)
    assert "survival_difference_mean" in res
    assert np.all(res["survival_difference_mean"] < 0.0)

    # 5. RMST difference (< 0 because group 1 has worse survival)
    assert "rmst_difference_mean" in res
    assert res["rmst_difference_mean"] < 0.0


if __name__ == "__main__":
    test_causal_pdp_support_audit()
    test_compute_pdp_dispatch()
    test_pdp_metrics()
    test_continuous_shift_pdp()
    test_hazard_and_risk_difference_metrics()
    print("All causal PDP tests passed successfully!")
