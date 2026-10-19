"""
01_quickstart_causal_survival.py
================================
End-to-end tutorial demonstrating SurvBART on a realistic synthetic clinical survival cohort.
Fully synthetic dataset with known ground-truth treatment effect, non-linear confounding,
and support audit (zero clinical PHI).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bart_survival import (
    SurvBART,
    simulate_confounded_survival_data
)


def main():
    print("=" * 70)
    print("SurvBART Quickstart: Causal Survival Analysis & Support-Audited PDP")
    print("=" * 70)

    # 1. Simulate observational clinical survival cohort
    true_hr = 1.75
    print(f"\n[1] Simulating observational cohort (N=350, True Treatment HR={true_hr})...")
    df, meta = simulate_confounded_survival_data(
        N=350,
        treatment_hr=true_hr,
        censoring_rate=0.25,
        seed=42
    )
    print(f"Cohort size: {len(df)} subjects")
    print(f"Observed event rate: {df['status'].mean():.1%}")
    print(f"Median follow-up time: {df['time'].median():.1f} days")
    print("\nFirst 5 subjects:")
    print(df[["inpatient", "age", "stage", "los", "time", "status"]].head())

    # 2. Initialize SurvBART with fast preset for rapid execution
    print("\n[2] Initializing SurvBART model...")
    # For full scientific analyses, use SurvBART.preset_standard() (50 trees, 1000 draws, 250 tune, 4 chains, thin=4)
    # Here we use preset_fast with 25 trees and 500 draws for demonstration
    model = SurvBART.preset_fast(
        trees=20,
        draws=500,
        tune=250,
        chains=2,
        thin=2,
        split_method="empirical",
        max_cutpoints=512
    )
    print(f"Model configuration: {model.trees} trees, {model.draws} draws/chain, thin={model.thin}, split={model.split_method}")

    # 3. Fit model directly from tabular data and survival outcomes
    print("\n[3] Fitting SurvBART via .fit_survival()...")
    feature_cols = ["inpatient", "age", "stage", "los"]
    X = df[feature_cols]

    model.fit_survival(
        X=X,
        time=df["time"],
        status=df["status"],
        time_scale=5.0,  # Discretize continuous time into 5-day intervals
        progressbar=True,
        random_seed=42
    )
    print(f"Model fitted successfully across {len(model.training_times_)} discrete follow-up intervals.")

    # 4. Predict out-of-sample cumulative survival curves S(t)
    print("\n[4] Predicting cumulative survival curves S(t)...")
    surv_res = model.predict_survival(X, return_quantiles=True)
    mean_s_t = surv_res["mean"]  # (n_subjects, n_times)
    times = surv_res["times"]
    print(f"Survival curve array shape: {mean_s_t.shape}")
    print(f"Overall cohort mean survival at interval {int(times[-1])}: {mean_s_t[:, -1].mean():.3f}")

    # 5. Variable inclusion feature importance
    print("\n[5] Posterior Feature Importance (variable inclusion proportions):")
    feat_imp = model.feature_importance(sort=True)
    for feat, imp in feat_imp.items():
        print(f"  {feat:12s}: {imp:.4f} ({imp * 100:.1f}%)")

    # 6. Support-Audited Causal Treatment Effect Estimation
    print("\n[6] Estimating Causal Treatment Effect for 'inpatient' (0 vs 1)...")
    causal_effect = model.estimate_effect(
        X=X,
        primary_col="inpatient",
        values=[0, 1],
        audit_support=True,
        donor_seed=42
    )

    weighted_hr = causal_effect["hazard_ratio_weighted_mean"]
    weighted_ci = causal_effect["hazard_ratio_weighted_ci"]
    rd_horizon = causal_effect["risk_difference_horizon_mean"]
    flagged = causal_effect["flagged_support_violations"]

    print(f"Flagged support violations handled: {flagged}")
    print(f"Ground-truth Hazard Ratio:  {true_hr:.2f}")
    print(f"Event-Weighted Summary HR:  {weighted_hr:.2f} (95% CI: [{weighted_ci[0]:.2f}, {weighted_ci[1]:.2f}])")
    print(f"Horizon Risk Difference:    {rd_horizon:+.2%} (Excess risk: Arm 1 - Arm 0)")

    # 7. Continuous Shift Intervention (Modified Treatment Policy)
    print("\n[7] Estimating Continuous Shift Intervention for 'age' (+5 years)...")
    shift_effect = model.estimate_effect(
        X=X,
        primary_col="age",
        delta=5.0
    )
    shift_hr = shift_effect["hazard_ratio_weighted_mean"]
    shift_ci = shift_effect["hazard_ratio_weighted_ci"]
    shift_rd = shift_effect["risk_difference_horizon_mean"]
    print(f"Shift (+5 yrs) Summary HR:  {shift_hr:.2f} (95% CI: [{shift_ci[0]:.2f}, {shift_ci[1]:.2f}])")
    print(f"Shift Horizon Risk Diff:    {shift_rd:+.2%}")

    print("\n" + "=" * 70)
    print("Tutorial execution completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

