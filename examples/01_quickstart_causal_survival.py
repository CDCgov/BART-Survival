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
    inpatient_hr = 1.75
    print(f"\n[1] Simulating observational cohort (N=350, Inpatient Direct HR={inpatient_hr})...")
    df, meta = simulate_confounded_survival_data(
        N=350,
        inpatient_hr=inpatient_hr,
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

    total_marginal_hr = causal_effect["total_marginal_hr"]
    total_marginal_ci = causal_effect["total_marginal_hr_ci"]
    total_conditional_hr = causal_effect["total_conditional_hr"]
    total_conditional_ci = causal_effect["total_conditional_hr_ci"]
    rd_horizon = causal_effect["risk_difference_horizon_mean"]
    flagged = causal_effect["flagged_support_violations"]

    print(f"Flagged support violations handled: {flagged}")
    print("\n" + "=" * 68)
    print("Causal Treatment Effect Decomposition:")
    print(f"  • Direct Inpatient HR (inpatient_hr):            {meta['inpatient_hr']:.2f} (holding LOS fixed)")
    print(f"  • Mediated LOS Excess Hazard Multiplier:        {meta['los_hazard_multiplier']:.2f}x (via positive hospital stay)")
    print(f"  • Analytical Expected Total Conditional HR:     {meta['expected_total_causal_hr']:.2f}")
    print(f"  • Analytical Expected Population Marginal HR:   {meta['expected_marginal_hr']:.2f}")
    print("-" * 68)
    print(f"  • SurvBART Total Marginal HR (Policy Effect):   {total_marginal_hr:.2f} (95% CI: [{total_marginal_ci[0]:.2f}, {total_marginal_ci[1]:.2f}])")
    print(f"  • SurvBART Total Conditional HR (Subject HR):   {total_conditional_hr:.2f} (95% CI: [{total_conditional_ci[0]:.2f}, {total_conditional_ci[1]:.2f}])")
    print(f"  • Horizon Risk Difference:                      {rd_horizon:+.2%} (Excess risk: Arm 1 - Arm 0)")
    print("=" * 68)
    print("Methodological Note on Causal Mediation & Positivity Audit:")
    print("  SurvBART's Support-Audited G-Computation evaluates the TOTAL CAUSAL EFFECT of the")
    print("  hospitalization pathway. Outpatients have LOS=0 while inpatients experience LOS > 0.")
    print("  Intervening to assign Inpatient status also assigns the patient a hospital stay, which")
    print(f"  contributes an additional {meta['los_hazard_multiplier']:.2f}x hazard. The estimated summary HR correctly")
    print(f"  captures the expected population marginal HR ({meta['expected_marginal_hr']:.2f}) and total causal HR ({meta['expected_total_causal_hr']:.2f}),")
    print(f"  rather than merely the conditional direct effect ({meta['inpatient_hr']:.2f}) that artificially holds LOS at 0.")
    print("=" * 68)

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

