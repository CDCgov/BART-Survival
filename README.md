
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07213/status.svg)](https://doi.org/10.21105/joss.07213)


## Overview

`BART-Survival` is a high-performance Python package for discrete-time survival analysis using Bayesian Additive Regression Trees (BART). Combining the power and flexibility of PyMC probabilistic programming with specialized discrete-time hazard modeling, `BART-Survival` enables fully non-parametric time-to-event estimation, non-linear covariate interaction modeling, and causal effect estimation without requiring proportional hazards or linearity assumptions.

### Key Highlights (2026 Release)

- **Intuitive High-Level API (`SurvBART`)**: End-to-end scikit-learn compatible workflow with `fit_survival()`, `predict_survival()`, `feature_importance()`, and `estimate_effect()`.
- **Standardized Production Presets**: Battle-tested presets (`preset_production()`, `preset_fast()`, `preset_high_precision()`) calibrated for MCMC stability and rapid prototyping.
- **Empirical Node Value Splitting**: Proposes tree split cutpoints at exact empirical midpoints (up to 512 candidates per feature), capturing step-function risk dynamics with high fidelity.
- **Support-Audited Causal G-Computation**: Automated detection of conditional support and positivity violations (e.g., zero length-of-stay for outpatients) with conditional donor resampling to eliminate off-support extrapolation.
- **Warm-Started DART Dirichlet Sparsity**: Optional variable selection prior to identify true risk drivers in high-dimensional tabular cohorts.
- **Fully Accessible Hyperparameters**: Direct, open parameter exposure on `SurvBART` with property bindings and InferenceData serialization.
- **Synthetic Survival Benchmarking**: Built-in data generators (`simulate_confounded_survival_data`) for reproducible testing with known ground-truth hazard ratios.
- **100% Backward Compatible**: Full compatibility with legacy `BartSurvModel`, `get_surv_pre_train`, and `get_posterior_test` pipelines.

---

## Installation

Install from PyPI:
```bash
pip install bart-survival
```

Or install the latest development branch locally:
```bash
git clone https://github.com/CDCgov/BART-Survival.git
cd BART-Survival
pip install -e .
```

---

## Quickstart

### 1. High-Level End-to-End Workflow

```python
import numpy as np
import pandas as pd
from bart_survival import SurvBART, simulate_confounded_survival_data

# 1. Generate realistic synthetic cohort with known ground-truth treatment effect
df, meta = simulate_confounded_survival_data(
    N=400, 
    treatment_hr=1.75, 
    censoring_rate=0.25, 
    seed=42
)
feature_cols = ["inpatient", "age", "stage", "los"]
X = df[feature_cols]

# 2. Initialize SurvBART with production preset
# (50 trees, 2500 draws, 1000 tune, 4 chains, thin=10, empirical 512 cutpoints)
model = SurvBART.preset_production()

# 3. Fit directly from tabular covariates and time-to-event outcomes
model.fit_survival(
    X=X,
    time=df["time"],
    status=df["status"],
    time_scale=5.0,     # Discretize continuous time into 5-day intervals
    progressbar=True,
    random_seed=42
)

# 4. Predict cumulative survival curves S(t) with 95% Credible Intervals
surv_summary = model.predict_survival(X, return_quantiles=True)
print("Survival curve shape (subjects x times):", surv_summary["mean"].shape)
print("Mean S(t) at final interval:", surv_summary["mean"][:, -1].mean())

# 5. Extract normalized posterior feature importance
print("\nFeature Split Importances:")
print(model.feature_importance())

# 6. Estimate Causal Treatment Effect (Support-Audited G-Computation)
# Audits conditional support for positivity violations (e.g. inpatient vs LOS)
causal_res = model.estimate_effect(
    X=X, 
    primary_col="inpatient", 
    values=[0, 1]
)
print("\nEvent-Weighted Summary HR:", causal_res["hazard_ratio_weighted_mean"])
print("95% Credible Interval:     ", causal_res["hazard_ratio_weighted_ci"])
print("Horizon Risk Difference:   ", causal_res["risk_difference_horizon_mean"])
print("Flagged Support Violations:", causal_res["flagged_support_violations"])

# 7. Continuous Shift Intervention (Modified Treatment Policy: Xi -> Xi + delta)
shift_res = model.estimate_effect(
    X=X,
    primary_col="age",
    delta=5.0  # Shift everyone's age by +5 years
)
print("\nAge +5yr Shift Summary HR: ", shift_res["hazard_ratio_weighted_mean"])
print("Age +5yr Risk Difference:  ", shift_res["risk_difference_horizon_mean"])
```

---

### 2. Standard Model Presets

`SurvBART` provides three pre-calibrated factory presets:

| Preset | Trees | Draws | Tune | Chains | Thin | Update Fraction | Use Case |
|---|---|---|---|---|---|---|---|
| `preset_production()` *(Default)* | 50 | 2500 | 1000 | 4 | 10 | 1.0 (100%) | Final study models, publication-grade convergence |
| `preset_fast()` | 20 | 500 | 250 | 2 | 5 | 1.0 (100%) | Fast exploration, interactive debugging, prototyping |
| `preset_high_precision()` | 100 | 5000 | 2000 | 4 | 10 | 1.0 (100%) | High-dimensional data, complex multi-way interactions |

```python
# Launch fast exploration model
fast_model = SurvBART.preset_fast(split_method="empirical")

# Or instantiate with explicit custom parameters
custom_model = SurvBART(
    trees=50,
    draws=2000,
    tune=800,
    chains=4,
    thin=10,
    update_fraction=1.0,
    use_sparsity=False,       # Set True to enable DART Dirichlet selection
    split_method="empirical", # Proposes empirical midpoints up to max_cutpoints
    max_cutpoints=512,
    min_node_size=5
)
```

---

### 3. Step-by-Step Pooled Cross-Section API (Granular Control)

For researchers requiring explicit control over pooled person-time representations:

```python
from bart_survival import (
    SurvBART,
    format_survival_data,
    format_prediction_data,
    compute_survival_curves,
    compute_causal_pdp,
    compute_hazard_ratio
)
from lifelines.datasets import load_rossi

# Load open Rossi recidivism benchmark cohort
rossi = load_rossi()
X_df = rossi[["fin", "age", "race", "wexp", "mar", "paro", "prio"]]
time = rossi["week"].values
status = rossi["arrest"].values

# 1. Format pooled person-time training cross-section
trn = format_survival_data(
    y_time=time,
    y_status=status,
    x=X_df.values,
    time_scale=4.0  # 4-week discrete intervals
)

# 2. Format out-of-sample prediction grid
post_test = format_prediction_data(
    y_time=time,
    y_status=status,
    x=X_df.values,
    time_scale=4.0
)

# 3. Fit SurvBART
model = SurvBART(trees=50, draws=1000, tune=500, chains=2, thin=5)
model.fit(
    y=trn["y"],
    X=trn["x"],
    coords=trn["coord"],
    progressbar=False,
    random_seed=42
)

# 4. Predict out-of-sample hazards and derive cumulative survival curves S(t)
post_preds = model.predict(post_test["post_x"], coords=post_test["coords"])
sv_curves = compute_survival_curves(post_preds, coords=post_test["coords"])

# S(t) shape: (n_draws, n_subjects, n_time_intervals)
print("Survival array shape:", sv_curves["sv"].shape)
```

---

---

## Causal G-Computation and Counterfactual Treatment Effects

### 1. Support-Audited Binary G-Computation
In observational clinical survival studies, counterfactual interventions often suffer from **positivity violations** or **structural zero dependencies**. For example, assigning `inpatient=0` (outpatient) while leaving `length_of_stay=12 days` creates non-existent clinical combinations that force tree ensembles into off-support extrapolation.

`SurvBART` addresses this via `estimate_effect()` and `compute_causal_pdp()`:
1. **Positivity & Support Audit**: Automatically scans baseline covariates conditional on exposure levels to identify deterministic zeros, non-overlapping domains, or severe mass-ratio violations.
2. **Conditional Donor Resampling**: Re-samples dependent features conditional on exposure from observed donor subcohorts while preserving genuine baseline confounders.
3. **Counterfactual Predictions**: Evaluates both intervention branches through the posterior tree ensemble to yield unconfounded survival curves $S_1(t)$ and $S_0(t)$.

### 2. Continuous Shift Interventions (Modified Treatment Policies)
For continuous exposures (e.g., biomarker levels, blood pressure, age, radiation dose), arbitrary fixed point comparisons (e.g., $X=1$ vs $X=5$) frequently cause extreme positivity violations. `SurvBART` supports **shift interventions** (Modified Treatment Policies):
$$\tilde{X}_i = X_i + \Delta \quad \text{vs.} \quad X_i$$
Evaluating a shift preserves empirical covariate correlations and natural population distributions while estimating realistic, policy-relevant counterfactual effects.

### 3. Hazard Ratios and Risk Differences
`model.estimate_effect(...)` and `compute_hazard_and_risk_difference(...)` compute comprehensive effect metrics (always formatted as "Intervention (1) over Reference (0)"):
- **Event-Weighted Summary Hazard Ratio (`hazard_ratio_weighted_mean`)**: The standard summary HR, calculated as the geometric mean across follow-up intervals weighted by empirical event counts ($w_t = d_t / \sum d_t$). Matches the Cox proportional hazards estimand.
- **Time-Varying Hazard Ratio Curve (`hazard_ratio_curve_mean`)**: Interval-specific hazard ratio computed via the complementary log-log link: $\mathrm{HR}(t) = \log(1 - \bar{h}_1(t)) / \log(1 - \bar{h}_0(t))$.
- **Discrete Risk Ratio (`discrete_risk_ratio_mean`)**: Raw, interval-by-interval discrete risk ratio $\bar{h}_1(t) / \bar{h}_0(t)$ without temporal averaging.
- **Risk Difference (`risk_difference_mean`, `risk_difference_horizon_mean`)**: Excess cumulative failure risk: $\mathrm{RD}(t) = (1 - S_1(t)) - (1 - S_0(t)) = S_0(t) - S_1(t)$.
- **Restricted Mean Survival Time Difference (`rmst_difference_mean`)**: Area between the survival curves over the follow-up horizon.

---

## Documentation

Comprehensive documentation, API references, and mathematical methodology:
- [BART-Survival Documentation](https://cdcgov.github.io/BART-Survival/)
- [API Reference](https://cdcgov.github.io/BART-Survival/build/html/api.html)
- [JOSS Paper](https://doi.org/10.21105/joss.07213)

---
---
# CDCgov General Disclaimers
This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm).  GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise. 

## Related Documents
* [Open Practices](open_practices.md)
* [Rules of Behavior](rules_of_behavior.md)
* [Disclaimer](DISCLAIMER.md)
* [Contribution Notice](CONTRIBUTING.md)
* [Code of Conduct](code-of-conduct.md)
* [Licence](licence)
  
## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice
The repository utilizes code licensed under the terms of the Apache Software
License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](DISCLAIMER.md)
and [Code of Conduct](code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records, but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices
Please refer to [CDC's Template Repository](https://github.com/CDCgov/template) for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/main/CONTRIBUTING.md), [public domain notices and disclaimers](https://github.com/CDCgov/template/blob/main/DISCLAIMER.md), and [code of conduct](https://github.com/CDCgov/template/blob/main/code-of-conduct.md).


