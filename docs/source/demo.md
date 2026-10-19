
# Demonstration 

`BART-Survival` supports both an intuitive high-level API (`SurvBART`) for rapid, boilerplate-free modeling and a granular low-level API for custom pooled person-time workflows.

---

## 1. High-Level API (`SurvBART`)

In this workflow, tabular covariates and time-to-event outcomes are passed directly to `SurvBART`. Discretization, pooled person-time data expansion, out-of-sample grid formatting, and posterior survival curve derivation are handled automatically.

```python
import numpy as np
import pandas as pd
from bart_survival import SurvBART, simulate_confounded_survival_data

# Generate a synthetic observational clinical cohort with known treatment effect
df, meta = simulate_confounded_survival_data(
    N=300, 
    treatment_hr=1.8, 
    censoring_rate=0.2, 
    seed=42
)
features = ["inpatient", "age", "stage", "los"]
X = df[features]

# 1. Initialize SurvBART using the recommended standard scientific preset
# (50 trees, 1000 draws, 250 tune, 4 chains, thin=4, 512 empirical cutpoints)
model = SurvBART.preset_standard()

# 2. Fit directly from tabular covariates and survival outcomes
model.fit_survival(
    X=X,
    time=df["time"],
    status=df["status"],
    time_scale=5.0,  # 5-day intervals
    progressbar=True,
    random_seed=42
)

# 3. Predict cumulative survival curves S(t) with 95% Credible Intervals
surv_summary = model.predict_survival(X, return_quantiles=True)
print("Survival curve shape:", surv_summary["mean"].shape)
print("Mean S(t):", surv_summary["mean"].mean(axis=0))

# 4. Normalized variable inclusion proportions (Feature Importance)
print("\nPosterior Feature Importance:")
print(model.feature_importance())

# 5. Causal Treatment Effect with Conditional Support Audit
# Detects positivity violations (e.g., zero LOS among outpatients)
effect = model.estimate_effect(
    X=X, 
    primary_col="inpatient", 
    values=[0, 1]
)
print("\nEstimated Hazard Ratio Curve:", effect["hazard_ratio_mean"])
print("Flagged Support Violations:", effect["flagged_support_violations"])
```

---

## 2. Granular Step-by-Step API (TAD & PAD)

The low-level API provides full transparency into the pooled person-time Training Augmented Dataset (**TAD**) and Prediction Augmented Dataset (**PAD**). In this demonstration, the open *Rossi* recidivism benchmark dataset from `lifelines` is used [@rossi1980].

```python
from lifelines.datasets import load_rossi
from bart_survival import (
    SurvBART,
    format_survival_data,
    format_prediction_data,
    compute_survival_curves
)
import numpy as np

# Load Rossi recidivism dataset
rossi = load_rossi()
time = rossi["week"].values
status = rossi["arrest"].values
X = rossi[["fin", "age", "race", "wexp", "mar", "paro", "prio"]].values

# 1. Transform into pooled person-time TAD dataset
# Time (originally in weeks) is scaled by 4 into monthly intervals
trn = format_survival_data(
    y_time=time,
    y_status=status,
    x=X,
    time_scale=4.0
)

# 2. Generate out-of-sample prediction PAD grid
post_test = format_prediction_data(
    y_time=time,
    y_status=status,
    x=X,
    time_scale=4.0
)

# 3. Initialize SurvBART with custom parameters
model = SurvBART(
    trees=50,
    draws=1000,
    tune=500,
    chains=4,
    thin=5,
    update_fraction=1.0,
    split_method="empirical"
)

# 4. Fit model on TAD
model.fit(
    y=trn["y"],
    X=trn["x"],
    weights=trn["w"],
    coords=trn["coord"],
    progressbar=True,
    random_seed=5
)

# 5. Predict out-of-sample hazards and compute S(t)
post_preds = model.predict(post_test["post_x"], coords=post_test["coords"])
sv_prob = compute_survival_curves(post_preds, coords=post_test["coords"])

# S(t) has shape (n_draws, n_subjects, n_time_intervals)
print("S(t) shape:", sv_prob["sv"].shape)

# Population-level average survival curve S(t) across subjects and draws:
mean_survival = sv_prob["sv"].mean(axis=(0, 1))
print("Population Mean S(t):", mean_survival)
```


