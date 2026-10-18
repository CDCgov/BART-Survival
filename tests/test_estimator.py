"""Unit tests for SurvBART / BartSurvModel estimator, presets, convenience methods, and prediction.
"""

import numpy as np
import pandas as pd
from bart_survival import (
    SurvBART,
    BartSurvModel,
    format_survival_data,
    format_prediction_data,
    compute_survival_curves,
    simulate_confounded_survival_data,
)


def test_estimator_initialization_and_presets():
    """Verify default parameters, open constructor, and factory presets."""
    # 1. Default initialization
    model = SurvBART()
    assert model.trees == 50
    assert model.draws == 2500
    assert model.tune == 1000
    assert model.chains == 4
    assert model.thin == 10
    assert model.thin_idata is True
    assert model.use_sparsity is False
    assert model.update_fraction == 1.0
    assert model.split_method == "empirical"
    assert not model.is_fitted_

    # 2. Preset fast
    fast_m = SurvBART.preset_fast()
    assert fast_m.trees == 20
    assert fast_m.draws == 500
    assert fast_m.tune == 250
    assert fast_m.chains == 2
    assert fast_m.thin == 5

    # 3. Preset high precision
    hp_m = SurvBART.preset_high_precision()
    assert hp_m.trees == 100
    assert hp_m.draws == 5000
    assert hp_m.tune == 2000
    assert hp_m.chains == 4
    assert hp_m.thin == 10


def test_estimator_convenience_methods():
    """Verify fit_survival, predict_survival, feature_importance, and estimate_effect."""
    df, meta = simulate_confounded_survival_data(
        N=70, 
        treatment_hr=2.0, 
        censoring_rate=0.2, 
        seed=42
    )
    feature_cols = ["inpatient", "age", "stage"]
    X = df[feature_cols]

    # Initialize fast model for test suite speed
    model = SurvBART(
        trees=10,
        draws=30,
        tune=30,
        chains=2,
        cores=2,
        thin=1,
        thin_idata=False,
        update_fraction=1.0,
        use_sparsity=False
    )

    # 1. Test fit_survival (direct end-to-end fit from DataFrame and survival outcomes)
    model.fit_survival(
        X=X,
        time=df["time"],
        status=df["status"],
        time_scale=10.0,
        progressbar=False,
        random_seed=42
    )
    assert model.is_fitted_
    assert model.feature_names_ == feature_cols

    # 2. Test predict_survival (end-to-end survival curve generation)
    sv_curves = model.predict_survival(X)
    assert sv_curves.shape[1] == len(df)
    # Check monotonicity of S(t)
    for t_idx in range(sv_curves.shape[2] - 1):
        assert np.all(sv_curves[:, :, t_idx] >= sv_curves[:, :, t_idx + 1] - 1e-6)

    # Test return_quantiles
    sv_summary = model.predict_survival(X, return_quantiles=True)
    assert "mean" in sv_summary and "lower" in sv_summary and "upper" in sv_summary
    assert sv_summary["mean"].shape == (len(df), len(model.training_times_))

    # 3. Test feature_importance()
    feat_imp = model.feature_importance(sort=True)
    assert isinstance(feat_imp, pd.Series)
    assert len(feat_imp) == len(feature_cols) + 1  # includes discrete time 't'
    assert np.isclose(feat_imp.sum(), 1.0)

    # 4. Test estimate_effect() on binary exposure
    effect = model.estimate_effect(X, primary_col="inpatient", values=[0, 1])
    assert "hazard_ratio_weighted_mean" in effect
    assert "hazard_ratio_unweighted_mean" in effect
    assert "hazard_ratio_curve_mean" in effect
    assert "discrete_risk_ratio_mean" in effect
    assert "risk_difference_mean" in effect
    assert "risk_difference_weighted_mean" in effect
    assert "rmst_difference_mean" in effect
    assert "flagged_support_violations" in effect
    assert len(effect["hazard_ratio_curve_mean"]) == len(model.training_times_)
    assert len(effect["discrete_risk_ratio_mean"]) == len(model.training_times_)
    assert len(effect["risk_difference_mean"]) == len(model.training_times_)
    assert effect["variable_type"] == "binary"

    # 5. Test estimate_effect() on continuous exposure (shift effect)
    cont_effect = model.estimate_effect(X, primary_col="age", delta=5.0)
    assert cont_effect["variable_type"] == "continuous_shift"
    assert cont_effect["delta"] == 5.0
    assert "hazard_ratio_weighted_mean" in cont_effect
    assert "hazard_ratio_unweighted_mean" in cont_effect
    assert "risk_difference_mean" in cont_effect
    assert len(cont_effect["risk_difference_mean"]) == len(model.training_times_)


if __name__ == "__main__":
    test_estimator_initialization_and_presets()
    test_estimator_convenience_methods()
    print("All estimator tests passed successfully!")
