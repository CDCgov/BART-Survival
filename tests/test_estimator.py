"""Unit tests for SurvBART / BartSurvModel estimator, prediction, and serialization.
"""

import os
import tempfile
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


def test_estimator_initialization():
    """Verify default and custom configuration initialization."""
    # 1. Default initialization
    model = SurvBART()
    assert model.model_config["trees"] == 50
    assert model.model_config["alpha"] == 0.95
    assert model.sampler_config["draws"] == 100
    assert model.sampler_config["cores"] == 2
    assert not model.is_fitted_

    # 2. Custom configuration
    custom_model = SurvBART(
        model_config={"trees": 20, "alpha": 0.9},
        sampler_config={"draws": 50, "cores": 1, "chains": 1}
    )
    assert custom_model.model_config["trees"] == 20
    assert custom_model.sampler_config["draws"] == 50


def test_estimator_fit_predict_cycle():
    """Verify SurvBART fitting, out-of-sample prediction, and survival curve generation."""
    df, meta = simulate_confounded_survival_data(
        N=80, 
        treatment_hr=2.0, 
        censoring_rate=0.2, 
        seed=10
    )
    feature_cols = ["inpatient", "age", "stage"]
    X = df[feature_cols].values
    y_time = df["time"].values
    y_status = df["status"].values

    # Preprocess training pooled cross-section
    trn = format_survival_data(
        y_time=y_time, 
        y_status=y_status, 
        x=X, 
        time_scale=10.0
    )
    assert trn["y"].shape[0] == trn["x"].shape[0]
    assert trn["x"].shape[1] == len(feature_cols) + 1  # time + features

    # Test prediction data grid
    post_test = format_prediction_data(
        y_time=y_time, 
        y_status=y_status, 
        x=X, 
        time_scale=10.0
    )
    assert post_test["post_x"].shape[1] == len(feature_cols) + 1

    # Fit fast test model
    model = SurvBART(
        model_config={"trees": 15},
        sampler_config={
            "draws": 50,
            "tune": 50,
            "cores": 2,
            "chains": 2,
            "compute_convergence_checks": False,
        }
    )
    model.fit(
        y=trn["y"],
        X=trn["x"],
        weights=trn["w"],
        coords=trn["coord"],
        progressbar=False,
        random_seed=42,
    )
    assert model.is_fitted_
    assert model.idata is not None

    # Predict via both .predict() and .sample_posterior_predictive()
    preds1 = model.predict(X_pred=post_test["post_x"], coords=post_test["coords"])
    preds2 = model.sample_posterior_predictive(X_pred=post_test["post_x"], coords=post_test["coords"])

    assert preds1.shape == preds2.shape
    assert preds1.shape[0] == 100  # 50 draws * 2 chains
    assert preds1.shape[1] == post_test["post_x"].shape[0]
    assert np.all((preds1 >= 0.0) & (preds1 <= 1.0))

    # Convert to survival curves
    sv_res = compute_survival_curves(preds1, post_test["coords"])
    assert "prob" in sv_res and "sv" in sv_res
    # Survival curves must be monotonically non-increasing across time points
    sv_matrix = sv_res["sv"]  # shape: (n_draws, n_subjects, n_times)
    assert sv_matrix.shape[1] == len(df)
    for t_idx in range(sv_matrix.shape[2] - 1):
        assert np.all(sv_matrix[:, :, t_idx] >= sv_matrix[:, :, t_idx + 1] - 1e-6)


if __name__ == "__main__":
    test_estimator_initialization()
    test_estimator_fit_predict_cycle()
    print("All estimator tests passed successfully!")

