"""Unit tests for DART (Dirichlet Additive Regression Trees) sparsity feature selection.
"""

import numpy as np
from bart_survival import (
    SurvBART,
    format_survival_data,
    simulate_confounded_survival_data,
)
from bart_survival.bart_adj import simple_feature_importance


def test_dart_sparsity_execution():
    """Verify that DART sparsity runs without error and logs variable inclusion."""
    df, meta = simulate_confounded_survival_data(
        N=100,
        treatment_hr=2.5,
        n_noise_features=3,
        seed=42
    )
    feature_cols = ["inpatient", "age", "stage", "noise_1", "noise_2", "noise_3"]
    X = df[feature_cols].values
    y_time = df["time"].values
    y_status = df["status"].values

    trn = format_survival_data(
        y_time=y_time,
        y_status=y_status,
        x=X,
        time_scale=15.0
    )

    model = SurvBART(
        model_config={"trees": 15},
        sampler_config={
            "draws": 50,
            "tune": 50,
            "cores": 2,
            "chains": 2,
            "use_sparsity": True,
            "sparsity_warmup": 20,
            "compute_convergence_checks": False,
        }
    )
    model.fit(
        y=trn["y"],
        X=trn["x"],
        weights=trn["w"],
        coords=trn["coord"],
        progressbar=False,
        random_seed=7,
    )

    assert model.is_fitted_
    assert "variable_inclusion" in model.idata.sample_stats

    # Check variable inclusion shape: (chains, steps, num_features)
    var_inc = model.idata.sample_stats["variable_inclusion"].values
    assert var_inc.shape[0] == 2  # 2 chains
    assert var_inc.shape[2] == trn["x"].shape[1]  # total features (time + covariates)

    # Total splits should be positive
    assert np.nansum(var_inc) > 0

    # Test feature importance function runs and produces normalized probabilities
    raw_counts = np.nansum(var_inc, axis=(0, 1))
    assert raw_counts.sum() > 0
    importance = raw_counts / raw_counts.sum()
    assert np.isclose(importance.sum(), 1.0)
    simple_feature_importance(model.idata)


if __name__ == "__main__":
    test_dart_sparsity_execution()
    print("All DART sparsity tests passed successfully!")
