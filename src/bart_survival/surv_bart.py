"""
bart_survival.surv_bart
=======================
High-level API for Bayesian Additive Regression Trees (BART) survival analysis.

Provides:
- SurvBART (alias BartSurvModel): Estimator class with fit, predict, and serialization.
- format_survival_data (alias get_surv_pre_train): Preprocess time-to-event data into pooled cross-sections.
- format_prediction_data (alias get_posterior_test): Prepare test covariates across prediction time intervals.
- compute_survival_curves (alias get_sv_prob): Cumulative product survival probability transformation.
- compute_pdp (alias get_pdp): Partial dependence data generation.
- compute_causal_pdp (alias get_causal_pdp): Causal G-Computation with support violation audit.
- compute_hazard_ratio (alias pdp_rr_metric): Posterior Risk Ratio / Hazard Ratio extraction.
- compute_pdp_difference (alias pdp_diff_metric): Posterior marginal survival difference extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import hashlib
from pathlib import Path
import arviz as az
import pymc as pm
import pymc_bart as pmb
import xarray as xr
import json
import scipy.stats as sp
import warnings
import pytensor.tensor as pt
from pymc_bart.utils import _sample_posterior
import cloudpickle as cpkl
from numpy.random import RandomState
import multiprocessing as mp
from .bart_adj import FastSurvMHBART, predict_out_of_sample


class BartSurvModel:
    """Bayesian Additive Regression Trees (BART) Survival Model (SurvBART).

    Implements discrete-time probit hazard modeling using an ensemble of sum-of-trees.
    Supports warm-started Dirichlet sparsity selection (DART), empirical unique midpoint
    cutpoints (512 resolution), MCMC thinning, and support-audited Causal G-Computation.
    """
    _model_type = "BART_Survival"
    version = "0.2.1"

    def __init__(
        self,
        trees: int = 50,
        draws: int = 2500,
        tune: int = 1000,
        chains: int = 4,
        cores: Optional[int] = None,
        thin: int = 10,
        thin_idata: bool = False,
        use_sparsity: bool = False,
        split_method: str = "empirical",
        max_cutpoints: int = 512,
        update_fraction: float = 1.0,
        min_node_size: int = 5,
        k: float = 2.0,
        alpha: float = 0.95,
        beta: float = 2.0,
        sparsity_warmup: Optional[int] = None,
        unique_midpoints: Optional[bool] = None,
        split_rules: Optional[List[Any]] = None,
        split_prior: Optional[Any] = None,
        model_config: Optional[Dict[str, Any]] = None, 
        sampler_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize BartSurvModel / SurvBART.

        Args:
            trees (int, optional): Number of trees in the ensemble. Defaults to 50.
            draws (int, optional): Posterior MCMC draws per chain. Defaults to 2500.
            tune (int, optional): Warmup / tuning iterations per chain. Defaults to 1000.
            chains (int, optional): Number of parallel Markov chains. Defaults to 4.
            cores (Optional[int], optional): Number of CPU processes. Defaults to min(chains, cpu_count).
            thin (int, optional): Thinning interval for posterior draws. Defaults to 10.
            thin_idata (bool, optional): Legacy flag; thinning is applied safely after reshaping tree arrays. Defaults to False.
            use_sparsity (bool, optional): Whether to enable DART Dirichlet sparsity. Defaults to False.
            split_method (str, optional): Strategy for proposing split cutpoints ('empirical' or 'uniform').
                'empirical' evaluates unique midpoints up to max_cutpoints. Defaults to 'empirical'.
            max_cutpoints (int, optional): Maximum unique midpoints evaluated per feature. Defaults to 512.
            update_fraction (float, optional): Fraction of trees proposed per sweep. Defaults to 1.0 (all trees).
            min_node_size (int, optional): Minimum observation count per leaf node. Defaults to 5.
            k (float, optional): Leaf prior shrinkage parameter (tau = 3 / (k * sqrt(M))). Defaults to 2.0.
            alpha (float, optional): Tree depth prior parameter alpha * (1 + depth)^(-beta). Defaults to 0.95.
            beta (float, optional): Tree depth prior parameter alpha * (1 + depth)^(-beta). Defaults to 2.0.
            sparsity_warmup (Optional[int], optional): Tuning steps before Dirichlet updates begin.
            unique_midpoints (Optional[bool], optional): Legacy alias for split_method='empirical'. Defaults to None.
            split_rules (Optional[List[Any]], optional): PyMC split rules per feature. Defaults to None.
            split_prior (Optional[Any], optional): Prior on split variables. Defaults to None.
            model_config (Optional[Dict[str, Any]], optional): Legacy model configuration dictionary.
            sampler_config (Optional[Dict[str, Any]], optional): Legacy sampler configuration dictionary.
        """
        model_cfg = self.get_default_model_config()
        sampler_cfg = self.get_default_sampler_config()

        # Update from keyword arguments
        model_cfg.update({
            "trees": trees,
            "alpha": alpha,
            "beta": beta,
            "split_rules": split_rules,
            "split_prior": split_prior,
            "split_method": split_method,
            "max_cutpoints": max_cutpoints,
        })
        sampler_cfg.update({
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "cores": cores,
            "thin": thin,
            "thin_idata": thin_idata,
            "use_sparsity": use_sparsity,
            "update_fraction": update_fraction,
            "min_node_size": min_node_size,
            "k": k,
            "sparsity_warmup": sparsity_warmup,
        })

        if unique_midpoints is not None:
            sampler_cfg["unique_midpoints"] = bool(unique_midpoints)
        else:
            sampler_cfg["unique_midpoints"] = (str(split_method).lower() in ("empirical", "empirical_512", "midpoints", "true", "1"))

        # Merge legacy dicts if provided
        if model_config is not None:
            model_cfg.update(model_config)
        if sampler_config is not None:
            sampler_cfg.update(sampler_config)

        self.model_config = model_cfg
        self.sampler_config = sampler_cfg

        # Direct property bindings
        self.trees = self.M = int(self.model_config["trees"])
        self.draws = int(self.sampler_config["draws"])
        self.tune = int(self.sampler_config["tune"])
        self.chains = int(self.sampler_config["chains"])
        self.cores = self.sampler_config.get("cores", None)
        self.thin = int(self.sampler_config["thin"])
        self.thin_idata = bool(self.sampler_config["thin_idata"])
        self.use_sparsity = bool(self.sampler_config["use_sparsity"])
        self.update_fraction = float(self.sampler_config["update_fraction"])
        self.min_node_size = int(self.sampler_config["min_node_size"])
        self.k = float(self.sampler_config["k"])
        self.split_method = str(self.model_config.get("split_method", "empirical"))
        self.max_cutpoints = int(self.model_config.get("max_cutpoints", 512))

        self.model = None
        self.idata: Optional[az.InferenceData] = None
        self.is_fitted_ = False
        self.feature_names_: Optional[List[str]] = None
        self.time_scale_: Optional[float] = None
        self.training_times_: Optional[np.ndarray] = None
        self.event_counts_: Optional[np.ndarray] = None
        self.event_weights_: Optional[np.ndarray] = None

    @classmethod
    def preset_standard(cls, **overrides: Any) -> "BartSurvModel":
        """Standard scientific preset: 50 trees, 1,000 draws, 250 tune, 4 chains, thin 4 (1,000 final draws), update_fraction 1.0, empirical cutpoints."""
        cfg: Dict[str, Any] = dict(
            trees=50,
            draws=1000,
            tune=250,
            chains=4,
            thin=4,
            thin_idata=False,
            update_fraction=1.0,
            use_sparsity=False,
            split_method="empirical",
        )
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def preset_production(cls, **overrides: Any) -> "BartSurvModel":
        """Legacy alias for preset_standard."""
        return cls.preset_standard(**overrides)

    @classmethod
    def preset_default(cls, **overrides: Any) -> "BartSurvModel":
        """Default recommended preset (alias for preset_standard)."""
        return cls.preset_standard(**overrides)

    @classmethod
    def preset_fast(cls, **overrides: Any) -> "BartSurvModel":
        """Fast exploration preset: 25 trees, 500 draws, 250 tune, 4 chains, thin 5, update_fraction 1.0."""
        cfg: Dict[str, Any] = dict(
            trees=25,
            draws=500,
            tune=250,
            chains=4,
            thin=5,
            thin_idata=False,
            update_fraction=1.0,
            use_sparsity=False,
            split_method="empirical",
        )
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def preset_high_precision(cls, **overrides: Any) -> "BartSurvModel":
        """High-precision publication preset: 100 trees, 5000 draws, 2000 tune, 4 chains, thin 10, update_fraction 1.0."""
        cfg: Dict[str, Any] = dict(
            trees=100,
            draws=5000,
            tune=2000,
            chains=4,
            thin=10,
            thin_idata=False,
            update_fraction=1.0,
            use_sparsity=False,
            split_method="empirical",
        )
        cfg.update(overrides)
        return cls(**cfg)

    def build_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray,                        
        weights: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None,
        predictor_names: Optional[List[str]] = None,
        **kwargs: Any
    ) -> pm.Model: 
        """Builds the PyMC probabilistic model for BART discrete-time survival.

        Args:
            X (np.ndarray): Covariate matrix in long-time form [time, x1, x2, ...].
            y (np.ndarray): Event status in long-time form (0=censored/alive, 1=event).
            weights (Optional[np.ndarray], optional): Sample observation weights. Defaults to None.
            coords (Optional[np.ndarray], optional): Subject coordinate index alignment. Defaults to None.
            predictor_names (Optional[List[str]], optional): Variable names. Defaults to None.
            **kwargs: Additional model parameters.

        Returns:
            pm.Model: The compiled PyMC model.
        """
        self._generate_and_preprocess_model_data(X, y, weights, predictor_names)
        split_rules = self.model_config.get("split_rules", None)
        if split_rules is None:
            SPLIT_RULES = [pmb.ContinuousSplitRule] * self.X.shape[1]
        else:
            SPLIT_RULES = [eval(rule) if isinstance(rule, str) else rule for rule in split_rules]
        SPLIT_PRIOR = self.model_config.get("split_prior")
        self.M = int(self.model_config.get("trees", 50))

        # Probit hazard link offset
        self.offset = sp.norm.ppf(np.mean(self.y))
        mcoords = {"xvars": self.predictor_names}
        k = float(self.sampler_config.get("k", 2.0))
        update_fraction = float(self.sampler_config.get("update_fraction", 1.0))
        min_node_size = int(self.sampler_config.get("min_node_size", 5))
        
        unique_midpoints = self.sampler_config.get("unique_midpoints", None)
        if unique_midpoints is None:
            split_method = str(self.model_config.get("split_method", "empirical")).lower()
            unique_midpoints = (split_method in ("empirical", "empirical_512", "midpoints", "true", "1"))
        else:
            unique_midpoints = bool(unique_midpoints)

        self.use_sparsity = bool(self.sampler_config.get("use_sparsity", False))
        tune = int(self.sampler_config.get("tune", 1000))
        default_warmup = min(300, max(50, tune // 2))
        sparsity_warmup = self.sampler_config.get("sparsity_warmup")
        if sparsity_warmup is None:
            sparsity_warmup = default_warmup
        else:
            sparsity_warmup = int(sparsity_warmup)
        
        # initialize model
        alpha = float(self.model_config.get("alpha", 0.95))
        beta = float(self.model_config.get("beta", 2.0))
        with pm.Model(coords=mcoords) as self.model:    
            trees = pmb.BART("trees", X=np.zeros((1, self.X.shape[1])), Y=np.zeros(1), m=self.M, alpha=alpha, beta=beta)
            if hasattr(trees, "owner") and hasattr(trees.owner, "op") and hasattr(trees.owner.op, "all_trees"):
                self._manager = getattr(trees.owner.op.all_trees, "_manager", None)
            else:
                self._manager = None
            prob = 0.5 * (1.0 + pt.erf(trees / pt.sqrt(2.0)))
            obs = pm.Bernoulli("obs", p=prob, observed=self.y)
        
            self.custom_step = FastSurvMHBART(
                vars=[trees], 
                use_sparsity=self.use_sparsity, 
                sparsity_warmup=sparsity_warmup,
                target_x=self.X,
                target_y=self.y,
                update_fraction=update_fraction,
                k=k, 
                min_node_size=min_node_size,
                unique_midpoints=unique_midpoints,
                return_estimates=False
            )
            self.offset = self.custom_step.offset
        return self.model

    def sample_model(
        self, 
        **kwargs: Any
    ) -> az.InferenceData:
        """Initiates training/sampling of the model.

        Returns:
            az.InferenceData: Posterior data collected from training the model.
        """
        if self.model is None:
            raise RuntimeError(
                "The model hasn't been built yet, call .build_model() first or call .fit() instead."
            )

        draws = int(self.sampler_config.get("draws", 2500))
        tune = int(self.sampler_config.get("tune", 1000))
        chains = int(self.sampler_config.get("chains", 4))
        cores = self.sampler_config.get("cores", None)
        if cores is None:
            import os
            cpu_cnt = os.cpu_count() or 4
            cores = min(chains, max(1, cpu_cnt))

        with self.model:
            sampler_args = {**self.sampler_config, **kwargs}
            pbar = sampler_args.get("progressbar", kwargs.get("progressbar", True))
            idata = pm.sample(
                draws=draws, 
                tune=tune, 
                chains=chains, 
                cores=cores,
                step=self.custom_step,
                compute_convergence_checks=False,
                progressbar=pbar,
                discard_tuned_samples=True
            )
        # Note: Do NOT slice idata along `draw` directly because PyMC flattens 2D step stats
        # (F_trees, T_trees, L_trees, R_trees, V_trees) of shape (m_trees, max_nodes) along `draw`,
        # making the sample_stats `draw` dimension length `draws * m_trees`.
        # Thinning is applied cleanly after reshaping to (n_samples_raw, m_trees, max_nodes) in predict_out_of_sample.
        idata = self.set_idata_attrs(idata)
        return idata

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None,
        progressbar: bool = True,
        predictor_names: Optional[List[str]] = None,
        random_seed: Optional[Union[int, RandomState]] = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Fit the BART survival model using MCMC sampling.

        Args:
            y (np.ndarray): Binary event indicator in long-time pooled format (1=event, 0=censored/alive).
            X (np.ndarray): Covariate matrix in long-time format where column 0 is discrete time,
                followed by baseline covariates [time, x1, x2, ...].
            weights (Optional[np.ndarray], optional): Sample observation weights (e.g. for case-cohort designs). Defaults to None.
            coords (Optional[np.ndarray], optional): Subject coordinate index alignment for observations. Defaults to None.
            progressbar (bool, optional): Whether to display MCMC progress bar. Defaults to True.
            predictor_names (Optional[List[str]], optional): Names of features in X matrix. Defaults to None.
            random_seed (Optional[Union[int, RandomState]], optional): Random seed for reproducibility. Defaults to None.
            **kwargs: Additional sampling configuration overrides (e.g. draws, tune, cores, chains).

        Returns:
            az.InferenceData: Posterior inference data object containing sampled trees and chain statistics.
        """
        # build model
        self.build_model(X, y, weights, coords, predictor_names)
        if predictor_names is not None and len(predictor_names) > 1 and self.feature_names_ is None:
            self.feature_names_ = list(predictor_names[1:])
        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)

        # sample the model
        self.idata = self.sample_model(**sampler_config)
        self.is_fitted_ = True

        # Calculate empirical event counts and interval weights if not already set
        if self.event_weights_ is None and self.X is not None and self.y is not None:
            unique_t = np.sort(np.unique(self.X[:, 0]))
            self.training_times_ = unique_t
            y_flat = np.asarray(self.y, dtype=np.int32).flatten()
            self.event_counts_ = np.array([
                np.sum((self.X[:, 0] == t) & (y_flat == 1)) for t in unique_t
            ], dtype=np.float64)
            tot_ev = float(np.sum(self.event_counts_))
            self.event_weights_ = self.event_counts_ / tot_ev if tot_ev > 0 else np.ones(len(unique_t)) / len(unique_t)

        self.all_trees = list(self.idata.sample_stats)
        # Release background SyncManager process now that MCMC sampling is complete
        self.close()
        return self.idata

    def close(self) -> None:
        """Explicitly releases background SyncManager server processes and frees IPC memory."""
        if getattr(self, "_manager", None) is not None:
            try:
                self._manager.shutdown()
                self._manager = None
            except Exception:
                pass
        if hasattr(self, "custom_step") and hasattr(self.custom_step, "all_trees"):
            mgr = getattr(self.custom_step.all_trees, "_manager", None)
            if mgr is not None:
                try:
                    mgr.shutdown()
                except Exception:
                    pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "BartSurvModel":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def set_idata_attrs(
        self, 
        idata:Optional[az.InferenceData]=None
    ) -> az.InferenceData:
        """Sets the additional information in the idata object.

        Args:
            idata (Optional[az.InferenceData], optional): Idata. Defaults to None.

        Returns:
            az.InferenceData: Idata
        """

        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")
        idata.attrs["id"] = self.id
        idata.attrs["model_type"] = self._model_type
        idata.attrs["version"] = self.version
        idata.attrs["sampler_config"] = json.dumps(self.sampler_config)
        idata.attrs["model_config"] = json.dumps(self._serializable_model_config)
        if hasattr(self, "offset") and self.offset is not None:
            if isinstance(self.offset, tuple):
                idata.attrs["offset"] = json.dumps([self.offset[0].tolist(), self.offset[1].tolist()])
            elif np.ndim(self.offset) == 0:
                idata.attrs["offset"] = float(self.offset)
        return idata

    def sample_posterior_predictive(
        self, 
        X_pred: np.ndarray, 
        coords: Optional[np.ndarray] = None, 
        extend_idata: bool = False, 
        **kwargs: Any
    ) -> np.ndarray:
        """Derives posterior predictive hazard probabilities on new covariates.

        Args:
            X_pred (np.ndarray): Covariate matrix in long-time format [time, x1, x2, ...].
            coords (Optional[np.ndarray], optional): Coordinates associated with subject indices. Defaults to None.
            extend_idata (bool, optional): Legacy compatibility flag. Defaults to False.
            **kwargs: Additional keyword arguments.
        
        Returns:
            np.ndarray: Array of shape (n_draws, n_obs) containing predicted hazard probabilities.
        """ 
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted in this instance. Either refit or predict with bart_predict")
        
        offset = getattr(self, "offset", None)
        if offset is None and hasattr(self, "custom_step") and self.custom_step is not None:
            offset = getattr(self.custom_step, "offset", None)
        if offset is None and hasattr(self, "idata") and self.idata is not None and "offset" in self.idata.attrs:
            raw_off = self.idata.attrs["offset"]
            if isinstance(raw_off, str):
                loaded = json.loads(raw_off)
                if isinstance(loaded, list) and len(loaded) == 2:
                    offset = (np.array(loaded[0], dtype=np.float64), np.array(loaded[1], dtype=np.float32))
                else:
                    offset = float(loaded)
            elif isinstance(raw_off, (int, float)):
                offset = float(raw_off)

        thin = int(self.sampler_config.get("thin", 1))
        preds = predict_out_of_sample(self.idata, X_pred, offset, m_trees=self.M, thin=thin)
        return preds

    def predict(
        self, 
        X_pred: np.ndarray, 
        coords: Optional[np.ndarray] = None, 
        **kwargs: Any
    ) -> np.ndarray:
        """Alias for sample_posterior_predictive following scikit-learn convention.

        Args:
            X_pred (np.ndarray): Covariate matrix in long-time format [time, x1, x2, ...].
            coords (Optional[np.ndarray], optional): Coordinates associated with subject indices. Defaults to None.
            **kwargs: Additional prediction keyword options.

        Returns:
            np.ndarray: Array of shape (n_draws, n_obs) containing predicted hazard probabilities.
        """
        return self.sample_posterior_predictive(X_pred=X_pred, coords=coords, **kwargs)

    def fit_survival(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        time: Union[np.ndarray, pd.Series, List[float]],
        status: Union[np.ndarray, pd.Series, List[int]],
        time_scale: Optional[float] = None,
        weights: Optional[np.ndarray] = None,
        progressbar: bool = True,
        random_seed: Optional[Union[int, RandomState]] = None,
        **kwargs: Any
    ) -> az.InferenceData:
        """Fits SurvBART directly from baseline tabular covariates and survival outcomes.

        Eliminates manual boilerplate by automatically discretizing follow-up times and
        expanding tabular data into long-format person-time pooled cross-sections.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Baseline covariates.
            time (Union[np.ndarray, pd.Series, List[float]]): Observed event or censoring times.
            status (Union[np.ndarray, pd.Series, List[int]]): Binary event indicators (1=event, 0=censored).
            time_scale (Optional[float], optional): Interval discretization scale factor (e.g. 5 or 30). Defaults to None.
            weights (Optional[np.ndarray], optional): Sample observation weights. Defaults to None.
            progressbar (bool, optional): Whether to display MCMC progress bar. Defaults to True.
            random_seed (Optional[Union[int, RandomState]], optional): Random seed. Defaults to None.
            **kwargs: Additional sampling keyword overrides.

        Returns:
            az.InferenceData: Fitted posterior inference data object.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.values
        else:
            X_mat = np.asarray(X)
            self.feature_names_ = [f"x{i}" for i in range(X_mat.shape[1])]

        y_time = np.asarray(time, dtype=np.float64).flatten()
        y_status = np.asarray(status, dtype=np.int32).flatten()

        self.time_scale_ = time_scale
        scaled_time = np.ceil(y_time / time_scale) if time_scale is not None else y_time
        self.training_times_ = np.sort(np.unique(scaled_time))

        self.event_counts_ = np.array([
            np.sum((scaled_time == t) & (y_status == 1)) for t in self.training_times_
        ], dtype=np.float64)
        tot_ev = float(np.sum(self.event_counts_))
        self.event_weights_ = self.event_counts_ / tot_ev if tot_ev > 0 else np.ones(len(self.training_times_)) / len(self.training_times_)

        trn = format_survival_data(
            y_time=y_time,
            y_status=y_status,
            x=X_mat,
            weight=weights,
            time_scale=time_scale
        )

        names = ["t"] + self.feature_names_
        return self.fit(
            y=trn["y"],
            X=trn["x"],
            weights=trn["w"],
            coords=trn["coord"],
            progressbar=progressbar,
            predictor_names=names,
            random_seed=random_seed,
            **kwargs
        )

    def predict_survival(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        times: Optional[Union[np.ndarray, List[float]]] = None,
        return_quantiles: bool = False,
        qntile: List[float] = [0.025, 0.975],
        **kwargs: Any
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Calculates out-of-sample cumulative survival curves S(t) for new subjects.

        Args:
            X_test (Union[np.ndarray, pd.DataFrame]): Baseline covariates for test cohort.
            times (Optional[Union[np.ndarray, List[float]]], optional): Discrete time evaluation grid.
                If None, uses unique event times observed during training. Defaults to None.
            return_quantiles (bool, optional): If True, returns mean and credible interval quantiles.
                If False, returns full posterior draws of shape (n_draws, n_subjects, n_times). Defaults to False.
            qntile (List[float], optional): Credible interval quantiles. Defaults to [0.025, 0.975].
            **kwargs: Additional prediction keyword arguments.

        Returns:
            Union[np.ndarray, Dict[str, np.ndarray]]: Full survival curve array or summary dict.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet. Call fit() or fit_survival() first.")

        if isinstance(X_test, pd.DataFrame):
            X_mat = X_test.values
        else:
            X_mat = np.asarray(X_test)

        if times is None:
            if hasattr(self, "training_times_") and self.training_times_ is not None:
                eval_times = self.training_times_
            else:
                eval_times = self.uniq_times
            t_scale = None
        else:
            eval_times = np.asarray(times)
            t_scale = getattr(self, "time_scale_", None)

        dummy_status = np.zeros_like(eval_times, dtype=int)
        post_test = format_prediction_data(
            y_time=eval_times,
            y_status=dummy_status,
            x=X_mat,
            time_scale=t_scale
        )

        preds = self.predict(post_test["post_x"], coords=post_test["coords"], **kwargs)
        sv_dict = compute_survival_curves(preds, post_test["coords"])
        sv_curves = sv_dict["sv"]  # shape: (n_draws, n_subjects, n_times)

        if return_quantiles:
            sv_mean = sv_curves.mean(axis=0)
            sv_q = np.quantile(sv_curves, qntile, axis=0)
            return {
                "mean": sv_mean,
                "lower": sv_q[0],
                "upper": sv_q[1],
                "quantiles": sv_q,
                "times": eval_times,
                "prob": sv_dict["prob"],
                "sv": sv_curves,
            }
        return sv_curves

    def feature_importance(self, sort: bool = True) -> pd.Series:
        """Extracts normalized variable split inclusion proportions from the posterior tree ensemble.

        Args:
            sort (bool, optional): Whether to sort features in descending order of importance. Defaults to True.

        Returns:
            pd.Series: Feature importance series indexed by predictor names.
        """
        if not self.is_fitted_ or self.idata is None:
            raise RuntimeError("Model is not fitted yet.")

        var_inc = self.idata.sample_stats["variable_inclusion"].values
        raw_counts = np.nansum(var_inc, axis=(0, 1))
        total = raw_counts.sum()
        if total > 0:
            importances = raw_counts / total
        else:
            importances = np.zeros_like(raw_counts)

        names = self.predictor_names if hasattr(self, "predictor_names") and self.predictor_names is not None else [f"f{i}" for i in range(len(importances))]
        series = pd.Series(importances, index=names, name="importance")
        if sort:
            series = series.sort_values(ascending=False)
        return series

    def estimate_effect(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        primary_col: Union[str, int] = "inpatient",
        values: Optional[List[Union[int, float]]] = None,
        delta: Optional[float] = None,
        shift: Optional[bool] = None,
        times: Optional[Union[np.ndarray, List[float]]] = None,
        audit_support: bool = True,
        donor_seed: Optional[int] = 42,
        event_weights: Optional[np.ndarray] = None,
        clip_to_range: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Estimates counterfactual treatment effects using Causal G-Computation and Modified Treatment Policies.

        Supports both:
        1. Binary / Categorical exposures: Audits conditional support for positivity violations (e.g. deterministic
           zeros like LOS=0 for outpatients) and re-routes dependent features to conditional donor resampling.
        2. Continuous exposures: Evaluates a counterfactual shift intervention (Modified Treatment Policy) comparing
           natural course (X) vs shifted exposure (X + delta). Alternatively evaluates fixed 2-point contrasts if
           explicit values=[v0, v1] are supplied.

        Metrics computed:
        - Event-weighted summary Hazard Ratio (default, Cox-aligned via complementary log-log link).
        - Unweighted summary Hazard Ratio across follow-up intervals.
        - Time-varying Hazard Ratio curve HR(t) via complementary log-log link.
        - Time-varying discrete risk ratio curve RR_discrete(t) = h1(t) / h0(t) (not averaged).
        - Time-varying Risk Difference curve RD(t) = Risk_1(t) - Risk_0(t) = S_0(t) - S_1(t) (1 over 0).
        - Survival Difference curve S_1(t) - S_0(t) = -RD(t).
        - Event-weighted and unweighted summary Risk Differences.
        - Final horizon Risk Difference RD(t_end).
        - Restricted Mean Survival Time (RMST) difference.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Baseline cohort covariates.
            primary_col (Union[str, int], optional): Primary exposure variable name or column index. Defaults to "inpatient".
            values (Optional[List[Union[int, float]]], optional): Counterfactual intervention values.
                If None, auto-detects binary exposures ([0, 1]) or triggers continuous shift intervention.
            delta (Optional[float], optional): Shift increment for continuous exposures (X + delta). Defaults to 1.0.
            shift (Optional[bool], optional): Force continuous shift intervention mode. Defaults to None.
            times (Optional[Union[np.ndarray, List[float]]], optional): Evaluation time points. Defaults to None.
            audit_support (bool, optional): Whether to audit and flag support violations. Defaults to True.
            donor_seed (Optional[int], optional): Random seed for reproducible donor resampling. Defaults to 42.
            event_weights (Optional[np.ndarray], optional): Normalized interval weights for summary HR/RD.
                Defaults to model's empirical event_weights_ if available.
            clip_to_range (bool, optional): Whether to clip continuous shifted values to observed maximum. Defaults to False.
            **kwargs: Additional prediction keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary containing weighted and unweighted HRs, RDs, RMST, curves, credible intervals, and draws.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet.")

        if isinstance(X, np.ndarray):
            feat_names = getattr(self, "feature_names_", None)
            if feat_names is not None and len(feat_names) == X.shape[1]:
                X_df = pd.DataFrame(X, columns=feat_names)
            else:
                X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        if isinstance(primary_col, str):
            col_name = primary_col
        else:
            col_name = X_df.columns[primary_col]

        raw_col_vals = X_df[col_name].values
        uniq_vals = np.unique(raw_col_vals[~np.isnan(raw_col_vals)])
        is_binary = (len(uniq_vals) <= 2)

        if is_binary and not shift:
            if values is None:
                val_list = [float(uniq_vals[0]), float(uniq_vals[1])] if len(uniq_vals) == 2 else [0.0, 1.0]
            else:
                val_list = [float(v) for v in values]
            
            out_sk, c_idx, flagged = compute_causal_pdp(
                x=X_df,
                primary_col=col_name,
                values=val_list,
                audit_support=audit_support,
                donor_seed=donor_seed
            )
            var_type = "binary"
            applied_delta = None
        else:
            if values is not None and len(values) == 2 and not shift:
                val_list = [float(v) for v in values]
                out_sk, c_idx, flagged = compute_causal_pdp(
                    x=X_df,
                    primary_col=col_name,
                    values=val_list,
                    audit_support=audit_support,
                    donor_seed=donor_seed
                )
                var_type = "contrast"
                applied_delta = None
            else:
                applied_delta = 1.0 if delta is None else float(delta)
                out_sk, c_idx = compute_shift_pdp(
                    x=X_df,
                    primary_col=col_name,
                    delta=applied_delta,
                    clip_to_range=clip_to_range
                )
                flagged = []
                var_type = "continuous_shift"

        if times is None:
            if hasattr(self, "training_times_") and self.training_times_ is not None:
                eval_times = self.training_times_
            else:
                eval_times = self.uniq_times
            t_scale = None
        else:
            eval_times = np.asarray(times)
            t_scale = getattr(self, "time_scale_", None)

        dummy_status = np.zeros_like(eval_times, dtype=int)
        post_test = format_prediction_data(
            y_time=eval_times,
            y_status=dummy_status,
            x=out_sk,
            time_scale=t_scale
        )

        preds = self.predict(post_test["post_x"], coords=post_test["coords"], **kwargs)
        sv_dict = compute_survival_curves(preds, post_test["coords"])

        # Determine interval event weights
        if event_weights is not None:
            eff_weights = event_weights
        elif hasattr(self, "event_weights_") and self.event_weights_ is not None:
            eff_weights = self.event_weights_
        else:
            eff_weights = None

        n_cohort = len(X_df)
        effects_res = compute_hazard_and_risk_difference(
            pdp_val=sv_dict,
            idx=n_cohort,
            event_weights=eff_weights
        )

        effects_res.update({
            "times": eval_times,
            "flagged_support_violations": flagged,
            "variable_type": var_type,
            "delta": applied_delta,
            "sv_prob": sv_dict,
        })
        return effects_res
    
    def bart_predict(
        self, 
        X_pred:np.ndarray, 
        coords:np.ndarray, 
        size:int = None, 
        rng:RandomState=None, 
        **kwargs
    )->xr.DataArray:
        """Derives posterior predictions on updated dataset. Alternative method for re-loaded models.

        Args:
            X_pred (np.ndarray): Covariate matrix in long-time format.
            coords (np.ndarray): Coordinates associated with long-time format.
            size (int, optional): Sets sample of posterior draws. Defaults to None.
            rng (RandomState, optional): Random number generator for reproducable results. Defaults to None.

        Returns:
            xr.DataArray: DataArray containing the predicted outputs from the model and covariate matrix.
        """
        if not self.all_trees:
            raise RuntimeError("No tree structure loaded. Load a tree or train the model")
        trees = self.all_trees
        if size is None:
            size = len(self.all_trees)
        if rng is None:
            rng = np.random.default_rng(1)
        pt_x = pt.constant(X_pred)
        post_pred_mu_ = _sample_posterior(trees, pt_x, rng, size = size, shape=1)
        post_pred = pm.math.invprobit(post_pred_mu_ + self.offset).eval()
        post_pred = post_pred.reshape(post_pred.shape[0], post_pred.shape[1])
        post_pred = xr.DataArray(post_pred, name = "mu", 
                                    coords = {"sample":np.arange(0,post_pred.shape[0]),
                                            "p_obs": coords
                                    })
        return post_pred

    def _data_setter(
        self, 
        X:np.ndarray, 
        y:np.ndarray = None
    )->None:
        """Updates covariate matrix in the model object.

        Args:
            X (np.ndarray): Covariate matrix in long-time form.
            y (np.ndarray, optional): Event status in long-time form. Defaults to None.
        """
        with self.model:
            pm.set_data({"x_data": X})
            if y is not None:
                pm.set_data({"y_data": y.values if isinstance(y, pd.Series) else y})


    @classmethod
    def load(
        cls, 
        fname: str, 
        treename: str
    ):
        """Loads a saved model.

        Args:
            fname (str): Path to saved model object.
            treename (str): Path to saved tree object.

        Returns:
            _type_: Returns object of SurvBartModel class.
        """
        if ".nc" in fname:
            filepath = Path(str(fname))
            idata = az.from_netcdf(filepath)
        else:
            filepath = Path(str(fname))
            with open(filepath, mode="rb") as filein:
                idata = cpkl.load(filein)
        # needs to be converted, because json.loads was changing tuple to list
        model_config = cls._model_config_formatting(json.loads(idata.attrs["model_config"]))
        model = cls(
            model_config=model_config,
            sampler_config=json.loads(idata.attrs["sampler_config"]),
        )
        model.idata = idata
        dataset = idata.fit_data["x"].values
        X = dataset[:,2:]
        y = dataset[:,0]
        weights = dataset[:,1]
        coords = idata.posterior.coords["p_obs"].values
        predictor_names = idata.constant_data.coords["xvars"]
        model.build_model(X, y, weights, coords, predictor_names)

        # All previously used data is in idata.
        # cls.predictor_names = idata.predictor_names["x"].values
        if model.id != idata.attrs["id"]:
            raise ValueError(
                f"The file '{fname}' does not contain an inference data of the same model or configuration as '{cls._model_type}'"
            )
        
        with open(treename, mode="rb") as filein:
            cls.all_trees = cpkl.load(filein)
        return model

    def save(
        self, 
        idata_name:str, 
        all_tree_name:str
    ) -> None:
        """Saves a trained model and tree object.

        Args:
            idata_name (str): Path to saving model object.
            all_tree_name (str): Path to saving tree object.
        """
        if self.idata is not None and "posterior" in self.idata:
            file = idata_name
            if ".nc" in file:
                self.idata.to_netcdf(str(file))
            else:
                with open(file, mode='wb') as out:
                        cpkl.dump(self.idata, out)
        else:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
        
        if self.all_trees is not None:
            file = all_tree_name
            with open(file, mode='wb') as out:
                cpkl.dump(self.all_trees, out)
        else:
            raise RuntimeError("No tree structure has been detected. Make sure model has been fit")

    @classmethod
    def _model_config_formatting(
        cls, 
        model_config: Dict
    ) -> Dict:
        """Serialize config to add to idata.
        Because of json serialization, model_config values that were originally tuples or numpy are being encoded as lists.
        This function converts them back to tuples and numpy arrays to ensure correct id encoding.

        Args:
            model_config (Dict): Model configurations.

        Returns:
            Dict: Model configurations serialized.
        """
        for key in model_config:
            if isinstance(model_config[key], dict):
                for sub_key in model_config[key]:
                    if isinstance(model_config[key][sub_key], list):
                        # Check if "dims" key to convert it to tuple
                        if sub_key == "dims":
                            model_config[key][sub_key] = tuple(model_config[key][sub_key])
                        # Convert all other lists to numpy arrays
                        else:
                            model_config[key][sub_key] = np.array(model_config[key][sub_key])
        return model_config

    @staticmethod
    def get_default_model_config() -> Dict[str, Any]:
        """Default model prior and tree ensemble configuration."""
        return {
            "trees": 50,
            "alpha": 0.95,
            "beta": 2.0,
            "split_method": "empirical",
            "max_cutpoints": 512,
            "split_rules": None,
            "split_prior": None,
        }

    @staticmethod
    def get_default_sampler_config() -> Dict[str, Any]:
        """Default sampler and MCMC inference configuration."""
        return {
            "draws": 2500,
            "tune": 1000,
            "chains": 4,
            "cores": None,
            "thin": 10,
            "thin_idata": False,
            "update_fraction": 1.0,
            "min_node_size": 5,
            "k": 2.0,
            "use_sparsity": False,
            "unique_midpoints": True,
            "compute_convergence_checks": False,
        }

    @property
    def output_var(self)->str:
        return "y_pred"

    @property
    def _serializable_model_config(
        self
    ) -> Dict[str, Union[int, float, Dict]]:
        """Serialize Model Config

        Returns:
            Dict[str, Union[int, float, Dict]]: Serialized Model config.
        """
        model_config = self.model_config.copy()
        if "split_rules" in model_config and model_config["split_rules"] is not None:
            model_config["split_rules"] = [str(sp_rule) for sp_rule in model_config["split_rules"]]
        return model_config

    def _generate_and_preprocess_model_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        weights:np.ndarray,
        predictor_names: np.ndarray
    ) -> None:
        """Preprocess loaded data.

        Args:
            X (np.ndarray): Covariate matrix in long-time format.
            y (np.ndarray): Event status in long-time format.
            weights (np.ndarray): Weights associated with observations.
            predictor_names (np.ndarray): Names associated with covariates in X.
        """
        self.model_coords = None 
        assert type(X).__module__ == "numpy"
        assert type(y).__module__ == "numpy"
        self.X = X
        self.y = y
        self.uniq_times = np.unique(X[:,0])
        # set weights to 1 if not using a case_cohort design
        if weights is None:
            weights = np.ones(X.shape[0])
        self.weights = weights.reshape(weights.shape[0],1)
        # set predictor names
        # x should have time so make sure that first column is time
        if predictor_names is None:
            predictor_names = ["x" + str(i) for i in np.arange(X.shape[1])]
            predictor_names[0] = "t"
        self.predictor_names = predictor_names

    @property
    def id(self) -> str:
        """Generate a unique hash value for the model.

        The hash value is created using the last 16 characters of the SHA256 hash encoding, based on the model configuration,
        version, and model type.

        Returns:
            str: A string of length 16 characters containing a unique hash of the model.
        """
        hasher = hashlib.sha256()
        hasher.update(str(self.model_config.values()).encode())
        hasher.update(self.version.encode())
        hasher.update(self._model_type.encode())
        return hasher.hexdigest()[:16]

def get_surv_pre_train( 
    y_time: np.ndarray,
    y_status: np.ndarray,
    x: np.ndarray, 
    weight: Optional[np.ndarray] = None,
    time_scale: Optional[float] = None 
) -> Dict[str, np.ndarray]:
    """Generates pooled cross-sectional long-format dataset for discrete-time survival training.

    Transforms right-censored time-to-event outcomes into person-time binary event observations.
    Each individual contributes records for discrete intervals $t = 1, 2, \\dots, T_i$.

    Args:
        y_time (np.ndarray): Array of observed follow-up times.
        y_status (np.ndarray): Binary event indicator (1 if event occurred, 0 if right-censored).
        x (np.ndarray): Baseline covariate matrix of shape (n_individuals, n_features).
        weight (Optional[np.ndarray], optional): Observation weights (e.g. for case-cohort designs). Defaults to 1.0.
        time_scale (Optional[float], optional): Time discretization scale (e.g. 5 for 5-day intervals). Defaults to None.

    Returns:
        Dict[str, np.ndarray]:
            - 'y': Binary event indicator array of shape (n_person_time, 1).
            - 'x': Long-format covariate matrix where column 0 is discrete time, followed by covariates.
            - 'w': Observation weights aligned with person-time rows.
            - 'coord': Individual identifier integer array linking person-time rows to subjects.
    """
    if weight is None:
        weight = np.ones(y_time.shape[0])

    if time_scale is not None:
        yt = np.ceil(y_time / time_scale).flatten()
    else:
        yt = y_time.flatten()
    d_sk = np.array(y_status, dtype="int").flatten()
    t_uniq, t_inv = np.unique(yt, return_inverse=True)
    t_wide = np.tile(t_uniq, (y_time.shape[0], 1))
    d_wide = np.tile(np.zeros(t_uniq.shape[0]), (y_time.shape[0], 1))
    max_t = yt.max()

    # replace the time greater than survival time.
    for i in np.arange(t_wide.shape[1]):
        t_wide[t_wide[:, i] > yt, i] = max_t + 1
        d_wide[t_wide[:, i] == yt, i] = d_sk[t_wide[:, i] == yt]
        d_wide[t_wide[:, i] > yt, i] = 2
    
    # time
    t_wide = t_wide.reshape(t_wide.shape[0] * t_wide.shape[1], 1)
    t_wide = (t_wide[t_wide != max_t + 1]) # drop extra times
    t_wide = t_wide.reshape(t_wide.shape[0], 1)

    d_wide = d_wide.reshape(d_wide.shape[0] * d_wide.shape[1], 1)
    trn_y = (d_wide[d_wide != 2])
    trn_y = trn_y.reshape(trn_y.shape[0], 1)
    
    # x and delta expansion
    tx = np.repeat(x, t_inv + 1, axis=0)
    trn_x = np.hstack([t_wide, tx])

    # weight expansion
    weight_long = np.repeat(weight, t_inv + 1)
    weight_long = weight_long.reshape(weight_long.shape[0], 1)

    # get coords
    coords = np.repeat(np.arange(0, x.shape[0]), t_inv + 1)
    return {
        "y": trn_y, 
        "x": trn_x, 
        "w": weight_long, 
        "coord": coords
    }


def get_posterior_test(
    y_time: np.ndarray,
    y_status: np.ndarray,
    x: np.ndarray,
    time_scale: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """Generates evaluation grid in long-time format for out-of-sample posterior predictions.

    Constructs counterfactual cross-product of discrete evaluation times and subject covariates.

    Args:
        y_time (np.ndarray): Follow-up times defining the discrete time evaluation grid.
        y_status (np.ndarray): Event indicators (used for historical reference).
        x (np.ndarray): Baseline covariate matrix to evaluate.
        time_scale (Optional[float], optional): Time discretization scale. Defaults to None.

    Returns:
        Dict[str, np.ndarray]:
            - 'post_x': Evaluated covariate matrix with time in column 0.
            - 'coords': Subject coordinate identifier array.
    """
    if time_scale is not None:
        y_time = np.ceil(y_time/time_scale)
    uniq_times = np.unique(y_time)
    s0 = x.shape[0] # length
    s1 = x.shape[1] # width
    # create time range
    d1 = uniq_times
    d2 = np.tile(d1,s0).reshape(d1.shape[0]*s0,1)
    d3 = np.tile(x, d1.shape[0]).reshape(s0*d1.shape[0], s1)
    out = np.hstack([d2,d3])
    # coordinates
    coords = np.repeat(np.arange(0, x.shape[0]), d1.shape[0])
    return {"post_x": out, "coords":coords}

    
def get_causal_pdp(
    x: pd.DataFrame, 
    primary_col: Union[str, int] = "inpatient", 
    values: list = [0, 1],
    audit_support: bool = True,
    donor_seed: Optional[int] = 42
) -> Tuple[np.ndarray, Dict, List[str]]:
    """Generates counterfactual data for Causal G-Computation with support violation audit.
    
    Identifies dependent features or post-exposure mediators that have structural support
    or positivity violations with the primary exposure (e.g., deterministic zeros like LOS=0
    when Inpatient=0). Performs conditional donor resampling for dependent features while
    leaving baseline confounders intact, strictly avoiding off-support extrapolation.
    
    Args:
        x (pd.DataFrame): Baseline covariate dataframe.
        primary_col (Union[str, int]): Primary exposure column name or index.
        values (list, optional): Counterfactual intervention values. Defaults to [0, 1].
        audit_support (bool, optional): Whether to audit and flag support violations. Defaults to True.
        donor_seed (int, optional): Random seed for reproducible donor resampling. Defaults to 42.
        
    Returns:
        Tuple[np.ndarray, Dict, List[str]]:
            - out_sk: Stacked counterfactual covariate matrix.
            - c_idx: Index coordinate dictionary compatible with get_posterior_test.
            - dependent_features: List of audited features exhibiting support violations.
    """
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    col_name = primary_col if isinstance(primary_col, str) else x.columns[primary_col]
    a = x[col_name].values
    dependent_features = []
    
    if audit_support and len(values) >= 2:
        for col in x.columns:
            if col == col_name:
                continue
            vals_0 = x.loc[a == values[0], col].values
            vals_1 = x.loc[a == values[1], col].values
            if len(vals_0) == 0 or len(vals_1) == 0:
                continue
            min_0, max_0 = float(np.min(vals_0)), float(np.max(vals_0))
            min_1, max_1 = float(np.min(vals_1)), float(np.max(vals_1))
            mass_0 = float(np.mean(vals_0 == 0.0))
            mass_1 = float(np.mean(vals_1 == 0.0))
            
            mass_violation = (mass_0 >= 0.5 and mass_1 < 0.1) or (mass_1 >= 0.5 and mass_0 < 0.1)
            degenerate_violation = (min_0 == max_0 and min_1 != max_1) or (min_1 == max_1 and min_0 != max_0)
            non_overlap = (max_0 < min_1) or (max_1 < min_0)
            
            if mass_violation or degenerate_violation or non_overlap:
                dependent_features.append(col)
                warnings.warn(
                    f"[Causal Audit] Covariate '{col}' violates conditional support with primary exposure '{col_name}'. "
                    f"Re-routing '{col}' to Conditional G-Computation donor resampling to prevent off-support extrapolation.",
                    UserWarning
                )

    rng = np.random.default_rng(donor_seed)
    cf_dfs = []
    for val in values:
        x_cf = x.copy()
        x_cf[col_name] = float(val)
        for col in dependent_features:
            donor_vals = x.loc[a == val, col].values
            if len(donor_vals) > 0:
                if np.all(donor_vals == 0):
                    x_cf[col] = 0.0
                else:
                    target_mask = (a != val)
                    if target_mask.sum() > 0:
                        x_cf.loc[target_mask, col] = rng.choice(donor_vals, size=target_mask.sum(), replace=True)
        cf_dfs.append(x_cf.values)
        
    s2 = len(values)
    s3 = x.shape[0]
    out_sk = np.vstack(cf_dfs)
    c_idx = np.repeat(np.arange(s2), s3)
    c_idx = np.unique(c_idx, return_index=True, return_counts=True, return_inverse=True)
    c_idx = dict({"val": c_idx[0], "idx": c_idx[1], "coord": c_idx[2], "cnt": c_idx[3], "len": out_sk.shape[0]})
    return out_sk, c_idx, dependent_features


def compute_shift_pdp(
    x: Union[pd.DataFrame, np.ndarray],
    primary_col: Union[str, int],
    delta: float = 1.0,
    clip_to_range: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generates counterfactual datasets for a continuous shift intervention (Modified Treatment Policy).

    Contrasts the natural course cohort (Arm 0: X) against a cohort where every individual's
    continuous exposure is shifted by delta units (Arm 1: X + delta).

    Args:
        x (Union[pd.DataFrame, np.ndarray]): Baseline covariate matrix or DataFrame.
        primary_col (Union[str, int]): Continuous variable name or column index to shift.
        delta (float, optional): Increment added to primary_col. Defaults to 1.0.
        clip_to_range (bool, optional): Whether to clip shifted values to observed maximum. Defaults to False.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - out_sk: Stacked counterfactual matrix of shape (2 * N, n_features).
            - c_idx: Index coordinate dictionary demarcating Arm 0 and Arm 1.
    """
    if isinstance(x, np.ndarray):
        x_df = pd.DataFrame(x)
        col_name = primary_col if isinstance(primary_col, int) else int(primary_col)
    else:
        x_df = x.copy()
        col_name = primary_col

    # Arm 0: Natural course (observed X)
    arm0 = x_df.copy()

    # Arm 1: Shifted by delta
    arm1 = x_df.copy()
    shifted_vals = arm1[col_name].values.astype(np.float64) + float(delta)
    if clip_to_range:
        max_val = np.max(x_df[col_name].values)
        shifted_vals = np.minimum(shifted_vals, max_val)
    arm1[col_name] = shifted_vals

    s3 = x_df.shape[0]
    out_sk = np.vstack([arm0.values, arm1.values])
    c_idx = np.repeat(np.arange(2), s3)
    c_idx_uniq = np.unique(c_idx, return_index=True, return_counts=True, return_inverse=True)
    c_idx_dict = {
        "val": c_idx_uniq[0],
        "idx": c_idx_uniq[1],
        "coord": c_idx_uniq[2],
        "cnt": c_idx_uniq[3],
        "len": out_sk.shape[0]
    }
    return out_sk, c_idx_dict


def get_pdp(
    x: Union[np.ndarray, pd.DataFrame], 
    var_col: List[Union[int, str]] = [], 
    values: List[Union[int, float]] = [], 
    qt: List[float] = [0.25, 0.5, 0.75], 
    sample_n=None,
    method: str = "marginal",
    audit_support: bool = False
) -> Union[Tuple[np.ndarray, Dict], Tuple[np.ndarray, Dict, List[str]]]:
    """Generates data for Partial Dependency Plots or Causal G-Computation.

    Args:
        x (Union[np.ndarray, pd.DataFrame]): Covariate matrix or DataFrame.    
        var_col (List[Union[int, str]], optional): Covariate column to generate pdp values. Defaults to [].
        values (List[Union[int, float]], optional): Values to test on for each covariate. Defaults to [].
        qt (List[float], optional): Quantiles to generate values on if none are given. Defaults to [0.25, 0.5, 0.75].
        sample_n (int, optional): Sample size of large dataset. Useful for working with large datasets. Defaults to None.
        method (str, optional): 'marginal' for standard PDP, or 'causal' for support-audited G-Computation. Defaults to 'marginal'.
        audit_support (bool, optional): If True, audits for positivity/support violations. Defaults to False.

    Returns:
        Union[Tuple[np.ndarray, Dict], Tuple[np.ndarray, Dict, List[str]]]: PDP covariate matrix and tracking dictionary.
    """
    if method == "causal" or audit_support:
        val_list = values[0] if (len(values) > 0 and values[0] is not None) else [0, 1]
        primary = var_col[0] if len(var_col) > 0 else 0
        return get_causal_pdp(x, primary_col=primary, values=val_list, audit_support=True)

    if isinstance(x, pd.DataFrame):
        # Resolve column names to integer indices if necessary
        resolved_var_col = []
        for v in var_col:
            if isinstance(v, str):
                resolved_var_col.append(x.columns.get_loc(v))
            else:
                resolved_var_col.append(int(v))
        var_col = resolved_var_col
        x = x.values

    if len(var_col) > 2:
        print("only upto 2 variables can be used at a time")
        pass

    if sample_n is not None:
        rn_idx = np.random.choice(np.arange(0, x.shape[0]), sample_n, replace=False)   
        x = x[rn_idx,:]     

    var_vals = []
    for idx, var in enumerate(var_col):
        if idx >= len(values) or values[idx] is None:
            val = np.quantile(x[:,var], qt)
            var_vals.append(val)
        else:
            var_vals.append(values[idx])
    s1 = len(var_col)
    if s1 > 1:
        cart = np.dstack(np.meshgrid(*var_vals)).reshape(-1, s1)
    else:
        cart = np.array(var_vals[0]).reshape(-1,1)
    s2 = cart.shape[0] # number of vals
    s3 = x.shape[0] # number of obs
    out_sk = np.tile(x, (s2,1)) # tile x into blocks of number of vals
    l_cart = np.repeat(cart, s3, axis=0) # repeat vals number of obs times
    for idx, var in enumerate(var_col): # fill the columns of the x*blocks with new values
        out_sk[:,var] = l_cart[:,idx]
    # index
    c_idx = np.repeat(np.arange(s2),s3)
    c_idx = np.unique(c_idx, return_index=True, return_counts=True, return_inverse=True)
    c_idx = dict({"val":c_idx[0], "idx":c_idx[1], "coord":c_idx[2], "cnt":c_idx[3], "len":out_sk.shape[0]})

    return out_sk, c_idx


def get_sv_prob(
    post: Union[np.ndarray, xr.DataArray, xr.Dataset],
    coords: np.ndarray
) -> Dict[str, np.ndarray]:
    """Generates discrete-time hazard and cumulative survival probability estimates.

    Computes S(t) = \\prod_{s=1}^t (1 - h(s)) across all posterior draws and subjects.

    Args:
        post (Union[np.ndarray, xr.DataArray, xr.Dataset]): Posterior predictive hazard draws.
        coords (np.ndarray): Subject coordinate indices to reshape predictions per subject and time.

    Returns:
        Dict[str, np.ndarray]:
            - 'prob': Hazard probability array of shape (n_draws, n_subjects, n_time_points).
            - 'sv': Cumulative survival probability array of shape (n_draws, n_subjects, n_time_points).
    """
    n, k = np.unique(coords, return_counts=True)
    
    if "Dataset" in str(post):
        prob = post.mu.values.reshape(-1, n.shape[0], k[0])
    elif hasattr(post, "values"):
        prob = post.values.reshape(-1, n.shape[0], k[0])
    else:
        prob = post.reshape(-1, n.shape[0], k[0])
    sv = np.cumprod(1.0 - prob, axis=2)
    return {
        "prob": prob, 
        "sv": sv
    }


def get_sv_mean_quant(
    sv: np.ndarray, 
    msk: np.ndarray,
    draws: bool = True, 
    qntile: List[float] = [0.025, 0.975]
) -> Dict[str, np.ndarray]:
    """Generates mean and quantile estimates of Survival Probabilities.

    Args:
        sv (np.ndarray): Survival probabilities of shape (n_draws, n_obs, n_times).
        msk (np.ndarray): Boolean mask selecting subpopulation of interest.
        draws (bool, optional): If True, aggregates across posterior draws. Defaults to True.
        qntile (List[float], optional): Credible interval quantiles. Defaults to [0.025, 0.975].

    Returns:
        Dict[str, np.ndarray]: Mean and quantile curves for masked and unmasked groups.
    """
    sv_mt = sv[:, msk, :]
    sv_mt_m = sv_mt.mean(axis=1) # mean per draw
    if draws:
        sv_mt_q = np.quantile(sv_mt_m, qntile, axis=0)
        sv_mt_m = sv_mt_m.mean(axis=0) # mean over draws

    sv_mf = sv[:, ~msk, :]
    sv_mf_m = sv_mf.mean(axis=1)
    if draws:
        sv_mf_q = np.quantile(sv_mf_m, qntile, axis=0)
        sv_mf_m = sv_mf_m.mean(axis=0)

    return {
        "mt_m": sv_mt_m, 
        "mt_q": sv_mt_q, 
        "mf_m": sv_mf_m, 
        "mf_q": sv_mf_q
    }


def pdp_diff_metric(
    pdp_val: Dict[str, np.ndarray], 
    idx: int, 
    qntile: List[float] = [0.025, 0.975]
) -> Dict[str, np.ndarray]:
    """Generates posterior estimate of marginal survival difference between counterfactual groups.

    Args:
        pdp_val (Dict[str, np.ndarray]): Posterior output dictionary containing 'sv'.
        idx (int): Split index demarcating counterfactual groups in stacked PDP matrix.
        qntile (List[float], optional): Quantiles for credible interval. Defaults to [0.025, 0.975].

    Returns:
        Dict[str, np.ndarray]:
            - 'diff_m': Posterior mean survival difference curve across time points.
            - 'diff_q': Posterior credible interval quantiles of the difference curve.
    """
    diff = (pdp_val["sv"][:, :idx, :] - pdp_val["sv"][:, idx:, :]).mean(axis=1) 
    d_m = diff.mean(axis=0)
    d_q = np.quantile(diff, qntile, axis=0)
    return {
        "diff_m": np.round(d_m, 3),
        "diff_q": np.round(d_q, 3)
    }


def pdp_rr_metric(
    pdp_val: Dict[str, np.ndarray], 
    idx: int, 
    qntile: List[float] = [0.025, 0.975]
) -> Dict[str, np.ndarray]:
    """Generates posterior Hazard Ratio / Risk Ratio between counterfactual intervention groups.

    Args:
        pdp_val (Dict[str, np.ndarray]): Posterior output dictionary containing 'prob'.
        idx (int): Split index demarcating counterfactual groups in stacked PDP matrix.
        qntile (List[float], optional): Quantiles for credible interval. Defaults to [0.025, 0.975].

    Returns:
        Dict[str, np.ndarray]:
            - 'rr_m': Posterior mean hazard ratio curve across time points.
            - 'rr_q': Posterior credible interval quantiles of the hazard ratio curve.
    """
    r = (pdp_val["prob"][:, idx:, :] / pdp_val["prob"][:, :idx, :]).mean(axis=1) 
    r_m = r.mean(axis=0)
    r_q = np.quantile(r, qntile, axis=0)
    return {
        "rr_m": np.round(r_m, 3), 
        "rr_q": np.round(r_q, 3)
    }


def compute_hazard_and_risk_difference(
    pdp_val: Dict[str, np.ndarray],
    idx: int,
    event_weights: Optional[np.ndarray] = None,
    qntile: List[float] = [0.025, 0.975],
    eps: float = 1e-7
) -> Dict[str, Any]:
    """Computes event-weighted and unweighted Hazard Ratios and Risk Differences from counterfactual predictions.

    Contrasts Intervention Arm 1 (obs >= idx) over Reference Arm 0 (obs < idx).

    Calculates:
    - Event-weighted summary Hazard Ratio (default, Cox-aligned via complementary log-log link).
    - Unweighted summary Hazard Ratio across follow-up intervals.
    - Time-varying Hazard Ratio curve HR(t) via complementary log-log link.
    - Time-varying discrete risk ratio curve RR_discrete(t) = h1(t) / h0(t) (not averaged).
    - Time-varying Risk Difference curve RD(t) = Risk_1(t) - Risk_0(t) = S_0(t) - S_1(t) (1 over 0).
    - Survival Difference curve S_1(t) - S_0(t) = -RD(t).
    - Event-weighted and unweighted summary Risk Differences.
    - Final horizon Risk Difference RD(t_end).
    - Restricted Mean Survival Time (RMST) difference.

    Args:
        pdp_val (Dict[str, np.ndarray]): Dictionary containing 'prob' (hazards) and 'sv' (survival curves).
            Arrays have shape (n_draws, 2 * N, n_times).
        idx (int): Split index demarcating Reference Arm 0 (:idx) and Intervention Arm 1 (idx:).
        event_weights (Optional[np.ndarray], optional): Normalized event weights per interval.
            If None, defaults to uniform interval weighting.
        qntile (List[float], optional): Credible interval quantiles. Defaults to [0.025, 0.975].
        eps (float, optional): Epsilon threshold for numerical safety. Defaults to 1e-7.

    Returns:
        Dict[str, Any]: Comprehensive dictionary containing summary statistics, credible intervals, and draw arrays.
    """
    prob = pdp_val["prob"]
    sv = pdp_val["sv"]
    n_times = prob.shape[2]

    safe_eps = max(eps, 1e-4)
    h0 = np.clip(prob[:, :idx, :].mean(axis=1), safe_eps, 1.0 - safe_eps)
    h1 = np.clip(prob[:, idx:, :].mean(axis=1), safe_eps, 1.0 - safe_eps)

    # Population marginal survival per draw across subjects
    s0 = sv[:, :idx, :].mean(axis=1)
    s1 = sv[:, idx:, :].mean(axis=1)

    # 1. Time-varying Hazard Ratio via cloglog link: log(1 - h1) / log(1 - h0)
    hr_raw = -np.log(1.0 - h1) / -np.log(1.0 - h0)
    hr_by_time = np.clip(np.nan_to_num(hr_raw, nan=1.0, posinf=50.0, neginf=0.02), 0.02, 50.0)

    # 2. Time-varying raw discrete risk ratio: h1 / h0 (not averaged across time)
    discrete_raw = h1 / h0
    discrete_rr_by_time = np.clip(np.nan_to_num(discrete_raw, nan=1.0, posinf=50.0, neginf=0.02), 0.02, 50.0)

    # Interval event weights
    if event_weights is None:
        w = np.ones(n_times, dtype=np.float64) / n_times
    else:
        w = np.asarray(event_weights, dtype=np.float64)
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones(n_times, dtype=np.float64) / n_times

    # 3. Event-weighted summary Hazard Ratio (Cox-aligned geometric mean across intervals)
    log_hr = np.log(hr_by_time)
    weighted_log_hr_draws = np.dot(log_hr, w)
    weighted_hr_draws = np.exp(weighted_log_hr_draws)

    # 4. Unweighted summary Hazard Ratio (geometric across time intervals)
    unweighted_log_hr_draws = np.mean(log_hr, axis=1)
    unweighted_hr_draws = np.exp(unweighted_log_hr_draws)

    # 5. Risk Difference (1 over 0: excess event risk): Risk_1(t) - Risk_0(t) = S_0(t) - S_1(t)
    rd_by_time = s0 - s1
    surv_diff_by_time = s1 - s0

    # 6. Summary Risk Differences
    weighted_rd_draws = np.dot(rd_by_time, w)
    unweighted_rd_draws = rd_by_time.mean(axis=1)
    horizon_rd_draws = rd_by_time[:, -1]

    # 7. Restricted Mean Survival Time (RMST) Difference
    rmst_diff_draws = surv_diff_by_time.sum(axis=1)

    return {
        # Summary Hazard Ratios (Cox-aligned posterior expectation of log-hazard ratio)
        "hazard_ratio_weighted_mean": float(np.exp(np.mean(weighted_log_hr_draws))),
        "hazard_ratio_weighted_geometric_mean": float(np.exp(np.mean(weighted_log_hr_draws))),
        "hazard_ratio_weighted_median": float(np.median(weighted_hr_draws)),
        "hazard_ratio_weighted_arithmetic_mean": float(np.mean(weighted_hr_draws)),
        "hazard_ratio_weighted_ci": np.quantile(weighted_hr_draws, qntile).tolist(),
        "hazard_ratio_weighted_draws": weighted_hr_draws,
        "hazard_ratio_unweighted_mean": float(np.exp(np.mean(unweighted_log_hr_draws))),
        "hazard_ratio_unweighted_geometric_mean": float(np.exp(np.mean(unweighted_log_hr_draws))),
        "hazard_ratio_unweighted_median": float(np.median(unweighted_hr_draws)),
        "hazard_ratio_unweighted_arithmetic_mean": float(np.mean(unweighted_hr_draws)),
        "hazard_ratio_unweighted_ci": np.quantile(unweighted_hr_draws, qntile).tolist(),
        "hazard_ratio_unweighted_draws": unweighted_hr_draws,
        # Time-varying HR and discrete risk ratio curves
        "hazard_ratio_curve_mean": np.round(hr_by_time.mean(axis=0), 3),
        "hazard_ratio_curve_ci": np.round(np.quantile(hr_by_time, qntile, axis=0), 3),
        "hazard_ratio_curve_draws": hr_by_time,
        "discrete_risk_ratio_mean": np.round(discrete_rr_by_time.mean(axis=0), 3),
        "discrete_risk_ratio_ci": np.round(np.quantile(discrete_rr_by_time, qntile, axis=0), 3),
        "discrete_risk_ratio_draws": discrete_rr_by_time,
        # Risk Differences (1 over 0: excess event risk)
        "risk_difference_mean": np.round(rd_by_time.mean(axis=0), 3),
        "risk_difference_ci": np.round(np.quantile(rd_by_time, qntile, axis=0), 3),
        "risk_difference_draws": rd_by_time,
        "risk_difference_weighted_mean": float(np.mean(weighted_rd_draws)),
        "risk_difference_weighted_ci": np.quantile(weighted_rd_draws, qntile).tolist(),
        "risk_difference_unweighted_mean": float(np.mean(unweighted_rd_draws)),
        "risk_difference_unweighted_ci": np.quantile(unweighted_rd_draws, qntile).tolist(),
        "risk_difference_horizon_mean": float(np.mean(horizon_rd_draws)),
        "risk_difference_horizon_ci": np.quantile(horizon_rd_draws, qntile).tolist(),
        # Marginal Counterfactual Survival Curves S1(t) and S0(t)
        "survival_1_mean": np.round(s1.mean(axis=0), 3),
        "survival_1_ci": np.round(np.quantile(s1, qntile, axis=0), 3),
        "survival_0_mean": np.round(s0.mean(axis=0), 3),
        "survival_0_ci": np.round(np.quantile(s0, qntile, axis=0), 3),
        "s1_curve_mean": np.round(s1.mean(axis=0), 3),
        "s1_curve_ci": np.round(np.quantile(s1, qntile, axis=0), 3),
        "s0_curve_mean": np.round(s0.mean(axis=0), 3),
        "s0_curve_ci": np.round(np.quantile(s0, qntile, axis=0), 3),
        # Survival Difference (S1 - S0)
        "survival_difference_mean": np.round(surv_diff_by_time.mean(axis=0), 3),
        "survival_difference_ci": np.round(np.quantile(surv_diff_by_time, qntile, axis=0), 3),
        # Restricted Mean Survival Time Difference
        "rmst_difference_mean": float(np.mean(rmst_diff_draws)),
        "rmst_difference_ci": np.quantile(rmst_diff_draws, qntile).tolist(),
        # Weights used
        "event_weights": w,
        # Backward compatibility aliases
        "hazard_ratio_mean": np.round(hr_by_time.mean(axis=0), 3),
        "hazard_ratio_ci": np.round(np.quantile(hr_by_time, qntile, axis=0), 3),
        "difference_mean": np.round(surv_diff_by_time.mean(axis=0), 3),
        "difference_ci": np.round(np.quantile(surv_diff_by_time, qntile, axis=0), 3),
    }


# ==========================================
# Modern API Aliases (100% Backward Compatible)
# ==========================================
SurvBART = BartSurvModel
format_survival_data = get_surv_pre_train
format_prediction_data = get_posterior_test
compute_survival_curves = get_sv_prob
compute_survival_quantiles = get_sv_mean_quant
compute_pdp = get_pdp
compute_causal_pdp = get_causal_pdp
get_shift_pdp = compute_shift_pdp
compute_pdp_difference = pdp_diff_metric
compute_hazard_ratio = pdp_rr_metric
pdp_hr_rd_metric = compute_hazard_and_risk_difference

__all__ = [
    # Estimator
    "SurvBART",
    "BartSurvModel",
    # Data transformation & pre-training
    "format_survival_data",
    "get_surv_pre_train",
    "format_prediction_data",
    "get_posterior_test",
    # Survival curves & metrics
    "compute_survival_curves",
    "get_sv_prob",
    "compute_survival_quantiles",
    "get_sv_mean_quant",
    # Partial dependence, shift interventions & causal g-computation
    "compute_pdp",
    "get_pdp",
    "compute_causal_pdp",
    "get_causal_pdp",
    "compute_shift_pdp",
    "get_shift_pdp",
    "compute_pdp_difference",
    "pdp_diff_metric",
    "compute_hazard_ratio",
    "pdp_rr_metric",
    "compute_hazard_and_risk_difference",
    "pdp_hr_rd_metric",
]
    