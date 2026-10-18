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

    Implements discrete-time probit hazard modeling using sum-of-trees ensemble.
    Compatible with scikit-survival style data formats and standard Bayesian workflows.
    """
    _model_type = "BART_Survival"
    version = "0.2"

    def __init__(
        self,
        model_config: Optional[Dict] = None, 
        sampler_config: Optional[Dict] = None
    ):
        """Initialize BartSurvModel / SurvBART.

        Args:
            model_config (Dict, optional): Prior and model configuration (e.g. trees, split_rules, split_prior).
            sampler_config (Dict, optional): MCMC sampling parameters (e.g. draws, tune, cores, chains, thin).
        """
        sampler_config = self.get_default_sampler_config() if sampler_config is None else sampler_config
        self.sampler_config = sampler_config
        model_config = self.get_default_model_config() if model_config is None else model_config
        self.model_config = model_config  # parameters for priors etc.
        self.model = None  # Set by build_model
        self.idata: Optional[az.InferenceData] = None  # idata is generated during fitting
        self.is_fitted_ = False



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
        self.M = self.model_config.get("trees", 20)

        # Probit hazard link offset
        self.offset = sp.norm.ppf(np.mean(self.y))
        mcoords = {"xvars": self.predictor_names}
        k = self.sampler_config.get("k", 2)
        update_fraction = self.sampler_config.get("update_fraction", .1)
        min_node_size = self.sampler_config.get("min_node_size", 5)
        unique_midpoints = self.sampler_config.get("unique_midpoints", False)
        self.use_sparsity = self.sampler_config.get("use_sparsity", False)
        tune = self.sampler_config.get("tune", 1000)
        default_warmup = min(300, max(50, tune // 2))
        sparsity_warmup = self.sampler_config.get("sparsity_warmup", default_warmup)
        
        # initalize model
        alpha = self.model_config.get("alpha", 0.95)
        beta = self.model_config.get("beta", 2.0)
        with pm.Model(coords=mcoords) as self.model:    
            # dummy place-holder - data actually set in the step backdoor
            trees = pmb.BART("trees", X=np.zeros((1,self.X.shape[1])), Y=np.zeros(1), m=self.M, alpha=alpha, beta=beta)
            prob = 0.5 * (1.0 + pt.erf(trees / pt.sqrt(2.0)))
            obs = pm.Bernoulli("obs", p=prob, observed=self.y)
        
            # initiallize the step
            self.custom_step = FastSurvMHBART(
                    vars=[trees], 
                    use_sparsity=self.use_sparsity, 
                    sparsity_warmup=sparsity_warmup,
                    target_x = self.X,
                    target_y = self.y,
                    update_fraction=update_fraction, # number of trees to update
                    k=k, 
                    min_node_size=min_node_size,
                    unique_midpoints=unique_midpoints,
                    return_estimate = False
                    )
            self.offset = self.custom_step.offset
            


    def sample_model(
        self, 
        **kwargs
    ) -> az.InferenceData:
        """Initiates training/sampling of the model.

        Returns:
            az.InferenceData: Posterior data collected from training the model.
        """
        if self.model is None:
            raise RuntimeError(
                "The model hasn't been built yet, call .build_model() first or call .fit() instead."
            )
        

        draws = self.sampler_config.get("draws", 2000)
        tune = self.sampler_config.get("tune", 1000)
        chains = self.sampler_config.get("chains", 4)
        cores = self.sampler_config.get("cores", 4)
        thin = self.sampler_config.get("thin", 1)
        thin_idata = self.sampler_config.get("thin_idata", False)

        with self.model:
            sampler_args = {**self.sampler_config, **kwargs}
            pbar = sampler_args.get("progressbar", kwargs.get("progressbar", True))
            idata = pm.sample(
                draws=draws, 
                tune=tune, 
                chains=chains, 
                cores=cores,
                step=self.custom_step,  # If None, PyMC handles it
                compute_convergence_checks=False,
                progressbar=pbar,
                discard_tuned_samples=True
            )
        if thin > 1 and thin_idata:
            idata = idata.sel(draw=slice(None, None, thin))
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
        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)

        # sample the model
        self.idata = self.sample_model(**sampler_config)
        self.is_fitted_ = True

        self.all_trees = list(self.idata.sample_stats)
        return self.idata

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

        thin = self.sampler_config.get("thin", 1)
        thin_idata = self.sampler_config.get("thin_idata", False)
        eff_thin = 1 if thin_idata else thin
        preds = predict_out_of_sample(self.idata, X_pred, offset, m_trees=self.M, thin=eff_thin)
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
        """Default model prior and tree ensemble configuration.

        Returns:
            Dict[str, Any]: Default configuration dict with 50 trees, alpha=0.95, beta=2.0.
        """
        return {
            "trees": 50,
            "alpha": 0.95,
            "beta": 2.0,
            "split_rules": None,
            "split_prior": None,
        }

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """Default Sampler Configuration.

        Returns:
            Dict: Default Sampler Configuration.
        """ 
        sampler_config: Dict = {
            "draws": 100,
            "tune": 100,
            "cores": 2,
            "chains": 2,
            "compute_convergence_checks": False
        }
        return sampler_config

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
compute_pdp_difference = pdp_diff_metric
compute_hazard_ratio = pdp_rr_metric

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
    # Partial dependence & causal g-computation
    "compute_pdp",
    "get_pdp",
    "compute_causal_pdp",
    "get_causal_pdp",
    "compute_pdp_difference",
    "pdp_diff_metric",
    "compute_hazard_ratio",
    "pdp_rr_metric",
]
    