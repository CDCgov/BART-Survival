
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
import scipy as sp
import warnings
import pytensor.tensor as pt
from pymc_bart.utils import _sample_posterior
import cloudpickle as cpkl
from numpy.random import RandomState
import multiprocessing as mp
from numba import njit, prange
import copy
import os
import copy

def patch_pymc_bart(num_trees=50, max_nodes=255):
    # ==========================================
    # 1. THE VAULT (Prevents Infinite Recursion)
    # ==========================================
    # We capture the TRUE native method exactly once and permanently lock it.
    if not hasattr(pmb.pgbart.PGBART, "_true_native_astep"):
        pmb.pgbart.PGBART._true_native_astep = pmb.pgbart.PGBART.astep

    print(f"Applying 1D-Flattened Patch (Trees: {num_trees}, Max Nodes: {max_nodes})...")

    NUM_TREES = num_trees
    MAX_NODES = max_nodes

    pmb.pgbart.PGBART.stats_dtypes = [{
        "variable_inclusion": object, 
        "tune": bool,                  
        "flat_f": object,              
        "flat_t": object,
        "flat_l": object,
        "flat_r": object,
        "flat_v": object,
    }]
    pmb.pgbart.PGBART.stats_dtypes_shapes = {}

    def _patched_astep(self, *args, **kwargs):
        current_pid = os.getpid()
        
        # --- THE FAST-FORWARD ENTROPY BREAKOUT ---
        if getattr(self, "_entropy_pid", None) != current_pid:
            np.random.seed(current_pid % 4294967295)
            if hasattr(self, "rng") and hasattr(self.rng, "random"):
                burn_count = (current_pid % 100) * 5000 + 1234
                _ = self.rng.random(burn_count)
            self._entropy_pid = current_pid

        # --- RUN SAMPLING FROM THE VAULT ---
        # (We strictly call the locked native method, never the dynamic one)
        res = self._true_native_astep(*args, **kwargs)
        
        point, stats = res
        target_stats = stats[0] if isinstance(stats, list) else stats

        # --- PRE-ALLOCATE FLAT 1D ARRAYS ---
        f_1d = np.full(NUM_TREES * MAX_NODES, -1, dtype=np.int16)
        l_1d = np.full(NUM_TREES * MAX_NODES, -1, dtype=np.int16)
        r_1d = np.full(NUM_TREES * MAX_NODES, -1, dtype=np.int16)
        t_1d = np.zeros(NUM_TREES * MAX_NODES, dtype=np.float32)
        v_1d = np.zeros(NUM_TREES * MAX_NODES, dtype=np.float32)
        
        if not self.tune:
            def extract_trees(obj):
                if hasattr(obj, "tree_structure"): return [obj]
                found = []
                if isinstance(obj, (list, tuple, np.ndarray)):
                    for item in obj: found.extend(extract_trees(item))
                return found
            
            active_forest = getattr(getattr(self, "bart", self), "trees", getattr(self, "all_trees", []))
            true_trees = extract_trees(active_forest)
            
            if len(true_trees) >= NUM_TREES:
                true_trees = true_trees[-NUM_TREES:]
                
            for tree_idx, tree in enumerate(true_trees):
                if tree_idx >= NUM_TREES: break
                offset = tree_idx * MAX_NODES
                sorted_nodes = sorted(tree.tree_structure.items())
                idx_map = {heap_idx: dense_idx for dense_idx, (heap_idx, _) in enumerate(sorted_nodes)}
                
                for dense_idx, (heap_idx, node) in enumerate(sorted_nodes):
                    if dense_idx >= MAX_NODES: break
                    write_idx = offset + dense_idx
                    
                    if node.is_leaf_node():
                        v_1d[write_idx] = float(np.atleast_1d(node.value)[0])
                    else:
                        f_1d[write_idx] = int(node.idx_split_variable)
                        t_1d[write_idx] = float(node.value)
                        l_1d[write_idx] = idx_map.get(2 * heap_idx + 1, -1)
                        r_1d[write_idx] = idx_map.get(2 * heap_idx + 2, -1)
                        
        target_stats.update({
            "flat_f": f_1d, "flat_t": t_1d, "flat_l": l_1d, "flat_r": r_1d, "flat_v": v_1d
        })
        
        # ==========================================
        # THE MULTIPROCESSING LISTPROXY HIJACK
        # ==========================================
        if getattr(self, "_hijacked_proxy", False) is False:
            print(f"\n[DEBUG] BEFORE HIJACK - Core Type: {type(self.all_trees)}")

            local_nested_list = []
            for dim_trees in getattr(self, "all_trees", []):
                local_nested_list.append(list(dim_trees))
            self.all_trees = local_nested_list
            self._hijacked_proxy = True
            print(f"[DEBUG] AFTER HIJACK - Core Type: {type(self.all_trees)}\n")
        
        print(f"[DEBUG] AFTER HIJACK - Core Type: {type(self.all_trees)}\n")
        return point, stats

    # Safely assign the patched method globally
    pmb.pgbart.PGBART.astep = _patched_astep
    pmb.pgbart.PGBART._is_live_patched = True


@njit
def _eval_tree(x_row, f_flat, t_flat, l_flat, r_flat, v_flat, root_idx, max_nodes, is_one_hot):
    node = root_idx
    limit = root_idx + max_nodes 
    num_feats = len(x_row)
    
    while True:
        if node < 0 or node >= len(f_flat): return 0.0 
        feat = f_flat[node]
        if feat == -1: return v_flat[node]
        if feat < 0 or feat >= num_feats: return v_flat[node]
            
        # ==========================================
        # THE FIX: DYNAMIC SPLIT RULES
        # ==========================================
        if is_one_hot[feat] == 1:
            # One-Hot Rule: Strict Equality
            if x_row[feat] == t_flat[node]:
                rel_next = l_flat[node]
            else:
                rel_next = r_flat[node]
        else:
            # Continuous Rule: Standard Inequality
            if x_row[feat] <= t_flat[node]:
                rel_next = l_flat[node]
            else:
                rel_next = r_flat[node]
        # ==========================================
            
        if rel_next <= 0: return v_flat[node]
        node = root_idx + rel_next
        if node >= limit: return v_flat[node]

@njit(parallel=True, fastmath=True)
def fast_bart_predict(X, F, T, L, R, V, offset, n_trees, max_nodes, is_one_hot):
    n_total_draws = F.shape[0]
    n_obs = X.shape[0]
    preds = np.zeros((n_total_draws, n_obs), dtype=np.float32)
    
    for d in prange(n_total_draws):
        for i in range(n_obs):
            draw_sum = 0.0
            for tr in range(n_trees):
                root_idx = tr * max_nodes
                draw_sum += _eval_tree(X[i], F[d], T[d], L[d], R[d], V[d], root_idx, max_nodes, is_one_hot)
            preds[d, i] = draw_sum + offset
    return preds


# test
class BartSurvModel():
    """BART Survival Model

    Returns:
        _type_: BartSurvModel
    """
    
    _model_type = "BART_Survival"
    version = "0.1"
    def __init__(
        self,
        model_config: Dict = None, 
        sampler_config: Dict = None
    ):
        """Initialize BartSurvModel.

        Args:
            model_config (Dict, optional): model configuration parameters. Defaults to None.
            sampler_config (Dict, optional): model sampling parameters. Defaults to None.
        """
        print("HI2")
        sampler_config = (self.get_default_sampler_config() if sampler_config is None else sampler_config)
        self.sampler_config = sampler_config
        model_config = self.get_default_model_config() if model_config is None else model_config
        self.model_config = model_config  # parameters for priors etc.
        self.model = None  # Set by build_model
        self.idata: Optional[az.InferenceData] = None  # idata is generated during fitting
        self.is_fitted_ = False

        # add patch
        
        patch_pymc_bart(self.model_config.get("trees", 50), self.model_config.get("num_nodes",255))

    def build_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray,                        
        weights: np.ndarray = None,
        coords: np.ndarray = None,
        predictor_names: list = None,
        **kwargs
    ): 
        """Builds the optimized PYMC base model with flexible weight scaling."""
        self._generate_and_preprocess_model_data(X, y, weights, predictor_names)
        
        # 2. Flexible Weight Routing
        weight_scaling = self.model_config.get("weight_scaling","ess")
        w_raw = self.weights.flatten()
        if weight_scaling == "ess":
            # Default: Scale to Kish's Effective Sample Size (Safest for Bayesian CIs)
            sum_w = np.sum(w_raw)
            sum_w_sq = np.sum(w_raw**2)
            if sum_w_sq > 0:
                ess = (sum_w ** 2) / sum_w_sq
                final_weights = w_raw * (ess / sum_w)
            else:
                final_weights = w_raw
        elif weight_scaling == "mean":
            # Scale weights so they sum exactly to the number of observations (N_obs)
            # Useful if the user wants standard Machine Learning loss-weighting
            mean_w = np.mean(w_raw)
            final_weights = w_raw / mean_w if mean_w > 0 else w_raw
        elif weight_scaling == "raw":
            # Total User Control: Use the exact weights provided, no questions asked.
            # (Users must ensure their weights don't break the Bayesian information identity)
            final_weights = w_raw
        else:
            raise ValueError(f"Unknown weight_scaling '{weight_scaling}'. Use 'ess', 'mean', or 'raw'.")
        final_weights = final_weights.reshape(-1, 1)

        # Get Model Configs
        SPLIT_RULES = [eval(rule) for rule in self.model_config.get("split_rules", None)]
        SPLIT_PRIOR = self.model_config.get("split_prior")
        M = self.model_config.get("trees", 20)
        ALPHA= self.model_config.get("alpha", .95)
        BETA = self.model_config.get("beta", 2)
        

        # Offset
        eps = 1e-6
        clipped_mean = np.clip(np.mean(self.y), eps, 1 - eps)
        self.offset = sp.stats.norm.ppf(clipped_mean)
        
        mcoords = {"xvars": self.predictor_names}

        with pm.Model(coords=mcoords) as self.model:    
            self.model.add_coord("p_obs", coords)
            
            # Load the dynamically scaled weights into the graph
            w = pm.Data("weights", final_weights)
            
            
            f = pmb.BART(
                "f", 
                X=self.X, 
                Y=self.y.flatten(), 
                m=M, 
                split_rules=SPLIT_RULES, 
                split_prior=SPLIT_PRIOR,
                alpha=ALPHA,
                beta=BETA
            )
            
            z = f + self.offset
            mu = pm.math.invprobit(z)
            
            base_logp = pm.logp(pm.Bernoulli.dist(p=mu), self.y.flatten())
            pm.Potential("weighted_obs", w.flatten() * base_logp)
            
        return self.model

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
        with self.model:
            sampler_args = {**self.sampler_config, **kwargs}
            idata = pm.sample(**sampler_args)
        idata = self.set_idata_attrs(idata)
        return idata

    def fit(
        self,
        y: np.ndarray, #only event outcome
        X: np.ndarray, #time+covariate expanded
        weights: np.ndarray=None, #long form weights
        coords: np.ndarray=None,
        progressbar: bool = True,
        predictor_names: List[str] = None,
        random_seed: RandomState = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Call to build and train the data.

        Args:
            y (np.ndarray): Event status in long-time form.
            X (np.ndarray): Covariate matrix in long-time form.
            weights (np.ndarray): Weights associated with each observation.
            coords (np.ndarray): Coordinates associated with long-time form. 
            progressbar (bool, optional): Displays training progress. Defaults to True.
            predictor_names (List[str], optional): Names of covariates in X matrix. Defaults to None.
            random_seed (RandomState, optional): Seed. Defaults to None.

        Returns:
            az.InferenceData: _description_
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

        # get tree struct
        # self.all_trees = list(self.model.f.owner.op.all_trees)
        # get x names

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning#,
            )

            self.idata["bart_metadata"] = xr.Dataset({
                "predictor_names": (("feature",), self.predictor_names),
                "offset":          ((), self.offset), # Empty tuple means it's a scalar
                "trees_M":         ((), self.model_config.get("trees", 20))
            })

            # --- POST PROCESSING EXTRACTION ---
            # --- POST PROCESSING EXTRACTION ---
            
            
 
            
            self.idata["bart_metadata"] = xr.Dataset({
                "predictor_names": (("feature",), self.predictor_names),
                "offset":          ((), self.offset),
                "trees_M":         ((), self.model_config.get("trees", 20))
            })
            
            raw_F = self.idata.sample_stats["flat_f"].values
            raw_T = self.idata.sample_stats["flat_t"].values
            raw_L = self.idata.sample_stats["flat_l"].values
            raw_R = self.idata.sample_stats["flat_r"].values
            raw_V = self.idata.sample_stats["flat_v"].values

            n_chains, n_draws, _ = raw_F.shape
            total_samples = n_chains * n_draws
            
            NUM_TREES = self.model_config.get("trees", 50)
            MAX_NODES = self.model_config.get("num_nodes", 255)
            print(NUM_TREES)
            print(MAX_NODES)
            self.F = raw_F.reshape(total_samples, NUM_TREES, MAX_NODES).astype(np.int16)
            self.T = raw_T.reshape(total_samples, NUM_TREES, MAX_NODES).astype(np.float32)
            self.L = raw_L.reshape(total_samples, NUM_TREES, MAX_NODES).astype(np.int16)
            self.R = raw_R.reshape(total_samples, NUM_TREES, MAX_NODES).astype(np.int16)
            self.V = raw_V.reshape(total_samples, NUM_TREES, MAX_NODES).astype(np.float32)

            self.idata["bart_trees"] = xr.Dataset({
                "feature":   (("sample", "tree", "node"), self.F),
                "threshold": (("sample", "tree", "node"), self.T),
                "left":      (("sample", "tree", "node"), self.L),
                "right":     (("sample", "tree", "node"), self.R),
                "value":     (("sample", "tree", "node"), self.V)
            })

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
        return idata
    

    def predict_posterior(self, X_pred, coords=None, extend_idata=True, **kwargs):
        if not hasattr(self, "idata") or "bart_trees" not in self.idata:
            raise ValueError("Model must be fitted before predicting.")

        # 1. Force exact column alignment
        if hasattr(X_pred, "columns"):  
            if hasattr(self, "predictor_names"):
                X_pred = X_pred[self.predictor_names] 
            X_np = X_pred.values.astype(np.float32)
        else:
            X_np = np.asarray(X_pred, dtype=np.float32)

        # 2. Extract and SANITIZE the saved matrices
        n_chains = self.idata.posterior.sizes["chain"]
        n_draws = self.idata.posterior.sizes["draw"]
        n_total_draws = n_chains * n_draws


        def clean_int(arr_name):
            arr = self.idata.bart_trees[arr_name].values
            if arr.dtype == object: arr = np.stack(arr.flatten())
            # Force to float32 first to handle NaNs, then cast to int16
            arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=-1.0)
            return arr.astype(np.int16).reshape(n_total_draws, -1)
        
        def clean_float(arr_name):
            arr = self.idata.bart_trees[arr_name].values
            if arr.dtype == object: arr = np.stack(arr.flatten())
            return np.asarray(arr, dtype=np.float32).reshape(n_total_draws, -1)

        F = clean_int("feature")
        L = clean_int("left")
        R = clean_int("right")
        T = clean_float("threshold")
        V = clean_float("value")
        offset = float(self.idata.bart_metadata["offset"].values)

        # Dynamically calculate your trees so you never have to hardcode it!
        # MAX_NODES = 255  # This must match whatever MAX_NODES is in your live patch
        MAX_NODES = self.model_config.get("num_nodes", 255)
        n_nodes = F.shape[1]
        n_trees = n_nodes // MAX_NODES

        # add in the onehot
        is_one_hot = np.zeros(X_np.shape[1], dtype=np.int32)
        for i, rule in enumerate(self.model_config.get("split_rules")):
            if "OneHot" in rule:
                is_one_hot[i]=1


        # 3. Execute the C-compiled engine
        mu_preds_flat = fast_bart_predict(X_np, F, T, L, R, V, offset, n_trees, MAX_NODES, is_one_hot)


        # 4. Reshape back to exactly match PyMC's (chain, draw, p_obs) structure
        n_chains = self.idata.posterior.sizes["chain"]
        n_draws = self.idata.posterior.sizes["draw"]
        num_obs = X_np.shape[0]
        
        mu_preds_3d = mu_preds_flat.reshape((n_chains, n_draws, num_obs))

        # 5. Apply Probit Link
        mu_prob = sp.stats.norm.cdf(mu_preds_3d)
        
        # 6. Pack into xarray
        if coords is None:
            coords = np.arange(num_obs)
            
        post_pred = xr.Dataset(
            {"mu": (("chain", "draw", "p_obs"), mu_prob)},
            coords={
                "chain": self.idata.posterior.chain,
                "draw": self.idata.posterior.draw,
                "p_obs": coords
            }
        )
        
        if extend_idata:
            self.idata["posterior_predictive"] = post_pred
            
        return post_pred

    def sample_posterior_predictive(
        self, 
        X_pred:np.ndarray, 
        coords:np.ndarray, 
        extend_idata:bool=False, 
        **kwargs
    )->xr.Dataset :
        """Derives posterior predictions on updated datasets.

        Args:
            X_pred (np.ndarray): Covariate matrix in long-time format.
            coords (np.ndarray): Coordinates associated with long-time format.
            extend_idata (bool, optional): Adds results to the existing idata object. Defaults to False.
        
        Returns:
            xr.Dataset: Dataset containing the predicted outputs from the model and covariate matrix.
        """ 
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted in this instance. Either refit or predict with bart_predict")
        with self.model:  # sample with new input data
            pm.set_data({"x_data":X_pred} , coords={"p_obs":coords})
            post_pred = pm.sample_posterior_predictive(self.idata, var_names=["mu"],**kwargs)
            if extend_idata:
                self.idata.extend(post_pred)
        posterior_predictive_samples = az.extract(
            post_pred, "posterior_predictive", combined=True
        )

        return posterior_predictive_samples.transpose()
    
    # def bart_predict(
    #     self, 
    #     X_pred:np.ndarray, 
    #     coords:np.ndarray, 
    #     size:int = None, 
    #     rng:RandomState=None, 
    #     **kwargs
    # )->xr.DataArray:
    #     """Derives posterior predictions on updated dataset. Alternative method for re-loaded models.

    #     Args:
    #         X_pred (np.ndarray): Covariate matrix in long-time format.
    #         coords (np.ndarray): Coordinates associated with long-time format.
    #         size (int, optional): Sets sample of posterior draws. Defaults to None.
    #         rng (RandomState, optional): Random number generator for reproducable results. Defaults to None.

    #     Returns:
    #         xr.DataArray: DataArray containing the predicted outputs from the model and covariate matrix.
    #     """
    #     if not self.all_trees:
    #         raise RuntimeError("No tree structure loaded. Load a tree or train the model")
    #     trees = self.all_trees
    #     if size is None:
    #         size = len(self.all_trees)
    #     if rng is None:
    #         rng = np.random.default_rng(1)
    #     pt_x = pt.constant(X_pred)
    #     post_pred_mu_ = _sample_posterior(trees, pt_x, rng, size = size, shape=1)
    #     post_pred = pm.math.invprobit(post_pred_mu_ + self.offset).eval()
    #     post_pred = post_pred.reshape(post_pred.shape[0], post_pred.shape[1])
    #     # print(post_pred.shape)
    #     post_pred = xr.DataArray(post_pred, name = "mu", 
    #                                 coords = {"sample":np.arange(0,post_pred.shape[0]),
    #                                         "p_obs": coords
    #                                 })
    #     return post_pred

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
    def get_default_model_config() -> Dict:
        raise RuntimeError("NO DEFAULT MODEL CONFIGS, MUST SPECIFY")

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
        # self.X = X
        # self.y = y
        # Force strict C-contiguous memory alignment and 32-bit precision
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.float32)

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
    y_time:np.ndarray,
    y_status:np.ndarray,
    x:np.ndarray, 
    weight:Optional[np.ndarray] = None,
    time_scale = None 
)-> Dict:
    """Generates long-time formatted event status and covariate matrix.

    The SurvBartModel operates using a discrete time format. This means each observation is represented by a series of observations for each time point up to the event time. 

    Args:
        y_time (np.ndarray): Event time
        y_status (nd.array): Event status indicator (1 if event occurs, 0 if censored)
        x (np.ndarray): Covariate matrix.
        weight (Optional[np.ndarray], optional): Weights associated with each observation. If non provided, then each observation will have weights of 1. Defaults to None.
        time_scale (Optional[int]): time scaleing factor
    Returns:
        Dict: Dictionary containg all of the training data including an event status array, covariate matrix, weights and coordinates associated with the long-time format.
    """
    if weight is None:
        weight = np.ones(y_time.shape[0])

    if time_scale is not None:
        yt = np.ceil(y_time/time_scale).flatten()
    else:
        yt = y_time.flatten()
    # t_sk = y_sk["Survival_in_days"]

    d_sk = np.array(y_status, dtype="int").flatten()
    t_uniq, t_inv = np.unique(yt, return_inverse=True)
    t_wide = np.tile(t_uniq, (y_time.shape[0],1))
    # print(t_wide.shape)
    # print(t_wide)
    d_wide = np.tile(np.zeros(t_uniq.shape[0]), (y_time.shape[0],1))
    # print(d_wide.shape)
    max_t = yt.max()
    # print(max_t)


    # replace the time greater than t_sk. LOOP is over columns, so it should be fast with the exception of unscalled, many days
    for i in np.arange(t_wide.shape[1]):
        # print(yt)
        # print(t_wide[t_wide[:,i]>yt,i])
        t_wide[t_wide[:,i] > yt,i]=max_t + 1
        d_wide[t_wide[:,i] == yt,i] = d_sk[t_wide[:,i]==yt]
        d_wide[t_wide[:,i] > yt, i] = 2
    
    # time
    t_wide = t_wide.reshape(t_wide.shape[0]*t_wide.shape[1],1)
    t_wide = (t_wide[t_wide != max_t + 1]) # drop extra times
    t_wide = t_wide.reshape(t_wide.shape[0],1)

    d_wide = d_wide.reshape(d_wide.shape[0]*d_wide.shape[1],1)
    trn_y = (d_wide[d_wide != 2])
    trn_y = trn_y.reshape(trn_y.shape[0],1)
    
    # x and delta expansion
    tx = np.repeat(x, t_inv+1, axis=0)
    trn_x = np.hstack([t_wide, tx])

    # weight expansion
    weight_long = np.repeat(weight, t_inv+1)
    weight_long = weight_long.reshape(weight_long.shape[0],1)

    # get coords
    coords = np.repeat(np.arange(0,x.shape[0]), t_inv+1)
    return {
            "y":trn_y, 
            "x":trn_x, 
            "w":weight_long, 
            "coord":coords
        }


def get_posterior_test(
    y_time: np.ndarray,
    y_status: np.ndarray,
    x:np.ndarray,
    time_scale = None
)->Dict:
    """Generates long-time format for posterior distribution testing.
    
    To analyze the posterior distribution, posterior predictive estimates need to be generated. Similar to the training data, the data for posterior predictions must also be in a long-time format.

    Args:
        y_time (np.ndarray): Event time
        y_status (nd.array): Event status indicator (1 if event occurs, 0 if censored)
        x_test (np.ndarray): Covariate matrix.
        time_scale (np.ndarray): Time Scaling factor
    Returns:
        Dict: Covariate matrix in long-time format and associated coordinates for the long-time format.
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

    
def get_pdp(
    x:np.ndarray, 
    var_col:List[int] = [], 
    values:List[Union[int,float]] = [], 
    qt:List[float] =[0.25,0.5,0.75], 
    sample_n=None
)->Union[np.ndarray, Dict]:
    """Generates data for Partial Dependency Plots.

    Args:
        x (np.ndarray): Covariate matrix.    
        var_col (List[int], optional): Covariate column to generate pdp values. Defaults to [].
        values (List[Union[int,float]], optional): Values to test on for each covariate. Defaults to [].
        qt (List[float], optional): Quantiles to generate values on if non are given. Defaults to [0.25,0.5,0.75].
        sample_n (_type_, optional): Sample size of large dataset. Useful for working with large datasets. Defaults to None.

    Returns:
        Union[np.ndarray, Dict]: PDP covariate matrix and dictionary for tracking PDP components.
    """
    if len(var_col) > 2:
        print("only upto 2 variables can be used at a time")
        pass

    if sample_n is not None:
        rn_idx = np.random.choice(np.arange(0, x.shape[0]), sample_n, replace=False)   
        x = x[rn_idx,:]     

    var_vals = []
    for idx, var in enumerate(var_col):
        if values[idx] is None:
            val =np.quantile(x[:,var], qt)
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
    post: Union[xr.DataArray, xr.Dataset]
)->Dict:
    """Generates the Survival Probability estimates for time points.

    Args:
        post (Union[xr.DataArray, xr.Dataset]): Posterior output.

    Returns:
        Dict: Risk Probability (Hazard) and Survival Probability for each draw, observation and time.
    """
    n,k = np.unique(post.p_obs.values, return_counts=True)
    if "Dataset" in str(post):
        prob = post.mu.values.reshape(-1, n.shape[0], k[0])
    else:
        prob = post.values.reshape(-1, n.shape[0], k[0])
    sv = np.cumprod(1-prob, 2)
    return {
        "prob":prob, 
        "sv":sv
        }

def get_sv_mean_quant(
    sv: np.ndarray, 
    msk: np.ndarray,
    draws: bool = True, 
    qntile: List[float] = [0.025, 0.975]
)->Dict:
    """Generates mean and quantile estimates of Survival Probabilties.

    Args:
        sv (np.ndarray): Survival probabilties.
        msk (np.ndarray): Mask for selecting values to average (class of a covariate).
        draws (bool, optional): If true, will average over the draws of the posterior, as well as masked values. Defaults to True.
        qntile (List[float], optional): Quantiles to average over. Defaults to [0.025, 0.975].

    Returns:
        Dict: Mask True mean, Mask True quantiles, Mask False mean, Mask False quantiles.
    """
    # binary mask means and quantiles
    #tmask
    sv_mt = sv[:,msk,:]
    sv_mt_m = sv_mt.mean(axis=1) # mean per draw
    if draws:
        sv_mt_q = np.quantile(sv_mt_m, qntile, axis = 0)
        sv_mt_m = sv_mt_m.mean(axis=0) # mean over draws

    #fmask
    sv_mf = sv[:,~msk,:]
    sv_mf_m = sv_mf.mean(axis=1)
    if draws:
        sv_mf_q = np.quantile(sv_mf_m, qntile, axis = 0)
        sv_mf_m = sv_mf_m.mean(axis=0)

    return {
        "mt_m":sv_mt_m, 
        "mt_q":sv_mt_q, 
        "mf_m":sv_mf_m, 
        "mf_q":sv_mf_q
    }

# get diff metric 
def pdp_diff_metric(
    pdp_val: Dict, 
    idx: np.ndarray, 
    qntile: List[float] = [0.025, 0.975]
)->Dict:
    """Generate estimate of marginal difference from PDP posterior.

    Args:
        pdp_val (Dict): Posterior of PDP predictions.
        idx (np.ndarray): Index of PDP sets.
        qntile (List[float], optional): Quantile values for Credible Interval. Defaults to [0.025, 0.975].

    Returns:
        Dict: Survival Probability Difference Mean, Survival Probability Difference Quantiles.
    """
    diff = (pdp_val["sv"][:,:idx,:] - pdp_val["sv"][:,idx:,:]).mean(1) 
    d_m = diff.mean(0)
    d_q = np.quantile(diff, qntile, axis=0)
    return {"diff_m": np.round(d_m,3),
             "diff_q": np.round(d_q,3)}

def pdp_rr_metric(
    pdp_val: Dict, 
    idx:np.ndarray, 
    qntile:list = [0.025, 0.975]
)->Dict:
    """Generates a Risk Ratio (Hazard Ratio) from pdp values.

    Args:
        pdp_val (Dict): Posterior of PDP predictions.
        idx (np.ndarray): Index of PDP sets.
        qntile (list, optional): Quantile valeus for Credible Interval. Defaults to [0.025, 0.975].

    Returns:
        Dict: Risk Ratio Mean, Risk Ratio Quantiles.
    """
    r = (pdp_val["prob"][:,idx:,:] / pdp_val["prob"][:,:idx,:]).mean(1) 
    r_m = r.mean(0)
    r_q = np.quantile(r, qntile, axis=0)
    return {"rr_m": np.round(r_m,3), 
            "rr_q":np.round(r_q,3)}
    
