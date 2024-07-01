
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
        sampler_config = (self.get_default_sampler_config() if sampler_config is None else sampler_config)
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
        weights: np.ndarray,
        coords: np.ndarray,
        predictor_names: list,
        **kwargs
    ): 
        """Builds the PYMC base model.

        Args:
            X (np.ndarray): Covariate matrix in long-time form.
            y (np.ndarray): Event status in long-time form.
            weights (np.ndarray): Array of weights.
            coords (np.ndarray): Array of coordinates associated with long-time form.   
            predictor_names (list): List of names for variables.    

        Returns:
            _type_: BartSurvModel
        """
        # check x,y,weights        
        self._generate_and_preprocess_model_data(X, y, weights, predictor_names)
        # Get model Configs
        SPLIT_RULES = [eval(rule) for rule in self.model_config.get("split_rules", None)]
        SPLIT_PRIOR = self.model_config.get("split_prior")
        M = self.model_config.get("trees", 20)

        # custom logp
        def logp_bern(value, mu, w):
            return w * pm.logp(pm.Bernoulli.dist(mu), value)
        # extension of the bernoulli, the distribution is used as normal
        def dist_bern(mu, w, size):
            return pm.Bernoulli.dist(mu, size=size)
        # offset
        self.offset = sp.norm.ppf(np.mean(self.y))
        # model coords
        mcoords = {"xvars":self.predictor_names}

        with pm.Model(coords=mcoords) as self.model:    
            self.model.add_coord("p_obs", coords, mutable=True)
            x_data = pm.MutableData("x_data", self.X, dims=("p_obs", "xvars"))
            w = pm.MutableData("weights", self.weights)
            # change names of y_values
            f = pmb.BART("f", X=x_data, Y=self.y.flatten(), m=M, split_rules = SPLIT_RULES, split_prior = SPLIT_PRIOR)
            z = pm.Deterministic("z", (f + self.offset))
            mu = pm.Deterministic("mu", pm.math.invprobit(z), dims=("p_obs"))
            pm.CustomDist("y_pred", mu, w.flatten(), dist=dist_bern, logp=logp_bern, observed=self.y.flatten(), shape = x_data.shape[0])            
    
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
        self.all_trees = list(self.model.f.owner.op.all_trees)
        # get x names


        # create idata of testing data
        combined_data = np.hstack([self.y, self.weights, self.X])
        # assert all(combined_data.columns), "All columns must have non-empty names"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning#,
            )
            self.idata.add_groups(fit_data=xr.DataArray(combined_data))  # type: ignore
            self.idata.add_groups(predictor_names = xr.DataArray(self.predictor_names))
            self.idata.add_groups(offset = xr.DataArray(self.offset))
        return self.idata  # type: ignore

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
        # print(post_pred.shape)
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

def get_time_transform(
    t_event:np.ndarray, 
    time_scale:int = 10
)->np.ndarray:
    """Down scale event time data.

    The BartSurvModel is a resource heavy model. Because each observation is expanded into a long-time format, using non-scaled times will increase the training data by number of time points up-to each event times. To reduce the computational burden it is recommended to down-scale to the event times. 
    In example, if there are 90 days in the event-time and down-scaling of 30 was applied, it would return times 1,2,3 corresponding to days 30,60,90.

    Args:
        t_event (np.ndarray): Event times.
        time_scale (int, optional): Scale by which to reduce event times. Defaults to 10.

    Returns:
        np.ndarray: Down-scaled event times.
    """
    return np.ceil(t_event/time_scale)
    
def get_y_sklearn(
    status:np.ndarray, 
    t_event:np.ndarray
)->np.ndarray:
    """Reformats event status and event times to the sklearn-survival default format.

    Args:
        status (np.ndarray): Event status.
        t_event (np.ndarray): Event time.

    Returns:
        np.ndarray: Event status/time in sklearn format.
    """
    y = np.array(list(zip(np.array(status, dtype="bool"), t_event)), dtype=[("Status","?"),("Survival_in_days", "<f8")])
    return y

def get_surv_pre_train(
    y_sk:np.ndarray, 
    x_sk:np.ndarray, 
    weight:Optional[np.ndarray] = None
)-> Dict:
    """Generates long-time formatted event status and covariate matrix.

    The SurvBartModel operates using a discrete time format. This means each observation is represented by a series of observations for each time point up to the event time. 

    Args:
        y_sk (np.ndarray): Event time/status in y_sk format.
        x_sk (np.ndarray): Covariate matrix.
        weight (Optional[np.ndarray], optional): Weights associated with each observation. If non provided, then each observation will have weights of 1. Defaults to None.

    Returns:
        Dict: Dictionary containg all of the training data including an event status array, covariate matrix, weights and coordinates associated with the long-time format.
    """
    if weight is None:
        weight = np.ones(y_sk.shape[0])
    t_sk = y_sk["Survival_in_days"]
    d_sk = np.array(y_sk["Status"], dtype="int")
    t_uniq, t_inv = np.unique(t_sk, return_inverse=True)
    t_wide = np.tile(t_uniq, (y_sk.shape[0],1))
    d_wide = np.tile(np.zeros(t_uniq.shape[0]), (y_sk.shape[0],1))
    max_t = t_sk.max()

    # replace the time greater than t_sk. LOOP is over columns, so it should be fast with the exception of unscalled, many days
    for i in np.arange(t_wide.shape[1]):
        t_wide[t_wide[:,i] > t_sk,i]=max_t + 1
        d_wide[t_wide[:,i] == t_sk,i] = d_sk[t_wide[:,i]==t_sk]
        d_wide[t_wide[:,i] > t_sk, i] = 2
    
    # time
    t_wide = t_wide.reshape(t_wide.shape[0]*t_wide.shape[1],1)
    t_wide = (t_wide[t_wide != max_t + 1]) # drop extra times
    t_wide = t_wide.reshape(t_wide.shape[0],1)

    d_wide = d_wide.reshape(d_wide.shape[0]*d_wide.shape[1],1)
    trn_y = (d_wide[d_wide != 2])
    trn_y = trn_y.reshape(trn_y.shape[0],1)
    
    # x and delta expansion
    tx = np.repeat(x_sk, t_inv+1, axis=0)
    trn_x = np.hstack([t_wide, tx])

    # weight expansion
    weight_long = np.repeat(weight, t_inv+1)
    weight_long = weight_long.reshape(weight_long.shape[0],1)

    # get coords
    coords = np.repeat(np.arange(0,x_sk.shape[0]), t_inv+1)
    return {
            "y":trn_y, 
            "x":trn_x, 
            "w":weight_long, 
            "coord":coords
        }


def get_posterior_test(
    y_sk:np.ndarray, 
    x_test:np.ndarray
)->Dict:
    """Generates long-time format for posterior distribution testing.
    
    To analyze the posterior distribution, posterior predictive estimates need to be generated. Similar to the training data, the data for posterior predictions must also be in a long-time format.

    Args:
        y_sk (np.ndarray): Event time/status in y_sk format.
        x_test (np.ndarray): Covariate matrix.

    Returns:
        Dict: Covariate matrix in long-time format and associated coordinates for the long-time format.
    """
    uniq_times = np.unique(y_sk["Survival_in_days"])
    s0 = x_test.shape[0] # length
    s1 = x_test.shape[1] # width
    # create time range
    d1 = uniq_times
    d2 = np.tile(d1,s0).reshape(d1.shape[0]*s0,1)
    d3 = np.tile(x_test, d1.shape[0]).reshape(s0*d1.shape[0], s1)
    out = np.hstack([d2,d3])
    # coordinates
    coords = np.repeat(np.arange(0, x_test.shape[0]), d1.shape[0])
    return {"post_x": out, "coords":coords}

    
def get_pdp(
    x_sk:np.ndarray, 
    var_col:List[int] = [], 
    values:List[Union[int,float]] = [], 
    qt:List[float] =[0.25,0.5,0.75], 
    sample_n=None
)->Union[np.ndarray, Dict]:
    """Generates data for Partial Dependency Plots.

    Args:
        x_sk (np.ndarray): Covariate matrix.    
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
        rn_idx = np.random.choice(np.arange(0, x_sk.shape[0]), sample_n, replace=False)   
        x_sk = x_sk[rn_idx,:]     

    var_vals = []
    for idx, var in enumerate(var_col):
        if values[idx] is None:
            val =np.quantile(x_sk[:,var], qt)
            var_vals.append(val)
        else:
            var_vals.append(values[idx])
    s1 = len(var_col)
    if s1 > 1:
        cart = np.dstack(np.meshgrid(*var_vals)).reshape(-1, s1)
    else:
        cart = np.array(var_vals[0]).reshape(-1,1)
    s2 = cart.shape[0] # number of vals
    s3 = x_sk.shape[0] # number of obs
    out_sk = np.tile(x_sk, (s2,1)) # tile x_sk into blocks of number of vals
    l_cart = np.repeat(cart, s3, axis=0) # repeat vals number of obs times
    for idx, var in enumerate(var_col): # fill the columns of the x_sk*blocks with new values
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
    
