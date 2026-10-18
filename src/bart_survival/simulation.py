"""
bart_survival.simulation
========================
Simulation engines and synthetic benchmark generators for survival analysis.

Provides:
- simulate_survival: Parametric Weibull survival time and hazard curve simulator.
- get_x_matrix: Covariate matrix generator with specified discrete and continuous distributions.
- simulate_confounded_survival_data: High-level realistic synthetic cohort generator
  incorporating confounding, treatment effects, structural support violations (e.g. Inpatient/LOS),
  and uninformative right censoring without using any real clinical data.
"""

from typing import Optional, Dict, Tuple, List, Union, Any
import numpy as np
import pandas as pd
from scipy.stats import bernoulli


def get_x_matrix(
    N: int = 100, 
    x_vars: int = 1, 
    VAR_CLASS: Optional[List[int]] = None, 
    VAR_PROB: Optional[List[Optional[float]]] = None, 
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Generates a covariate matrix to use in simulating survival functions and components.

    If x_vars > length VAR_CLASS, then any additionally generated variables are drawn from a uniform distribution (0,1).

    Args:
        N (int, optional): Number of observations. Defaults to 100.
        x_vars (int, optional): Number of variables. Defaults to 1.
        VAR_CLASS (List[int], optional): 1D array of integers defining range of numbers in each variable.
            VAR_CLASS of 2 will generate a binary variable. All other positive integers will draw from a
            uniform distribution with lower=0, upper=Integer. Defaults to None.
        VAR_PROB (List[Optional[float]], optional): Defines the probability parameter of binary variables
            created by VAR_CLASS==2. All non-binary variables should have a value of None. Defaults to None.
        rng (np.random.Generator, optional): Random Number Generator from numpy. Defaults to None.

    Returns:
        np.ndarray: 2D array of shape (N, x_vars), where rows are observations and columns are variables.
    """    
    if rng is None:
        rng = np.random.default_rng(seed=99)
    
    if VAR_CLASS is None:
        VAR_CLASS = [2]
    if VAR_PROB is None:
        VAR_PROB = [0.5]

    x_mat = np.zeros((N, x_vars))
    for idx, x in enumerate(VAR_CLASS):
        if x == 2:
            prob = VAR_PROB[idx] if idx < len(VAR_PROB) and VAR_PROB[idx] is not None else 0.5
            x1 = rng.binomial(1, prob, size=N)
        else:
            x1 = rng.uniform(0, x, size=N)
            x1 = np.round(x1, 3)
        x_mat[:, idx] = x1
    if x_vars > len(VAR_CLASS):
        for idx in np.arange(len(VAR_CLASS), x_vars):
            x_mat[:, idx] = rng.uniform(0, 1, size=N)
    return x_mat


def simulate_survival(
    x_mat: np.ndarray,
    scale_f: str, 
    shape_f: str = "1", 
    eos: Optional[float] = None,
    cens_scale: Optional[float] = None,
    time_scale: Optional[float] = None,
    true_only: bool = False,
    rng: Optional[np.random.Generator] = None
) -> Union[Tuple[Dict[str, Any], Optional[Dict[str, Any]]], Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]]:
    """Generates survival outcome data from a covariate matrix using a Weibull distribution.

    Args:
        x_mat (np.ndarray): 2D array of the covariate matrix.
        scale_f (str): Equation that defines the scale parameter as a combination of covariates (e.g. 'np.exp(2 + .4*x_mat[:,0])').
        shape_f (str, optional): Equation that defines the shape parameter. Defaults to "1" (constant exponential hazard).
        eos (float, optional): End of Study time point (administrative right-censoring truncation). Defaults to None.
        cens_scale (float, optional): Scale multiplier for random exponential censoring. Defaults to None.
        time_scale (float, optional): Downscaling factor for event and discrete time grid intervals. Defaults to None.
        true_only (bool, optional): If True, returns only ground-truth survival/hazard curves. Defaults to False.
        rng (np.random.Generator, optional): Random Number Generator for reproducible draws. Defaults to None.

    Returns:
        Union[Tuple[Dict, Optional[Dict]], Tuple[Dict, Dict, Optional[Dict]]]:
            If true_only: (sv_dict, sv_scale_dict)
            Otherwise: (event_dict, sv_dict, sv_scale_dict)
    """
    N = x_mat.shape[0]
    if rng is None:
        rng = np.random.default_rng(seed=99)

    scale = eval(scale_f)
    if "x_mat" not in str(shape_f):
        shape = np.repeat(eval(str(shape_f)), N)
    else:
        shape = eval(str(shape_f))

    if not true_only:
        unif = rng.uniform(size=N)
        tlat = inverse_weibull_survival(scale=scale, shape=shape, u=unif)
        cens_percentage = None
        if cens_scale is not None:            
            cens = np.ceil(rng.exponential(size=N, scale=cens_scale * scale.mean())).reshape(-1, 1)
            t_event = np.minimum(cens, np.ceil(tlat))
            status = (tlat <= cens) * 1
            cens_percentage = float(status.sum() / N)
        else:
            t_event = np.ceil(tlat)
            status = np.ones(N).reshape(-1, 1)
        
        if eos is not None:
            eos_msk = t_event > eos
            t_event[eos_msk] = eos
            status[eos_msk] = 0

        event_dict: Dict[str, Any] = {"t_event": t_event, "status": status, "cens_percentage": cens_percentage}
        if time_scale is not None:
            t_event_scale = np.ceil(t_event / time_scale)
            event_dict["t_event_scale"] = t_event_scale
        
        t_max = int(t_event.max())
        t = np.arange(1, t_max + 1, 1, dtype="int")
    else:
        t_max = int(eos) if eos is not None else 100
        t = np.arange(1, t_max + 1, 1, dtype="int")
    
    hz_mat = weibull_hazard(scale, shape, t)
    sv_mat = weibull_survival(scale, shape, t)
    
    sv_dict = {"sv_true": sv_mat, "hz_true": hz_mat, "true_times": t}
    sv_scale_dict = None
    if time_scale is not None:
        t_scale = np.arange(time_scale, t_max + 1, time_scale, dtype="int")
        sv_mat_scale = sv_mat[:, t_scale - 1]
        hz_mat_scale = hz_mat[:, t_scale - 1]
        t_scale2 = np.ceil(t_scale / time_scale)
        sv_scale_dict = {"sv_true": sv_mat_scale, "hz_true": hz_mat_scale, "true_times": t_scale, "scale_times": t_scale2}    

    if true_only:
        return sv_dict, sv_scale_dict        
    return event_dict, sv_dict, sv_scale_dict


def check_inputs(
    scale: Union[float, np.ndarray], 
    shape: Union[float, np.ndarray], 
    u: Union[float, np.ndarray] = np.array([0.9]), 
    t: Union[float, np.ndarray] = np.array([1])
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validates and coerces numerical parameters for Weibull functions.

    Args:
        scale (Union[float, np.ndarray]): Scale parameter.
        shape (Union[float, np.ndarray]): Shape parameter.
        u (Union[float, np.ndarray], optional): Uniform quantile draw in (0, 1). Defaults to 0.9.
        t (Union[float, np.ndarray], optional): Time points. Defaults to 1.

    Raises:
        ValueError: If any parameter violates domain boundaries (scale > 0, shape > 0, 0 < u < 1, t > 0).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Coerced (scale, shape, u, t) arrays.
    """    
    if not isinstance(shape, np.ndarray):
        shape = np.array([shape])
    if not isinstance(scale, np.ndarray):
        scale = np.array([scale])
    if not isinstance(u, np.ndarray):
        u = np.array([u])
    if not isinstance(t, np.ndarray):
        t = np.array([t])

    errors = []
    if np.any(u <= 0) or np.any(u >= 1):
        errors.append("u must be between 0 and 1")
    if np.any(scale <= 0):        
        errors.append("scale must be > 0")
    if np.any(shape <= 0):
        errors.append("shape must be > 0")
    if np.any(t <= 0):
        errors.append("t must be > 0")
    if errors:
        raise ValueError(str(errors))
    
    return scale, shape, u, t

    
def inverse_weibull_survival(
    scale: Union[float, np.ndarray], 
    shape: Union[float, np.ndarray], 
    u: Union[float, np.ndarray]
) -> np.ndarray:
    """Generates event times from an inverse Weibull distribution.

    Args:
        scale (Union[float, np.ndarray]): Scale parameter.
        shape (Union[float, np.ndarray]): Shape parameter.
        u (Union[float, np.ndarray]): Uniform random draw in (0, 1).

    Returns:
        np.ndarray: Simulated event times of shape (N, 1).
    """    
    scale, shape, u, _ = check_inputs(scale, shape, u=u)
    event_times = scale.reshape(-1, 1) * np.power(-np.log(u), 1.0 / shape).reshape(-1, 1)
    return event_times


def weibull_hazard(
    scale: Union[float, np.ndarray], 
    shape: Union[float, np.ndarray], 
    t: Union[float, np.ndarray]
) -> np.ndarray:
    """Calculates hazard rates at specified time points from a Weibull distribution.

    Args:
        scale (Union[float, np.ndarray]): Scale parameter.
        shape (Union[float, np.ndarray]): Shape parameter.
        t (Union[float, np.ndarray]): Time points to evaluate.

    Returns:
        np.ndarray: 2D array of shape (N, len(t)) containing hazard rates.
    """    
    scale, shape, _, t = check_inputs(scale, shape, t=t)
    hz = (1.0 / scale.reshape(-1, 1)) * shape.reshape(-1, 1) * np.power(t / scale.reshape(-1, 1), shape.reshape(-1, 1) - 1.0)
    return hz


def weibull_survival(
    scale: Union[float, np.ndarray], 
    shape: Union[float, np.ndarray], 
    t: Union[float, np.ndarray]
) -> np.ndarray:
    """Calculates survival probabilities at specified time points from a Weibull distribution.

    Args:
        scale (Union[float, np.ndarray]): Scale parameter.
        shape (Union[float, np.ndarray]): Shape parameter.
        t (Union[float, np.ndarray]): Time points to evaluate.

    Returns:
        np.ndarray: 2D array of shape (N, len(t)) containing survival probabilities.
    """    
    scale, shape, _, t = check_inputs(scale, shape, t=t)
    prob = np.exp(-np.power(t / scale.reshape(-1, 1), shape.reshape(-1, 1)))
    return prob


def simulate_confounded_survival_data(
    N: int = 500,
    treatment_hr: float = 2.0,
    censoring_rate: float = 0.25,
    include_support_violation: bool = True,
    n_noise_features: int = 5,
    seed: Optional[int] = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates a realistic synthetic survival dataset for benchmarking and testing.

    Models a clinical scenario with:
    - Baseline confounders: 'age' (continuous) and 'stage' (ordinal 1, 2, 3).
    - Primary exposure: 'inpatient' (binary 0/1) assigned via logistic propensity.
    - Structural positivity/support violator: 'los' (Length of Stay).
      If include_support_violation is True, P(los > 0 | inpatient = 0) = 0 deterministically,
      creating an off-support boundary that challenges naive PDP and requires Causal G-Computation.
    - Noise features: Uninformative continuous variables to benchmark DART feature selection.
    - Right-censoring: Exponential uninformative censoring calibrated to target censoring_rate.

    Strictly synthetic: Uses 100% mathematical formulations and contains NO patient or restricted clinical data.

    Args:
        N (int, optional): Cohort sample size. Defaults to 500.
        treatment_hr (float, optional): Ground-truth hazard ratio for primary exposure. Defaults to 2.0.
        censoring_rate (float, optional): Target proportion of right-censored subjects. Defaults to 0.25.
        include_support_violation (bool, optional): If True, induces structural zero LOS for outpatients. Defaults to True.
        n_noise_features (int, optional): Number of uninformative noise features for DART testing. Defaults to 5.
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - df: DataFrame containing ['time', 'status', 'inpatient', 'los', 'age', 'stage'] + noise columns.
            - meta: Metadata dictionary with true parameters, column categories, and true HR.
    """
    rng = np.random.default_rng(seed)

    # 1. Baseline confounders
    age = np.clip(rng.normal(loc=62.0, scale=10.0, size=N), 25.0, 90.0)
    stage = rng.choice([1, 2, 3], size=N, p=[0.4, 0.4, 0.2])
    biomarker = rng.normal(loc=0.0, scale=1.0, size=N)

    # 2. Exposure assignment (propensity score)
    logit_p = -0.4 + 0.03 * (age - 60.0) + 0.5 * (stage - 1) + 0.3 * biomarker
    prob_inpatient = 1.0 / (1.0 + np.exp(-logit_p))
    inpatient = rng.binomial(1, prob_inpatient)

    # 3. Dependent mediator / support violation feature (Length of Stay)
    los = np.zeros(N, dtype=np.float64)
    if include_support_violation:
        inpat_mask = (inpatient == 1)
        if inpat_mask.sum() > 0:
            los[inpat_mask] = np.round(rng.gamma(shape=2.5, scale=2.0, size=inpat_mask.sum()) + 1.0, 1)
    else:
        los = np.round(rng.gamma(shape=2.0, scale=2.0, size=N) + 1.0, 1)

    # 4. Uninformative noise features (for DART testing)
    noise_dict: Dict[str, np.ndarray] = {}
    for k in range(n_noise_features):
        noise_dict[f"noise_{k+1}"] = np.round(rng.uniform(0.0, 1.0, size=N), 4)

    # 5. True Survival Time Generation
    # Under proportional hazards: lambda_i = exp(beta0 - log(HR)*inpatient - beta_cov * X)
    beta_treat = np.log(treatment_hr)
    beta_age = 0.02
    beta_stage = 0.25
    beta_los = 0.04

    base_log_scale = 3.6
    log_scale = base_log_scale - beta_treat * inpatient - beta_age * (age - 60.0) - beta_stage * (stage - 1) - beta_los * np.maximum(0.0, los - 2.0)
    scale = np.exp(log_scale)
    shape = np.ones(N, dtype=np.float64)  # Exponential baseline hazard (shape=1)

    u_draws = rng.uniform(0.001, 0.999, size=N)
    t_latent = inverse_weibull_survival(scale=scale, shape=shape, u=u_draws).flatten()

    # 6. Right-censoring mechanism
    if censoring_rate > 0:
        cens_mult = (1.0 - censoring_rate) / censoring_rate
        c_times = rng.exponential(scale=scale.mean() * cens_mult, size=N)
        obs_time = np.ceil(np.minimum(t_latent, c_times))
        obs_time = np.maximum(1.0, obs_time)
        status = (t_latent <= c_times).astype(int)
    else:
        obs_time = np.ceil(t_latent)
        status = np.ones(N, dtype=int)

    # Construct DataFrame
    data: Dict[str, Any] = {
        "time": obs_time,
        "status": status,
        "inpatient": inpatient,
        "los": los,
        "age": np.round(age, 1),
        "stage": stage,
        "biomarker": np.round(biomarker, 3),
    }
    data.update(noise_dict)
    df = pd.DataFrame(data)

    feature_cols = ["inpatient", "los", "age", "stage", "biomarker"] + list(noise_dict.keys())
    meta: Dict[str, Any] = {
        "true_hazard_ratio": treatment_hr,
        "treatment_column": "inpatient",
        "support_violation_column": "los" if include_support_violation else None,
        "confounder_columns": ["age", "stage", "biomarker"],
        "noise_columns": list(noise_dict.keys()),
        "feature_columns": feature_cols,
        "censoring_proportion": float(1.0 - status.mean()),
        "n_samples": N,
    }

    return df, meta
