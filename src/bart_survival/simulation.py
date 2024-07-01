"""Generate a simulated survival datasets"""
import numpy as np
from scipy.stats import bernoulli

def get_x_matrix(
    N=100, 
    x_vars=1, 
    VAR_CLASS = None, 
    VAR_PROB = None, 
    rng = None
):
    """Generates a covariate matrix to use in simulating survival functions and components.
    If x_vars > length VAR_CLASS, then any additionally generated variables are drawn from a uniform distribution (0,1).

    Args:
        N (int, optional): Number of observations. Defaults to 100.
        x_vars (int, optional): Number of variabiables. Defaults to 1.
    VAR_CLASS (ndarray, optional): 1D array of integers defining range of numbers in each variable. VAR_CLASS of 2 will generate a binary variable. All other positive integers will draw from a uniform distribution with lower=0, upper=Integer. Defaults to None.
        VAR_PROB (ndarray, optional): 1D array defines the probability parameter of binary variables created by `VAR_CLASS`==2. All non-binary variables should have a value of `None`. Defaults to None.
        rng (_type_, optional): Random Number Generator from numpy. Used to generate multiple datasets. Defaults to None.

    Returns:
        ndarray: 2D array, where rows are observations and columns are variables.
    """    

    if rng is None:
        rng = np.random.default_rng(seed=99)
    
    bern = bernoulli
    bern.random_state = rng

    # create an x matrix
    x_mat = np.zeros((N, x_vars))
    for idx, x in enumerate(VAR_CLASS):
        if x == 2:
            # x1 = sp.bernoulli.rvs(VAR_PROB[idx], size = N)
            # x1 = bern.rvs(VAR_PROB[idx], size = N)
            x1 = rng.binomial(1, VAR_PROB[idx], size = N)
        else:
            x1 = rng.uniform(0, VAR_CLASS[idx], size = N)
            x1 = np.round(x1, 3)
        x_mat[:,idx] = x1
    if x_vars > len(VAR_CLASS):
        for idx in np.arange(len(VAR_CLASS), x_vars):
            x_mat[:,idx] = rng.uniform(0,1,size=N)
    return x_mat

def simulate_survival(
    x_mat = None,
    scale_f=None, 
    shape_f = None, 
    eos = None,
    cens_scale = None,
    time_scale = None,
    true_only = False,
    rng = None
):
    """Generates survival outcome data from a covariate data. 
    
    Using a covariate matrix and a linear equation that defines the linear combination of the covariates, the function will generate a Hazard Rate and Survival probability curve for each observation in the covariate matrix. 
    Additionaly, an event time for each observation.
    Randomized right censoring and End-of-Study censoring can also be incorporated into the simulated dataset.

    Survival times are generated from a Weibull Distribution with a user specified by the evaluation of the `shape_f` and `scale_f` parameters.
        Increasing the scale will increase the number of time points.
        Shape controls if the hazard rate is increasing or decreasing.
    
    `scale_f` must be parameterized as 'np.exp(...linear equation...)' this allows appropriate computation of hazard rates.
    Expected HR can be computed through `np.power(np.exp(Beta_x), shape)`.

    A few documents that demonstarte the weibull distribution and function in Survival Analysis can be found here:
        https://wilmarigl.de/wp-content/uploads/2018/01/tutorial_hr_parsurvmodels.pdf 
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda3668.htm
    

    Args:
        x_mat (ndarray, optional): 2D array of the covariate matrix. Defaults to None.
        scale_f (string, optional): Equation that defines the scale parameter as a combination the covariate matrix. Defaults to None.
        shape_f (string, optional): Equation that defines the shape parameter as a combination of the covariate matrix. Combination with the covariate matrix is not always needed and will provide a Survival Components that follow a proportional hazard. Defaults to None.
        eos (int, optional): End of Study time point. Allows truncation of time-points as full-stop right censoring for generated event times. Defaults to None.
        cens_scale (int, optional): Incorporates uninformative censoring into the generated event times. Defaults to None.
        time_scale (int, optional): Downscales event times and Survival component times to value. Defaults to None.
        true_only (bool, optional): If `True` returns only the Survival components and not the generated event times. Defaults to False.
        rng (_type_, optional): Random Number Generator to generate sequential survival datasets. Defaults to None.

    Returns:
        tuple: Tuble of dictionaries containing the various survival components. 
    """

    N = x_mat.shape[0]
    if rng is None:
        rng = np.random.default_rng(seed=99)

    # lambda and alpha
    scale = eval(scale_f)
    if "x_mat" not in shape_f:
        shape = np.repeat(eval(shape_f), N)
    else:
        shape = eval(shape_f)
    print(f"mean shape {shape.mean()}")
    print(f"mean scale {scale.mean()}")
    if not true_only:
        # generate event_times and status
        unif = rng.uniform(size=N)
        tlat = inverse_weibull_survival(scale=scale, shape=shape, u=unif)
        print(f"mean time draws {tlat.mean()}")
        # generate censoring for using an exponential distirubution
        cens_percentage = None
        if cens_scale is not None:            
            cens = np.ceil(rng.exponential(size = N, scale = cens_scale * scale.mean())).reshape(-1,1)
            print(f"cens mean {cens.mean()}")
            t_event  = np.minimum(cens, np.ceil(tlat))
            status = (tlat <= cens) * 1
            cens_percentage = status.sum()/N
            # print(status)
            # print(status.sum())
        else:
            t_event = np.ceil(tlat)
            status = np.ones(N).reshape(-1,1)
        
        # eos censoring
        if eos is not None:
            eos_msk = t_event > eos
            t_event[eos_msk] = eos
            status[eos_msk] = 0

        event_dict = {"t_event":t_event, "status":status, "cens_percentage":cens_percentage}
        # adjust time scale
        if time_scale is not None:
            t_event_scale = np.ceil(t_event/time_scale)
            event_dict = {"t_event":t_event, "t_event_scale": t_event_scale, "status":status, "cens_percentage":cens_percentage}
        # get event times
        t_max = int(t_event.max())
        t = np.arange(1,t_max+1, 1, dtype="int")
    else:
        if eos is None:
            t_max=100     # hack that defaults number of time points to 100
        # t = np.linspace(1,t_max+1, t_max+2, dtype="int")
        t = np.arange(1,t_max+1, 1, dtype="int")
    

    # Generate survival and hazard actual values
    

    hz_mat = weibull_hazard(scale, shape, t)
    sv_mat = weibull_survival(scale, shape, t)
    

    sv_dict = {"sv_true":sv_mat, "hz_true":hz_mat, "true_times":t}
    # down-scale
    sv_scale_dict = None
    if time_scale is not None:
        t_scale = np.arange(time_scale, t_max+1, time_scale, dtype="int")
        sv_mat_scale = sv_mat[:,t_scale-1]
        hz_mat_scale = hz_mat[:,t_scale-1]
        t_scale2 = np.ceil(t_scale/time_scale)
        sv_scale_dict = {"sv_true":sv_mat_scale, "hz_true":hz_mat_scale, "true_times":t_scale, "scale_times":t_scale2}    
    

    # Do not return x_mat anymore
    if true_only:
        return sv_dict, sv_scale_dict        
    return event_dict, sv_dict, sv_scale_dict

def check_inputs(scale, shape, u=np.array([.9]), t=np.array([1])):
    """Checks input types. If inputs are not `type` ndarray then converts to ndarray.

    Args:
        scale (ndarray): sets scale parameter for weibull Survival componenets
        shape (ndarray): sets shape parameter for weibull Survival components
        u (ndarray, optional): _description_. Defaults to np.array([.9]).
        t (ndarray, optional): _description_. Defaults to np.array([1]).

    Raises:
        Exception: Raises exception if (u<=0 or u>=1), scale<=0, shape<=0, t<=0

    Returns:
        tuple: tuple containing scale, shape, u, t
    """    
    ret = True
    if type(shape) != np.ndarray:
        shape = np.array([shape])
    if type(scale) != np.ndarray:
        scale = np.array([scale])
    if type(u) != np.ndarray:
        u = np.array([u])
    if type(t) != np.ndarray:
        t = np.array([t])
    errors = []
    if (any(u <=0) or any(u>=1)):
        ret=False
        errors.append("u must be between 0 and 1")
    if (any(scale <=0)):        
        ret=False
        errors.append("scale must be > 0")
    if (any(shape <=0)):
        ret=False
        errors.append("shape must be > 0")
    if (any(t <= 0)):
        ret=False
        errors.append("t must be > 0")
    if (ret != True):
        # print(errors)
        raise Exception(errors)
    
    return scale, shape, u, t

    
def inverse_weibull_survival(scale, shape, u):
    """Generates event times from an inverse weibull distribution.

    Args:
        scale (ndarray): Scale parameter.
        shape (ndarray): Shape parameter.
        u (ndarray): Draw from uniform probability distribution [0,1].

    Returns:
        ndarray: Event times given scale, shape, u parameters.
    """    
    scale, shape, u, t = check_inputs(scale,shape, u=u)
    event_times = scale.reshape(-1,1) * np.power(-np.log(u), 1/shape).reshape(-1,1)
    return event_times

def weibull_hazard(scale, shape, t):
    """Generates hazard rates at time from weibull distribution.

    Args:
        scale (ndarray): Scale parameter.
        shape (ndarray): Shape parameter.
        t (ndarray): Time points to generate hazard rates.

    Returns:
        ndarray: 2D ndarray that contains survival probabilites for `t.shape[0]` times points and `scale.shape[0]` observations.
    """    
    scale, shape, u, t = check_inputs(scale,shape, t=t)
    hz = 1/scale.reshape(-1,1) * shape.reshape(-1,1) * np.power(t/scale.reshape(-1,1), shape.reshape(-1,1)-1)
    return hz


def weibull_survival(scale, shape, t):
    """Generates survival probabilites from weibull distribution.

    Args:
        scale (ndarray): Scale parameter.
        shape (ndarray): Shape parameter.
        t (ndarray): Time points to generate survival probabilities.

    Returns:
        ndarray: 2D ndarray that contains survival probabilites for `t.shape[0]` times points and `scale.shape[0]` observations.
    """    
    scale, shape, u, t = check_inputs(scale,shape, t=t)
    prob = np.exp(-np.power(t/scale.reshape(-1,1), shape.reshape(-1,1)))
    return prob
