import pandas as pd
import numpy as np
import lifelines as ll
import sksurv as sks
import matplotlib.pyplot as plt
from bart_survival import surv_bart as sb
from bart_survival import simulation as sm
import lifelines as ll
from lifelines import KaplanMeierFitter
import subprocess
import threading as th
import multiprocessing as mp

def get_sim(rng, N, type, x_vars, VAR_CLASS, VAR_PROB, scale_f, shape_f, cens_scale):
    """Generates simulation dataest

    Args:
        rng (_type_): if int, generate new rng seeded as int value.
        N (_type_): _description_
        type (_type_): _description_
        x_vars (_type_): _description_
        VAR_CLASS (_type_): _description_
        VAR_PROB (_type_): _description_
        scale_f (_type_): _description_
        shape_f (_type_): _description_
        cens_scale (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_mat = sm.get_x_matrix(
        N = N,
        x_vars = x_vars, 
        VAR_CLASS=VAR_CLASS,
        VAR_PROB= VAR_PROB
    )

    event_dict, sv_true, sv_scale_true = sm.simulate_survival(
        x_mat = x_mat,
        scale_f=scale_f,
        shape_f=shape_f,
        cens_scale=cens_scale,
        rng = rng
    )
    return type, x_mat, event_dict, sv_true, sv_scale_true


def ci_at_times(ci, k_t, uniq_t):
    """extends the CI to the k_sv values at specific times.
    - implemented in twopop_kpm

    Args:
        ci (_type_): _description_
        k_t (_type_): _description_
        uniq_t (_type_): _description_

    Returns:
        _type_: _description_
    """
    k_s = k_t.shape[0]
    uniq_idx =np.arange(uniq_t.shape[0])
    out = np.zeros(shape=(2,uniq_t.shape[0]))
    for i in range(k_s):
        if i == k_s-1:
            msk = uniq_idx[uniq_t >= k_t[i]]
        else:
            lwr = k_t[i]
            upr = k_t[i+1]
            msk = uniq_idx[(lwr<=uniq_t) & (uniq_t < upr)]
        out[:,msk] = ci[:,i].reshape(-1,1)
    return out

def twopop_kpm(event_dict, uniq_t):
    """Kaplain meier fitter. 
    Returns the k_sv and k_sv_ci for the given uniq_times

    Args:
        event_dict (_type_): _description_
        uniq_t (_type_): _description_

    Returns:
        _type_: _description_
    """
    kpm = ll.KaplanMeierFitter()
    kpm.fit(durations=event_dict["t_event"], event_observed=event_dict["status"])
    k1_a = kpm.survival_function_at_times(uniq_t).to_numpy()
    k1_ci = kpm.confidence_interval_survival_function_.values[1:,:].T
    k_t = kpm.timeline[1:]
    ci_a = ci_at_times(k1_ci, k_t, uniq_t)
    return k1_a, ci_a, uniq_t
    # return k1, k1_ci, k_t, k1_a, ci_a, uniq_t

def ed_sub(event_dict, x_mat, x_val):
    """gets the event dict for the sub population.
    - Used in the two population tests

    Args:
        event_dict (_type_): _description_
        x_mat (_type_): _description_
        x_val (_type_): _description_

    Returns:
        _type_: _description_
    """
    msk = x_mat[:,0] == x_val
    ed_out = {
        "t_event":event_dict["t_event"][msk],
        "status":event_dict["status"][msk]
    }
    uniq_t = np.unique(ed_out["t_event"])
    return ed_out, uniq_t


def save_to_csv(event_dict, x_mat, file="exp1_tmp.csv"):
    """helper function that passes the simulated event dataset to a csv
    - Used for R-bart

    Args:
        event_dict (_type_): _description_
        x_mat (_type_): _description_
        file (str, optional): _description_. Defaults to "exp1_tmp.csv".
    """
    col = ["t","s"] + ["x"+str(i) for i in range(x_mat.shape[1])]
    df = pd.DataFrame(np.hstack(
        [
            event_dict["t_event"],
            event_dict["status"],
            x_mat
        ]
    ),columns= col)
    exp_name = file
    path = f"../data/{exp_name}"
    with open(path, 'w') as f:
        df.to_csv(f, index=False)
    # df.to_csv(path,index = False)


    
def get_py_bart_surv(x_mat, event_dict, model_dict, sampler_dict):
    y_sk = sb.get_y_sklearn(status = event_dict["status"], t_event=event_dict["t_event"])
    trn = sb.get_surv_pre_train(y_sk = y_sk, x_sk = x_mat, weight=None)
    post_test = sb.get_posterior_test(y_sk = y_sk, x_test = x_mat)
    small_post_x = np.unique(post_test["post_x"][:,0]).reshape(-1,1)
    small_coords = np.hstack([np.repeat(0, small_post_x.shape[0]), np.repeat(1, small_post_x.shape[0]), ])
    small_post_x = np.vstack([small_post_x, small_post_x])
    # assert False
    # return post_test, small_post_x, small_coords

    BSM = sb.BartSurvModel(model_config=model_dict, sampler_config=sampler_dict)

    # fit with just the time column
    BSM.fit(y=trn["y"], X=trn["x"][:,0].reshape(-1,1), weights=trn["w"], coords = trn["coord"], random_seed=99)
    
    post1 = BSM.sample_posterior_predictive(X_pred=small_post_x, coords=small_coords)
    sv_prob = sb.get_sv_prob(post1)
    sv_m = sv_prob["sv"].mean(1).mean(0)
    uniq_t = BSM.uniq_times
    sv_q = np.quantile(sv_prob["sv"].mean(1), [0.025,0.975], axis=0)
    del BSM
    
    ###
    childs = mp.active_children()
    for child in childs:
        child.kill()
        # print(f"CHILD: {child}")

    return (sv_m, sv_q), uniq_t

def get_r_bart1(event_dict, x_mat):
    save_to_csv(event_dict, x_mat)
    subprocess.call("/usr/bin/Rscript --vanilla ../R/bart_1.R", shell=True)
    with open("../data/exp1_tmp_out2.csv", "r") as f:
        r_sv = pd.read_csv(f).values
    # r_sv = pd.read_csv("../data/exp1_tmp_out2.csv").values
    r_sv_m = r_sv[:,0].T
    r_sv_q = r_sv[:,1:].T
    return r_sv_m, r_sv_q


def get_quant_times(uniq_t, uniq=True):
    """gets the index of quantile values

    Args:
        uniq_t (array): Unique values of observed times
        uniq (bool, optional): Indicator returns just the unique quantiles if values repeat. Defaults to True.

    Returns:
        nd.array: Array of indexes of the quantiles.
    """
    qnt_t = np.quantile(uniq_t, [.1, .25, .5, .75, .9], method = "closest_observation")
    qnt_idx = np.array([(np.abs(uniq_t-i).argmin()) for i in qnt_t])
    if uniq:
        qnt_t = np.unique(qnt_t)
        qnt_idx=np.unique(qnt_idx)


    return qnt_t.astype("int"), qnt_idx.astype("int")


def sim_1s(seed, n, scenario, SPLIT_RULES, model_dict, sampler_dict):
    # set rng as seed given
    if type(seed) is int:
        rng = np.random.default_rng(seed)
    else:
        rng = seed
    # generate survival simulation
    scenario, x_mat, event_dict, sv_true, sv_scale_true = get_sim(rng, n, **scenario)
    # type, x_mat, event_dict, sv_true, sv_scale_true = get_sim(rng, N[0], **simple_1_2)
    cens_perc = event_dict["status"][event_dict["status"] == 0].shape[0]/event_dict["status"].shape[0]
    # get uniq times and quantiles as indexes
    uniq_t = np.unique(event_dict["t_event"]) 
    qnt_t, qnt_idx = get_quant_times(uniq_t, uniq = False)
    true_t = sv_true["true_times"]

    # singular sv true
    sv_true_r0 = sv_true["sv_true"][0,:]
    
    # fit kpm
    ed0, uniq_t2 = ed_sub(event_dict, x_mat, 1)
    k_sv = twopop_kpm(ed0, uniq_t)

    # fit bart_py
    pb_sv = get_py_bart_surv(x_mat, event_dict, model_dict, sampler_dict)
    
    # fit bart_r
    r_sv = get_r_bart1(event_dict, x_mat)
    return (cens_perc, true_t, uniq_t, (qnt_t, qnt_idx)), sv_true_r0, k_sv, pb_sv, r_sv




############################################################################## sim 2
def pb_sb_sub(sv_prob, msk):
    sv_m = sv_prob["sv"][:,msk,:].mean(0)
    sv_q = np.quantile(sv_prob["sv"][:,msk,:], [0.025,0.975], axis=0)    
    return sv_m, sv_q

def get_py_bart_surv2(x_mat, event_dict, model_dict, sampler_dict):
    y_sk = sb.get_y_sklearn(status = event_dict["status"], t_event=event_dict["t_event"])
    trn = sb.get_surv_pre_train(y_sk = y_sk, x_sk = x_mat, weight=None)
    post_test = sb.get_posterior_test(y_sk = y_sk, x_test = x_mat)
    small_post_x = np.unique(post_test["post_x"][:,0]).reshape(-1,1)
    xs = np.hstack([np.repeat(0, small_post_x.shape[0]), np.repeat(1, small_post_x.shape[0])])
    small_coords = np.hstack([np.repeat(0, small_post_x.shape[0]), np.repeat(1, small_post_x.shape[0]), ])
    small_post_x = np.vstack([small_post_x, small_post_x])
    small_post_x = np.hstack([small_post_x, xs.reshape(-1,1)])
    # assert False

    # return trn, post_test, small_post_x, small_coords
    BSM = sb.BartSurvModel(model_config=model_dict, sampler_config=sampler_dict)
    # fit with just the time column
    BSM.fit(y=trn["y"], X=trn["x"], weights=trn["w"], coords = trn["coord"], random_seed=99)
    post1 = BSM.sample_posterior_predictive(X_pred=small_post_x, coords=small_coords)
    sv_prob = sb.get_sv_prob(post1)
    sv_1 = pb_sb_sub(sv_prob, 0)
    sv_2 = pb_sb_sub(sv_prob, 1)
    uniq_t = BSM.uniq_times
    
    del BSM
    childs = mp.active_children()
    for child in childs:
        child.kill()
    return sv_1, sv_2, uniq_t

def get_r_bart2(event_dict, x_mat):
    save_to_csv(event_dict, x_mat, file="exp2_tmp.csv")
    subprocess.call("/usr/bin/Rscript --vanilla ../R/bart_2.R", shell=True)
    with open("../data/exp2_tmp_out2.csv","r") as f:
        r_sv = pd.read_csv(f).values
    # r_sv = pd.read_csv("../data/exp2_tmp_out2.csv").values
    msk = r_sv[:,0]==0
    r_sv_m1 = r_sv[msk,1].T
    r_sv_m2 = r_sv[~msk,1].T
    r_sv_q1 = r_sv[msk,2:].T
    r_sv_q2 = r_sv[~msk,2:].T
    return (r_sv_m1, r_sv_q1), (r_sv_m2, r_sv_q2)
    
def sim_2s(seed, n, scenario, SPLIT_RULES, model_dict, sampler_dict):
    # set rng as seed given
    if type(seed) is int:
        rng = np.random.default_rng(seed)
    else:
        rng = seed
    # generate survival simulation
    scenario, x_mat, event_dict, sv_true, sv_scale_true = get_sim(rng, n, **scenario)
    # get the unique x_test
    x_tst, x_tst_idx = np.unique(x_mat, return_index=True)
    cens_perc = event_dict["status"][event_dict["status"] == 0].shape[0]/event_dict["status"].shape[0]
    # get uniq times and quantiles as indexes
    uniq_t = np.unique(event_dict["t_event"]) 
    qnt_t, qnt_idx = get_quant_times(uniq_t, uniq = False)
    true_t = sv_true["true_times"]
    
    # singular sv true
    sv_true_r0 = sv_true["sv_true"][int(x_tst_idx[0]),:]
    sv_true_r1 = sv_true["sv_true"][int(x_tst_idx[1]),:]
    
    # fit kpm    
    ed1, uniq_t1 = ed_sub(event_dict, x_mat, 0)
    ed2, uniq_t2 = ed_sub(event_dict, x_mat, 1)
    k_sv1 = twopop_kpm(ed1, uniq_t)
    k_sv2 = twopop_kpm(ed2, uniq_t)
    
    # return k_sv1, k_sv2

    # fit bart_py
    pb_sv = get_py_bart_surv2(x_mat=x_mat, event_dict=event_dict, model_dict=model_dict, sampler_dict=sampler_dict)

    # fit bart_r
    r_sv = get_r_bart2(event_dict, x_mat)
    return (cens_perc, true_t, uniq_t, (qnt_t, qnt_idx)),(sv_true_r0, sv_true_r1),(k_sv1, k_sv2), pb_sv, r_sv

################################## Metrics
def check_array(v):
    if type(v) == list:
        v = np.array(v)
    return v

def rmse(true, est):
    true = check_array(true)
    est = check_array(est)
    out = np.sqrt(np.power((true - est),2).mean(0))
    return out

def bias(true, est):
    true = check_array(true)
    est = check_array(est)
    out = (true-est).mean(0)
    return out

def coverage(true, ci_est, calc = True):
    true = check_array(true)
    ci_est = check_array(ci_est)

    l = ci_est[:,0,:]
    u = ci_est[:,1,:]
    out = []
    for i in range(true.shape[1]):
        z = (l[:,i] <= true[:,i]) & (true[:,i] <= u[:,i])
        out.append(z)
    out = np.array(out).T
    if calc:
        return out.sum(0)/out.shape[0]
    else:
        return out

def iv_length(ci_est):
    ci_est = check_array(ci_est)
    l = ci_est[:,0,:]
    u = ci_est[:,1,:]
    out = u-l
    return out.mean(0)

def apply_metrics1(true0, est0, est_ci0):
    rmse0 = rmse(true0, est0)
    bias0 = bias(true0, est0)

    cov0 = coverage(true0, est_ci0, calc=True)
    ivl0 = iv_length(est_ci0)
    return {"rmse":rmse0, "bias":bias0, "cov":cov0, "ivl":ivl0}

def get_metrics1(
    sv_true_lst0, 
    k_sv_lst0,
    k_sv_ci_lst0, 
    pb_sv_lst0,
    pb_sv_ci_lst0, 
    r_sv_lst0, 
    r_sv_ci_lst0,
):
    k = apply_metrics1(
        sv_true_lst0, 
        k_sv_lst0, 
        k_sv_ci_lst0
    )
    p = apply_metrics1(
        sv_true_lst0, 
        pb_sv_lst0,
        pb_sv_ci_lst0
    )
    r = apply_metrics1(
        sv_true_lst0, 
        r_sv_lst0, 
        r_sv_ci_lst0
    )
    return k, p, r

def apply_metrics2(true0, true1, est0, est1, est_ci0, est_ci1):
    rmse0 = rmse(true0, est0)
    rmse1 = rmse(true1, est1)
    rmse_out = np.vstack([rmse0, rmse1]).mean(0)

    bias0 = bias(true0, est0)
    bias1 = bias(true1,est1)
    bias_out = np.vstack([bias0, bias1]).mean(0)


    cov0 = coverage(true0, est_ci0, calc=True)
    cov1 = coverage(true1, est_ci1, calc=True)
    cov_out = np.vstack([cov0, cov1]).mean(0)
    

    ivl0 = iv_length(est_ci0)
    ivl1 = iv_length(est_ci1)
    ivl_out = np.vstack([ivl0, ivl1]).mean(0)
    return {"rmse":rmse_out, "bias":bias_out, "cov":cov_out, "ivl":ivl_out}

def get_metrics2(
    sv_true_lst0, sv_true_lst1,
    k_sv_lst0, k_sv_lst1,
    k_sv_ci_lst0, k_sv_ci_lst1,
    pb_sv_lst0, pb_sv_lst1,
    pb_sv_ci_lst0, pb_sv_ci_lst1,
    r_sv_lst0, r_sv_lst1,
    r_sv_ci_lst0, r_sv_ci_lst1
):
    k = apply_metrics2(
        sv_true_lst0, sv_true_lst1,
        k_sv_lst0, k_sv_lst1,
        k_sv_ci_lst0, k_sv_ci_lst1
    )
    p = apply_metrics2(
        sv_true_lst0, sv_true_lst1,
        pb_sv_lst0, pb_sv_lst1,
        pb_sv_ci_lst0, pb_sv_ci_lst1
    )
    r = apply_metrics2(
        sv_true_lst0, sv_true_lst1,
        r_sv_lst0, r_sv_lst1,
        r_sv_ci_lst0, r_sv_ci_lst1
    )
    return k, p, r



