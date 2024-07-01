import pandas as pd
import numpy as np
import lifelines as ll
import sksurv as sks
import matplotlib.pyplot as plt
from bart_survival import surv_bart as sb


import lifelines as ll
from lifelines import KaplanMeierFitter
import subprocess
import threading as th
import multiprocessing as mp
import sys
sys.path.append("../src/")
import sim_adj as sm
import arviz as az
# import brt_adj as sb

# file not fully linked################################### paramsim
def get_py_bart_prm_1(x_mat, event_dict, model_dict, sampler_dict, tst):
    y_sk = sb.get_y_sklearn(status = event_dict["status"], t_event=event_dict["t_event"])
    trn = sb.get_surv_pre_train(y_sk = y_sk, x_sk = x_mat, weight=None)
    post_test = sb.get_posterior_test(y_sk = y_sk, x_test = x_mat)

    # return trn, post_test, small_post_x, small_coords
    BSM = sb.BartSurvModel(model_config=model_dict, sampler_config=sampler_dict)
    # fit with just the time column
    BSM.fit(y=trn["y"], X=trn["x"], weights=trn["w"], coords = trn["coord"], random_seed=99)
    post1 = BSM.sample_posterior_predictive(X_pred=post_test["post_x"], coords=post_test["coords"])
    sv_prob = sb.get_sv_prob(post1)
    uniq_t = BSM.uniq_times
    
    post_test_tst = sb.get_posterior_test(y_sk = y_sk, x_test = tst)
    post_tst = BSM.sample_posterior_predictive(X_pred=post_test_tst["post_x"], coords=post_test_tst["coords"])
    sv_prob_tst = sb.get_sv_prob(post_tst)
    # uniq_t = BSM.uniq_times

    # do test
    # post_test_test = sb.get_posterior_test(y_sk = y_sk, x_test = test)
    # post_test = BSM.sample_posterior_predictive(X_pred=post_test_test["post_x"], coords=post_test_test["coords"])
    # sv_prob_test = sb.get_sv_prob(post_test)
    # posttt = BSM.idata
    
    del BSM
    childs = mp.active_children()
    for child in childs:
        child.kill()
    return sv_prob, sv_prob_tst, uniq_t

def get_hdi_ci(pb_sv):
    s1 = pb_sv["sv"].shape[0]
    s2 = pb_sv["sv"].shape[1]
    s3 = pb_sv["sv"].shape[2]
    hdi_pre = az.convert_to_dataset(pb_sv["sv"].reshape(1,s1,s2,s3))
    #low-high
    hdi_lh = az.hdi(hdi_pre, hdi_prob=.95).x.values
    # low-high
    ci_lh = np.quantile(pb_sv["sv"], [0.025, 0.975], 0)
    return hdi_lh, ci_lh

def get_stats(true, pb_m, hdi_lh, ci_lh):
    # get psuedo-coverage
    cov_hdi = []
    cov_ci = []
    # pbm should be obs,times
    for i in range(pb_m.shape[1]):
        # rounded to the hundreth of a percent (no need to look smaller)
        t1 = (np.round(hdi_lh[:,i,0],4) <= np.round(true[:,i],4))  & (np.round(true[:,i],4) <= np.round(hdi_lh[:,i,1],4))
        t2 = (ci_lh[0,:,i] <= true[:,i])  & (true[:,i] <= ci_lh[1,:,i])
        cov_hdi.append(t1)
        cov_ci.append(t2)
    cov_hdi = (np.array(cov_hdi).T.sum(0))/pb_m.shape[0]
    cov_ci = (np.array(cov_ci).T.sum(0))/pb_m.shape[0]
    
    # mean_iv length
    hdi_iv = np.abs(hdi_lh[:,:,0] - hdi_lh[:,:,1]).mean(0)
    ci_iv = np.abs(ci_lh[0,:,:] - ci_lh[1,:,:]).mean(0)

    # get rmse and bias
    # rmse = msqr(true, pb_m)
    rmse = np.sqrt(np.mean(np.power(true-pb_m,2), axis = 0))
    bias = (true - pb_m).mean(0)
    return {"cov_hdi":cov_hdi.tolist(), "cov_ci":cov_ci.tolist(), "iv_hdi":hdi_iv.tolist(), "iv_ci":ci_iv.tolist(), "rmse":rmse.tolist(), "bias":bias.tolist()}
    

def sim_prm_1s(seed, n, scenario_, SPLIT_RULES, model_dict, sampler_dict):
    # set rng as seed given
    if type(seed) is int:
        rng = np.random.default_rng(seed)
    else:
        rng = seed
    
    #get mean survival
    good_sim = True
    blck = 0
    while good_sim:
        # generate survival simulation
        scenario, x_mat, event_dict, sv_true, sv_scale_true = get_sim(rng, n, **scenario_)
        scenario_tst, x_mat_tst, event_dict_tst, sv_true_tst, sv_scale_true_tst = get_sim(rng, n, **scenario_)
        
        # cens_perc = event_dict["status"][event_dict["status"] == 0].shape[0]/event_dict["status"].shape[0]
        
        # change the quantiles to be created from time
        # return(x_mat, event_dict, sv_true)
        # qnt_t = np.ceil(np.quantile(range(int(event_dict["t_event"].max())), [.05,0.1, .25, .5, .75,.9,.95])).astype("int")
        qnt_t = np.ceil(np.quantile(event_dict["t_event"], [.05,0.1, .25, .5, .75,.9,.95])).astype("int")
        # print(qnt_t)
        # return qnt_t
        event_dict2 = get_quant_events(qnt_t=qnt_t, event=event_dict)
        # return event_dict2
        # check that the qnt times and simulation are okay
        blck += 1
        if np.all(qnt_t == np.unique(event_dict2["t_event"])):
            print("good sim")            
            print(qnt_t)
            print(np.unique(event_dict2["t_event"]))
            good_sim = False
        else:
            print("redraw sim")
            print(qnt_t)
            print(np.unique(event_dict2["t_event"]))
        if blck == 10:
            print("tried 10 sims, couldn't get good sim, readjust")
            break
    
    # return x_mat, x_mat2
    # get the unique x_test
    # cens_perc2 = event_dict2["status"][event_dict2["status"] == 0].shape[0]/event_dict2["status"].shape[0]        
    
    # return x_mat, event_dict, event_dict2, sv_true, cens_perc, cens_perc2

    # uniq_t is the pre_adj event dict uniq times, used for verify, but not actual downstream
    uniq_t = np.unique(event_dict["t_event"]) 
    true_t = sv_true["true_times"]
    
    # fit bart_py
    pb_sv, pb_sv_tst, uniq_t = get_py_bart_prm_1(x_mat=x_mat, event_dict=event_dict2, model_dict=model_dict, sampler_dict=sampler_dict, tst = x_mat_tst)

    sv_true_t = sv_true["sv_true"][:, (uniq_t-1).astype("int")]
    sv_true_t_tst = sv_true_tst["sv_true"][:, (uniq_t-1).astype("int")]
    pb_sv_m = pb_sv["sv"].mean(0)
    pb_sv_m_tst = pb_sv_tst["sv"].mean(0)
    
    hdi, ci = get_hdi_ci(pb_sv)
    stats = get_stats(sv_true_t, pb_sv_m, hdi, ci)
    
    hdi_tst, ci_tst = get_hdi_ci(pb_sv_tst)
    stats_tst = get_stats(sv_true_t_tst, pb_sv_m_tst, hdi_tst, ci_tst)

    return sv_true, sv_true_tst, pb_sv, pb_sv_tst, uniq_t, stats, stats_tst, (hdi,ci), (hdi_tst, ci_tst)
    # fit bart_r
    # r_sv = get_r_bart3(event_dict2, x_mat)
    # do not return CI
    # return {"cens_perc":cens_perc, "cens_perc2":cens_perc2, "true_t":true_t, "uniq_t":uniq_t, "qnt_t":(qnt_t, qnt_t-1)}, sv_true, cph_sv, pb_sv_m, r_sv[0], sv_true_t, cph_sv_test, pb_sv_m_test
    # return (cens_perc, true_t, uniq_t, (qnt_t, qnt_t-1)),(sv_true_r0, sv_true_r1),(k_sv1, k_sv2), pb_sv, r_sv
