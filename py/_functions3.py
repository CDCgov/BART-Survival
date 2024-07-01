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
from _shared_funcs import save_to_csv, get_sim, get_quant_events

# this file isn't actually in use yet
# Complex 3
#############################################################
def get_py_bart_surv3_2(x_mat, event_dict, model_dict, sampler_dict, test):
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

    # do test
    post_test_test = sb.get_posterior_test(y_sk = y_sk, x_test = test)
    post_test = BSM.sample_posterior_predictive(X_pred=post_test_test["post_x"], coords=post_test_test["coords"])
    sv_prob_test = sb.get_sv_prob(post_test)

    
    del BSM
    childs = mp.active_children()
    for child in childs:
        child.kill()
    return sv_prob, uniq_t, sv_prob_test

def cox_ph_3_2(x_mat, event_dict, times, test):
    cph = ll.CoxPHFitter()
    q = np.hstack([event_dict["t_event"].reshape(-1,1),event_dict["status"].reshape(-1,1),x_mat]) 
    col = ["T", "E"] + [f"x_{i}" for i in range(x_mat.shape[1])] 
    q = pd.DataFrame(q, columns=col)

    cph.fit(q , "T", "E")
    cph_sv = cph.predict_survival_function(x_mat, times=times).to_numpy().T
    cph_sv_test = cph.predict_survival_function(test, times=times).to_numpy().T
    return cph_sv, cph_sv_test

def get_r_bart3_2(event_dict, x_mat):
    save_to_csv(event_dict, x_mat, file="exp3_tmp.csv")
    
    subprocess.call("/usr/local/bin/Rscript --vanilla ../R/bart_3.R", shell=True)
    # with open("../data/exp3_tmp_out3.csv","r") as f:
        # r_sv = pd.read_csv(f).values
    r_sv = pd.read_csv("../data/exp3_tmp_out3_mu.csv").values
    r_sv_l = pd.read_csv("../data/exp3_tmp_out3_cil.csv").values
    r_sv_h = pd.read_csv("../data/exp3_tmp_out3_cih.csv").values
    return r_sv, r_sv_l, r_sv_h

def sim_3_2s(seed, n, scenario_, SPLIT_RULES, model_dict, sampler_dict):
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
        cens_perc = event_dict["status"][event_dict["status"] == 0].shape[0]/event_dict["status"].shape[0]
        
        # change the quantiles to be created from time
        # print(range(int(event_dict["t_event"].max())))
        # range(event_dict["t_event"].max())
        qnt_t = np.ceil(np.quantile(range(int(event_dict["t_event"].max())), [.05, 0.1,.2, .3, .4, .5, .6,.7,.8, .9,1])).astype("int")
        event_dict2 = get_quant_events(qnt_t=qnt_t, event=event_dict)

        # get true mean
        # sv_true_mean = sv_true["sv_true"].mean(0)
        # get time quants
        # qnt_t = get_quant_times(sv_true_c=sv_true_mean, sv_true= sv_true, quant=[.95, .9, .75, .5, .25, .1, .01]).flatten()
        # reset simulated event times to quant times
        # event_dict2 = get_quant_events(qnt_t=qnt_t, event=event_dict)
        
        # check that the qnt times and simulation are okay
        blck += 1
        if np.all(qnt_t == np.unique(event_dict2["t_event"])):
            print("good sim")
            good_sim = False
        else:
            print("redraw sim")
            print(qnt_t)
            print(np.unique(event_dict2["t_event"]))
        if blck == 10:
            print("tried 10 sims, couldn't get good sim, readjust")
            break
    
    # assert False
    # get the test set, this will be a new set since rng is reused
    scenario_t, x_mat_t, event_dict_t, sv_true_t, sv_scale_true_t = get_sim(rng, n, **scenario_)

    # return x_mat, x_mat2
    # get the unique x_test
    cens_perc2 = event_dict2["status"][event_dict2["status"] == 0].shape[0]/event_dict2["status"].shape[0]        
    
    # return x_mat, event_dict, event_dict2, sv_true, cens_perc, cens_perc2

    # uniq_t is the pre_adj event dict uniq times, used for verify, but not actual downstream
    uniq_t = np.unique(event_dict["t_event"]) 
    true_t = sv_true["true_times"]
    
    # fit cox  
    cph_sv, cph_sv_test = cox_ph_3_2(x_mat, event_dict, times=qnt_t, test = x_mat_t)
    # cph_sv2 = cox_ph(x_mat, event_dict2, times=qnt_t)
    # fit bart_py
    pb_sv, uniq_t, pb_sv_test = get_py_bart_surv3_2(x_mat=x_mat, event_dict=event_dict2, model_dict=model_dict, sampler_dict=sampler_dict, test = x_mat_t)
    pb_sv_m = pb_sv["sv"].mean(0)
    pb_sv_m_test = pb_sv_test["sv"].mean(0)


    # fit bart_r
    r_sv = get_r_bart3_2(event_dict2, x_mat)
    # do not return CI
    return {"cens_perc":cens_perc, "cens_perc2":cens_perc2, "true_t":true_t, "uniq_t":uniq_t, "qnt_t":(qnt_t, qnt_t-1)}, sv_true, cph_sv, pb_sv_m, r_sv[0], sv_true_t, cph_sv_test, pb_sv_m_test
    # return (cens_perc, true_t, uniq_t, (qnt_t, qnt_t-1)),(sv_true_r0, sv_true_r1),(k_sv1, k_sv2), pb_sv, r_sv

