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



# from bart_survival import simulation as sm

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
        VAR_PROB= VAR_PROB,
        rng = rng
    )

    event_dict, sv_true, sv_scale_true = sm.simulate_survival(
        x_mat = x_mat,
        scale_f=scale_f,
        shape_f=shape_f,
        cens_scale=cens_scale,
        rng = rng
    )
    return type, x_mat, event_dict, sv_true, sv_scale_true

def get_quant_times(sv_true_c, sv_true, quant=[0.9,0.75,0.5,0.25,0.1]):
	qnt_t = []
	for i in quant:
		tmp = np.abs(sv_true_c - i)
		idx = tmp == tmp.min()
		qnt_t.append(sv_true["true_times"][idx])
	return np.array(qnt_t)

def get_quant_events(qnt_t, event):
	q = np.array(qnt_t)
	et_ = event["t_event"].copy()
	et_out = event["t_event"].copy()
	es_out = event["status"].copy()
	for i in range(q.shape[0]):
		if i == 0:
			msk = et_<=q[i]
			et_out[msk] = q[i]
			# print(np.unique(et_out))
		else:
			msk = (q[i-1] < et_) & (et_ <= q[i])
			et_out[msk] = q[i]
			# print(np.unique(et_out))
			if i == q.shape[0]-1:
				msk = et_ > q[i]
				et_out[msk] = q[i]
				es_out[msk] = 0

	return {"t_event":et_out, "status":es_out}


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
    # subprocess.call("/usr/bin/Rscript --vanilla ../R/bart_1.R", shell=True)
    subprocess.call("/usr/local/bin/Rscript --vanilla ../R/bart_1.R", shell=True)
    with open("../data/exp1_tmp_out2.csv", "r") as f:
        r_sv = pd.read_csv(f).values
    # r_sv = pd.read_csv("../data/exp1_tmp_out2.csv").values
    r_sv_m = r_sv[:,0].T
    r_sv_q = r_sv[:,1:].T
    return r_sv_m, r_sv_q



def sim_1s(seed, n, scenario, SPLIT_RULES, model_dict, sampler_dict):
    # set rng as seed given
    if type(seed) is int:
        rng = np.random.default_rng(seed)
    else:
        rng = seed
    # generate survival simulation
    scenario, x_mat, event_dict, sv_true, sv_scale_true = get_sim(rng, n, **scenario)
    
	#get mean survival
    sv_true_mean = sv_true["sv_true"].mean(0)
    qnt_t = get_quant_times(sv_true_c=sv_true_mean, sv_true= sv_true).flatten()
    print(qnt_t)
    event_dict = get_quant_events(qnt_t=qnt_t, event=event_dict)

    # type, x_mat, event_dict, sv_true, sv_scale_true = get_sim(rng, N[0], **simple_1_2)
    cens_perc = event_dict["status"][event_dict["status"] == 0].shape[0]/event_dict["status"].shape[0]
    # print(cens_perc)
    # return cens_perc
	# get uniq times and quantiles as indexes
    uniq_t = np.unique(event_dict["t_event"]) 
    # qnt_t, qnt_idx = get_quant_times(uniq_t, uniq = False)
    true_t = sv_true["true_times"]

    # singular sv true
    sv_true_r0 = sv_true["sv_true"][0,:]
    
    # fit kpm
    ed0, uniq_t2 = ed_sub(event_dict, x_mat, 1)
    k_sv = twopop_kpm(ed0, uniq_t)

    # return k_sv, x_mat, event_dict
    # fit bart_py
    pb_sv = get_py_bart_surv(x_mat, event_dict, model_dict, sampler_dict)
    
    # fit bart_r
    r_sv = get_r_bart1(event_dict, x_mat)
    # return (cens_perc, true_t, uniq_t, (qnt_t, qnt_idx)), sv_true_r0, k_sv, pb_sv, r_sv
    return (cens_perc, true_t, uniq_t, (qnt_t, qnt_t-1)), sv_true_r0, k_sv, pb_sv, r_sv




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
    # print(small_post_x)
    # print(event_dict)
    # print(trn["x"])
    # print(trn["y"])
    # quit()
    # print(xs)
    # print(small_post_x)
    # print(small_coords)
    # quit()
    # assert False

    # return trn, post_test, small_post_x, small_coords
    BSM = sb.BartSurvModel(model_config=model_dict, sampler_config=sampler_dict)
    # fit with just the time column
    BSM.fit(y=trn["y"], X=trn["x"], weights=trn["w"], coords = trn["coord"], random_seed=99)
    post1 = BSM.sample_posterior_predictive(X_pred=small_post_x, coords=small_coords)
    sv_prob = sb.get_sv_prob(post1)
    # print(sv_prob[""].shape)
    # print(sv_prob)
    # quit()
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
    # subprocess.call("/usr/bin/Rscript --vanilla ../R/bart_2.R", shell=True)
    subprocess.call("/usr/local/bin/Rscript --vanilla ../R/bart_2.R", shell=True)
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
    
    #get mean survival
    sv_true_mean = sv_true["sv_true"].mean(0)
    qnt_t = get_quant_times(sv_true_c=sv_true_mean, sv_true= sv_true).flatten()
    print(qnt_t)
    event_dict = get_quant_events(qnt_t=qnt_t, event=event_dict)

    
    # get the unique x_test
    x_tst, x_tst_idx = np.unique(x_mat, return_index=True)
    cens_perc = event_dict["status"][event_dict["status"] == 0].shape[0]/event_dict["status"].shape[0]
    
    # get uniq times and quantiles as indexes
    uniq_t = np.unique(event_dict["t_event"]) 
    # qnt_t, qnt_idx = get_quant_times(uniq_t, uniq = False)
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
    return (cens_perc, true_t, uniq_t, (qnt_t, qnt_t-1)),(sv_true_r0, sv_true_r1),(k_sv1, k_sv2), pb_sv, r_sv

# complex 1,2
#############################################################
def get_py_bart_surv3(x_mat, event_dict, model_dict, sampler_dict):
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
    
    del BSM
    childs = mp.active_children()
    for child in childs:
        child.kill()
    return sv_prob, uniq_t

def cox_ph(x_mat, event_dict, times):
    cph = ll.CoxPHFitter()
    q = np.hstack([event_dict["t_event"].reshape(-1,1),event_dict["status"].reshape(-1,1),x_mat]) 
    col = ["T", "E"] + [f"x_{i}" for i in range(x_mat.shape[1])] 
    q = pd.DataFrame(q, columns=col)

    cph.fit(q , "T", "E")
    cph_sv = cph.predict_survival_function(x_mat, times=times).to_numpy().T
    return cph_sv

def get_r_bart3(event_dict, x_mat):
    save_to_csv(event_dict, x_mat, file="exp3_tmp.csv")
    
    subprocess.call("/usr/local/bin/Rscript --vanilla ../R/bart_3.R", shell=True)
    # with open("../data/exp3_tmp_out3.csv","r") as f:
        # r_sv = pd.read_csv(f).values
    r_sv = pd.read_csv("../data/exp3_tmp_out3_mu.csv").values
    r_sv_l = pd.read_csv("../data/exp3_tmp_out3_cil.csv").values
    r_sv_h = pd.read_csv("../data/exp3_tmp_out3_cih.csv").values
    return r_sv, r_sv_l, r_sv_h

def sim_3s(seed, n, scenario_, SPLIT_RULES, model_dict, sampler_dict):
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
        
        # get true mean
        sv_true_mean = sv_true["sv_true"].mean(0)
        # get time quants
        qnt_t = get_quant_times(sv_true_c=sv_true_mean, sv_true= sv_true, quant=[.95, .9, .75, .5, .25, .1, .01]).flatten()
        # reset simulated event times to quant times
        event_dict2 = get_quant_events(qnt_t=qnt_t, event=event_dict)
        
        # check that the qnt times and simulation are okay
        blck += 1
        if np.all(qnt_t == np.unique(event_dict2["t_event"])):
            print("good sim")
            good_sim = False
        else:
            print("redraw sim")
        if blck == 10:
            print("tried 10 sims, couldn't get good sim, readjust")
            break
    
    # assert False

    # get the unique x_test
    cens_perc2 = event_dict2["status"][event_dict2["status"] == 0].shape[0]/event_dict2["status"].shape[0]        
    
    # get uniq times and quantiles as indexes
    # uniq times is the post adj event times, should be same as qnt times
    uniq_t = np.unique(event_dict["t_event"]) 
    true_t = sv_true["true_times"]
    
    # fit cox  
    cph_sv = cox_ph(x_mat, event_dict, times=qnt_t)
    # cph_sv2 = cox_ph(x_mat, event_dict2, times=qnt_t)
    # fit bart_py
    pb_sv = get_py_bart_surv3(x_mat=x_mat, event_dict=event_dict2, model_dict=model_dict, sampler_dict=sampler_dict)
    pb_sv_m = pb_sv[0]["sv"].mean(0)

    # fit bart_r
    r_sv = get_r_bart3(event_dict2, x_mat)
    # do not return CI
    return {"cens_perc":cens_perc, "cens_perc2":cens_perc2, "true_t":true_t, "uniq_t":uniq_t, "qnt_t":(qnt_t, qnt_t-1)}, sv_true, cph_sv, pb_sv_m, r_sv[0]
    # return (cens_perc, true_t, uniq_t, (qnt_t, qnt_t-1)),(sv_true_r0, sv_true_r1),(k_sv1, k_sv2), pb_sv, r_sv


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
    r_sv = get_r_bart3(event_dict2, x_mat)
    # do not return CI
    return {"cens_perc":cens_perc, "cens_perc2":cens_perc2, "true_t":true_t, "uniq_t":uniq_t, "qnt_t":(qnt_t, qnt_t-1)}, sv_true, cph_sv, pb_sv_m, r_sv[0], sv_true_t, cph_sv_test, pb_sv_m_test
    # return (cens_perc, true_t, uniq_t, (qnt_t, qnt_t-1)),(sv_true_r0, sv_true_r1),(k_sv1, k_sv2), pb_sv, r_sv


################################### paramsim
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


################################## Metrics
def rsqr(true, est):
	SSR = np.sum(np.power(true - est, 2))
	SST = np.sum(np.power(true - np.mean(true), 2))
	return 1-(SSR/SST)

def medabs(true, est):
	mab = np.median(np.abs(true-est))
	return mab

def msqr(true, est):
	msqr = np.sqrt(np.mean(np.power(true-est,2)))
	return msqr

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

def rmse3(true, est):
    true = check_array(true)
    est = check_array(est)
    out = np.sqrt(np.power((true - est),2).mean(1))
    return out

def bias3(true, est):
    true = check_array(true)
    est = check_array(est)
    out = (true-est).mean(1)
    return out

def apply_metrics3(true0, est0):
    rmse0 = rmse3(true0, est0)
    bias0 = bias3(true0, est0)
    return {"rmse":rmse0, "bias":bias0}


def get_metrics3(
    sv_true_lst0, 
    cph_sv_lst0,
    pb_sv_lst0, 
    r_sv_lst0, 
):
    c = apply_metrics3(
        sv_true_lst0, 
        cph_sv_lst0, 
    )
    p = apply_metrics3(
        sv_true_lst0, 
        pb_sv_lst0, 
    )
    r = apply_metrics3(
        sv_true_lst0, 
        r_sv_lst0, 
    )
    return c, p, r
