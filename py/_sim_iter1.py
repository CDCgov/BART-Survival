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
import importlib
import _functions1 as fn
import _conditions1 as cn
import _plot_fx as pltf
plt.ioff()

def iter_simulation_1s(iters, n, scenario, SPLIT_RULES, model_dict, sampler_dict):
    """_summary_

    Args:
        iters (_type_): _description_
        n (_type_): _description_
        scenario (_type_): _description_
        SPLIT_RULES (_type_): _description_
        model_dict (_type_): _description_
        sampler_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    meta_lst = []
    sv_true_lst0 = []
    k_sv_lst0 = []
    k_sv_ci_lst0 = []
    pb_sv_lst0 = []
    pb_sv_ci_lst0 = []
    r_sv_lst0 = []
    r_sv_ci_lst0 = []

    for i in range(n, n+iters):
        meta, sv_true, k_sv, pb_sv, r_sv = fn.sim_1s(seed=i, n=n, scenario=scenario, SPLIT_RULES=SPLIT_RULES, model_dict=model_dict, sampler_dict=sampler_dict)

        uniq_t = meta[3][0]
        uniq_idx = meta[3][1]

        true_t = meta[1]
        assert((true_t[uniq_t-1] == uniq_t).all)

        sv_t_0 = sv_true[uniq_t-1]
        k_sv_0 = k_sv[0][uniq_idx]
        k_sv_ci0 = k_sv[1][:,uniq_idx]
        p_sv_0 = pb_sv[0][0][uniq_idx]
        p_sv_ci0 = pb_sv[0][1][:,uniq_idx]
        r_sv_0 = r_sv[0][uniq_idx]
        r_sv_ci0 = r_sv[1][:,uniq_idx]

        meta_lst.append(meta)
        sv_true_lst0.append(sv_t_0)
        k_sv_lst0.append(k_sv_0)
        k_sv_ci_lst0.append(k_sv_ci0)
        pb_sv_lst0.append(p_sv_0)
        pb_sv_ci_lst0.append(p_sv_ci0)
        r_sv_lst0.append(r_sv_0)
        r_sv_ci_lst0.append(r_sv_ci0)
    
    fig = pltf.plots1(meta, sv_true, k_sv, pb_sv, r_sv)
    title = f"{scenario['type']}, n {n}"
    fig.suptitle(title)

    k,p,r = fn.get_metrics1(
        sv_true_lst0,
        k_sv_lst0,
        k_sv_ci_lst0,
        pb_sv_lst0,
        pb_sv_ci_lst0,
        r_sv_lst0,
        r_sv_ci_lst0
    )
    cens = np.array([m[0] for m in meta_lst]).mean()
    return meta_lst,cens, k,p,r, fig

def iter_simulation_2s(iters, n, scenario, SPLIT_RULES, model_dict, sampler_dict):
    """_summary_

    Args:
        iters (_type_): _description_
        n (_type_): _description_
        scenario (_type_): _description_
        SPLIT_RULES (_type_): _description_
        model_dict (_type_): _description_
        sampler_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    meta_lst = []
    sv_true_lst0 = []
    k_sv_lst0 = []
    k_sv_ci_lst0 = []
    pb_sv_lst0 = []
    pb_sv_ci_lst0 = []
    r_sv_lst0 = []
    r_sv_ci_lst0 = []

    sv_true_lst1 = []
    k_sv_lst1 = []
    k_sv_ci_lst1 = []
    pb_sv_lst1 = []
    pb_sv_ci_lst1 = []
    r_sv_lst1 = []
    r_sv_ci_lst1 = []

    for i in range(n, n+iters):
        meta, sv_true, k_sv, pb_sv, r_sv = fn.sim_2s(i, n, scenario, SPLIT_RULES, model_dict, sampler_dict)
        
        uniq_t = meta[3][0]
        uniq_idx = meta[3][1]

        true_t = meta[1]
        assert((true_t[uniq_t-1] == uniq_t).all)

        sv_t_0 = sv_true[0][uniq_t-1]
        sv_t_1 = sv_true[1][uniq_t-1]

        k_sv_0 = k_sv[0][0][uniq_idx]
        k_sv_1 = k_sv[1][0][uniq_idx]
        k_sv_ci0 = k_sv[0][1][:,uniq_idx]
        k_sv_ci1 = k_sv[1][1][:,uniq_idx]

        p_sv_0 = pb_sv[0][0][uniq_idx]
        p_sv_1 = pb_sv[1][0][uniq_idx]
        p_sv_ci0 = pb_sv[0][1][:,uniq_idx]
        p_sv_ci1 = pb_sv[1][1][:,uniq_idx]

        r_sv_0 = r_sv[0][0][uniq_idx]
        r_sv_1 = r_sv[1][0][uniq_idx]
        r_sv_ci0 = r_sv[0][1][:,uniq_idx]
        r_sv_ci1 = r_sv[1][1][:,uniq_idx]

        meta_lst.append(meta)
        sv_true_lst0.append(sv_t_0)
        k_sv_lst0.append(k_sv_0)
        k_sv_ci_lst0.append(k_sv_ci0)
        pb_sv_lst0.append(p_sv_0)
        pb_sv_ci_lst0.append(p_sv_ci0)
        r_sv_lst0.append(r_sv_0)
        r_sv_ci_lst0.append(r_sv_ci0)
        sv_true_lst1.append(sv_t_1)
        k_sv_lst1.append(k_sv_1)
        k_sv_ci_lst1.append(k_sv_ci1)
        pb_sv_lst1.append(p_sv_1)
        pb_sv_ci_lst1.append(p_sv_ci1)
        r_sv_lst1.append(r_sv_1)
        r_sv_ci_lst1.append(r_sv_ci1)

    fig = pltf.plots2(meta, sv_true, k_sv, pb_sv, r_sv)
    title = f"{scenario['type']} {n}"
    fig.suptitle(title)
    

    k, p, r = fn.get_metrics2(
        sv_true_lst0, sv_true_lst1,
        k_sv_lst0, k_sv_lst1,
        k_sv_ci_lst0, k_sv_ci_lst1,
        pb_sv_lst0, pb_sv_lst1,
        pb_sv_ci_lst0, pb_sv_ci_lst1,
        r_sv_lst0, r_sv_lst1,
        r_sv_ci_lst0, r_sv_ci_lst1
    )
    cens = np.array([m[0] for m in meta_lst]).mean()
    return meta_lst,cens, k,p,r, fig