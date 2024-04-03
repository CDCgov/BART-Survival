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
importlib.reload(pltf)
importlib.reload(fn)
importlib.reload(cn)

# complex regressions
def iter_simulation_complex1(iters, n, seed_addl, scenario, SPLIT_RULES, model_dict, sampler_dict, plot_all):
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
    cph_sv_lst0 = []
    pb_sv_lst0 = []
    r_sv_lst0 = []
    fig_l = []
        
    strt = n*seed_addl
    end = strt + iters
    for i in range(strt, end):
        print(f"ITERATION************** {i}")
        meta, sv_true, cph_sv, pb_sv, r_sv = fn.sim_3s(seed=i, n=n, scenario_=scenario, SPLIT_RULES=SPLIT_RULES, model_dict=model_dict, sampler_dict=sampler_dict)

        uniq_t = meta["qnt_t"][0]
        uniq_idx = meta["qnt_t"][1]

        true_t = meta["true_t"]
        assert((true_t[uniq_idx] == uniq_t).all)

        sv_t_0 = sv_true["sv_true"][:,uniq_idx] # this is obs
        cph_sv_0 = cph_sv # this is obs
        p_sv_0 = pb_sv # this is already the mean, this is obs
        r_sv_0 = r_sv # this is already the mean, this is obs
        # there is a unsafe error that can occur if the sim event times don't map 1:1 to the qnt_times
        # this occurs if like the 95% t is 2 and the minimum event time is 4.
        assert(sv_t_0.shape == cph_sv_0.shape == p_sv_0.shape == r_sv_0.shape)    

        meta_lst.append(meta)
        sv_true_lst0.append(sv_t_0)
        cph_sv_lst0.append(cph_sv_0)
        pb_sv_lst0.append(p_sv_0)
        r_sv_lst0.append(r_sv_0)

        if plot_all:
            fig = pltf.plots3(uniq_t, sv_t_0, cph_sv_0, p_sv_0, r_sv_0)
            title = f"{scenario['type']}, n {n}"
            fig.suptitle(title)
            fig_l.append(fig)
    
    if not plot_all:
        fig = pltf.plots3(meta, sv_t_0, cph_sv_0, p_sv_0, r_sv_0)
        title = f"{scenario['type']}, n {n}"
        fig.suptitle(title)
        fig_l.append(fig)

    c,p,r = fn.get_metrics3(
        sv_true_lst0,
        cph_sv_lst0,
        pb_sv_lst0,
        r_sv_lst0,
    )
    cens1 = np.array([m["cens_perc"] for m in meta_lst]).mean()
    cens2 = np.array([m["cens_perc2"] for m in meta_lst]).mean()
    return meta_lst, (strt, end), (cens1,cens2), c,p,r, fig_l

# def iter_simulation_2s(iters, n, seed_addl, scenario, SPLIT_RULES, model_dict, sampler_dict, plot_all=False):
#     """_summary_

#     Args:
#         iters (_type_): _description_
#         n (_type_): _description_
#         scenario (_type_): _description_
#         SPLIT_RULES (_type_): _description_
#         model_dict (_type_): _description_
#         sampler_dict (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     meta_lst = []
#     sv_true_lst0 = []
#     k_sv_lst0 = []
#     k_sv_ci_lst0 = []
#     pb_sv_lst0 = []
#     pb_sv_ci_lst0 = []
#     r_sv_lst0 = []
#     r_sv_ci_lst0 = []

#     sv_true_lst1 = []
#     k_sv_lst1 = []
#     k_sv_ci_lst1 = []
#     pb_sv_lst1 = []
#     pb_sv_ci_lst1 = []
#     r_sv_lst1 = []
#     r_sv_ci_lst1 = []
#     figs = []

#     strt = n*seed_addl
#     end = strt + iters
#     for i in range(strt, end):
#         print(f"ITERATION************** {i}")
#         meta, sv_true, k_sv, pb_sv, r_sv = fn.sim_2s(i, n, scenario, SPLIT_RULES, model_dict, sampler_dict)
        
#         uniq_t = meta[3][0]
#         uniq_idx = meta[3][1]

#         true_t = meta[1]
#         assert((true_t[uniq_t-1] == uniq_t).all)

#         sv_t_0 = sv_true[0][uniq_t-1]
#         sv_t_1 = sv_true[1][uniq_t-1]

#         k_sv_0 = k_sv[0][0]
#         k_sv_1 = k_sv[1][0]
#         k_sv_ci0 = k_sv[0][1]
#         k_sv_ci1 = k_sv[1][1]

#         print(sv_t_0)
        
#         p_sv_0 = pb_sv[0][0]
#         p_sv_1 = pb_sv[1][0]
#         p_sv_ci0 = pb_sv[0][1]
#         p_sv_ci1 = pb_sv[1][1]

#         r_sv_0 = r_sv[0][0]
#         r_sv_1 = r_sv[1][0]
#         r_sv_ci0 = r_sv[0][1]
#         r_sv_ci1 = r_sv[1][1]

#         # print(k_sv_0)
#         # print(p_sv_0)
#         # print(r_sv_0)

#         meta_lst.append(meta)
#         sv_true_lst0.append(sv_t_0)
#         k_sv_lst0.append(k_sv_0)
#         k_sv_ci_lst0.append(k_sv_ci0)
#         pb_sv_lst0.append(p_sv_0)
#         pb_sv_ci_lst0.append(p_sv_ci0)
#         r_sv_lst0.append(r_sv_0)
#         r_sv_ci_lst0.append(r_sv_ci0)

#         sv_true_lst1.append(sv_t_1)
#         k_sv_lst1.append(k_sv_1)
#         k_sv_ci_lst1.append(k_sv_ci1)
#         pb_sv_lst1.append(p_sv_1)
#         pb_sv_ci_lst1.append(p_sv_ci1)
#         r_sv_lst1.append(r_sv_1)
#         r_sv_ci_lst1.append(r_sv_ci1)
        
#         if plot_all:
#             fig = pltf.plots2(meta, sv_true, k_sv, pb_sv, r_sv)
#             title = f"{scenario['type']} {n}"
#             fig.suptitle(title)
#         figs.append(fig)

#     if plot_all==False:
#         fig = pltf.plots2(meta, sv_true, k_sv, pb_sv, r_sv)
#         title = f"{scenario['type']} {n}"
#         fig.suptitle(title)
    

#     k, p, r = fn.get_metrics2(
#         sv_true_lst0, sv_true_lst1,
#         k_sv_lst0, k_sv_lst1,
#         k_sv_ci_lst0, k_sv_ci_lst1,
#         pb_sv_lst0, pb_sv_lst1,
#         pb_sv_ci_lst0, pb_sv_ci_lst1,
#         r_sv_lst0, r_sv_lst1,
#         r_sv_ci_lst0, r_sv_ci_lst1
#     )
#     cens = np.array([m[0] for m in meta_lst]).mean()
    
#     if plot_all:
#         return meta_lst, (strt,end), cens, k,p,r, figs
#     return meta_lst, (strt,end), cens, k,p,r, fig

# def iter_simulation_3s(iters, n, seed_addl, scenario, SPLIT_RULES, model_dict, sampler_dict, plot_all=False):
#     """_summary_

#     Args:
#         iters (_type_): _description_
#         n (_type_): _description_
#         scenario (_type_): _description_
#         SPLIT_RULES (_type_): _description_
#         model_dict (_type_): _description_
#         sampler_dict (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     meta_lst = []
#     sv_true_lst0 = []
#     k_sv_lst0 = []
#     k_sv_ci_lst0 = []
#     pb_sv_lst0 = []
#     pb_sv_ci_lst0 = []
#     r_sv_lst0 = []
#     r_sv_ci_lst0 = []

#     sv_true_lst1 = []
#     k_sv_lst1 = []
#     k_sv_ci_lst1 = []
#     pb_sv_lst1 = []
#     pb_sv_ci_lst1 = []
#     r_sv_lst1 = []
#     r_sv_ci_lst1 = []
#     figs = []

#     strt = n*seed_addl
#     end = strt + iters
#     for i in range(strt, end):
#         print(f"ITERATION************** {i}")
#         meta, sv_true, k_sv, pb_sv, r_sv = fn.sim_2s(i, n, scenario, SPLIT_RULES, model_dict, sampler_dict)
        
#         # return meta, sv_true, k_sv, pb_sv, r_sv
#         uniq_t = meta[3][0]
#         uniq_idx = meta[3][1]

#         true_t = meta[1]
#         assert((true_t[uniq_t-1] == uniq_t).all)

#         sv_t_0 = sv_true[0][uniq_t-1]
#         sv_t_1 = sv_true[1][uniq_t-1]

#         k_sv_0 = k_sv[0][0]
#         k_sv_1 = k_sv[1][0]
#         k_sv_ci0 = k_sv[0][1]
#         k_sv_ci1 = k_sv[1][1]

#         print(sv_t_0)
        
#         p_sv_0 = pb_sv[0][0]
#         p_sv_1 = pb_sv[1][0]
#         p_sv_ci0 = pb_sv[0][1]
#         p_sv_ci1 = pb_sv[1][1]

#         r_sv_0 = r_sv[0][0]
#         r_sv_1 = r_sv[1][0]
#         r_sv_ci0 = r_sv[0][1]
#         r_sv_ci1 = r_sv[1][1]

#         # print(k_sv_0)
#         # print(p_sv_0)
#         # print(r_sv_0)

#         meta_lst.append(meta)
#         sv_true_lst0.append(sv_t_0)
#         k_sv_lst0.append(k_sv_0)
#         k_sv_ci_lst0.append(k_sv_ci0)
#         pb_sv_lst0.append(p_sv_0)
#         pb_sv_ci_lst0.append(p_sv_ci0)
#         r_sv_lst0.append(r_sv_0)
#         r_sv_ci_lst0.append(r_sv_ci0)

#         sv_true_lst1.append(sv_t_1)
#         k_sv_lst1.append(k_sv_1)
#         k_sv_ci_lst1.append(k_sv_ci1)
#         pb_sv_lst1.append(p_sv_1)
#         pb_sv_ci_lst1.append(p_sv_ci1)
#         r_sv_lst1.append(r_sv_1)
#         r_sv_ci_lst1.append(r_sv_ci1)
        
#         if plot_all:
#             fig = pltf.plots2(meta, sv_true, k_sv, pb_sv, r_sv)
#             title = f"{scenario['type']} {n}"
#             fig.suptitle(title)
#         figs.append(fig)

#     if plot_all==False:
#         fig = pltf.plots2(meta, sv_true, k_sv, pb_sv, r_sv)
#         title = f"{scenario['type']} {n}"
#         fig.suptitle(title)
    

#     k, p, r = fn.get_metrics2(
#         sv_true_lst0, sv_true_lst1,
#         k_sv_lst0, k_sv_lst1,
#         k_sv_ci_lst0, k_sv_ci_lst1,
#         pb_sv_lst0, pb_sv_lst1,
#         pb_sv_ci_lst0, pb_sv_ci_lst1,
#         r_sv_lst0, r_sv_lst1,
#         r_sv_ci_lst0, r_sv_ci_lst1
#     )
#     cens = np.array([m[0] for m in meta_lst]).mean()
    
#     if plot_all:
#         return meta_lst, (strt,end), cens, k,p,r, figs
#     return meta_lst, (strt,end), cens, k,p,r, fig