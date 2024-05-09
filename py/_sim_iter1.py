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

def iter_simulation_1s(iters, n, seed_addl, scenario, SPLIT_RULES, model_dict, sampler_dict):
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
    # meta_lst = []
    cens_lst = []
    sv_true_lst0 = []
    k_sv_lst0 = []
    k_sv_ci_lst0 = []
    pb_sv_lst0 = []
    pb_sv_ci_lst0 = []
    pb_sv_hdi_lst0 = []
    r_sv_lst0 = []
    r_sv_ci_lst0 = []

    
    strt = n*seed_addl
    end = strt + iters
    for i in range(strt, end):
        print(f"ITERATION************** {i}")
        # meta, sv_true, k_sv, pb_sv, r_sv = fn.sim_1s(seed=i, n=n, scenario=scenario, SPLIT_RULES=SPLIT_RULES, model_dict=model_dict, sampler_dict=sampler_dict)
        odict = fn.sim_1s(seed=i, n=n, scenario=scenario, SPLIT_RULES=SPLIT_RULES, model_dict=model_dict, sampler_dict=sampler_dict)

        # uniq_t = meta[3][0]
        # uniq_idx = meta[3][1]

        # true_t = meta[1]
        # assert((true_t[uniq_t-1] == uniq_t).all)

        sv_t_0 = odict["sv_true_u"]
        k_sv_0 = odict["k_sv"][0]
        k_sv_ci0 = odict["k_sv"][1]
        p_sv_0 = odict["pb_sv_m"]
        p_sv_ci0 = odict["pb_ci"]
        p_sv_hdi0 = odict["pb_hdi"]
        r_sv_0 = odict["r_sv"][0]
        r_sv_ci0 = odict["r_sv"][1]

        # sv_t_0 = sv_true[uniq_t-1]
        # k_sv_0 = k_sv[0]
        # k_sv_ci0 = k_sv[1]
        # p_sv_0 = pb_sv[0][0]
        # p_sv_ci0 = pb_sv[0][1]
        # r_sv_0 = r_sv[0]
        # r_sv_ci0 = r_sv[1]

        # meta_lst.append(meta)
        cens_lst.append(odict["cens_perc"])
        sv_true_lst0.append(sv_t_0)
        k_sv_lst0.append(k_sv_0)
        k_sv_ci_lst0.append(k_sv_ci0)
        pb_sv_lst0.append(p_sv_0)
        pb_sv_hdi_lst0.append(p_sv_hdi0)
        pb_sv_ci_lst0.append(p_sv_ci0)
        r_sv_lst0.append(r_sv_0)
        r_sv_ci_lst0.append(r_sv_ci0)
    
        
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.step(range(5), sv_t_0, where="mid", color="black", label="true")
    ax.step(range(5), r_sv_ci0[1,:], color="blue", alpha=0.3, linestyle="dashed", where="mid", label="r_ci")
    ax.step(range(5), r_sv_ci0[0,:], color="blue", alpha=0.3, linestyle="dashed", where="mid")
    ax.step(range(5), p_sv_hdi0[1,:], color="red", alpha=0.3,linestyle="dashed", where="mid", label="p_hdi")
    ax.step(range(5), p_sv_hdi0[0,:], color="red", alpha=0.3,linestyle="dashed", where="mid")
    ax.step(range(5), p_sv_ci0[1,:], color="green", alpha=0.3,linestyle="dashed", where="mid", label = "p_ci")
    ax.step(range(5), p_sv_ci0[0,:], color="green", alpha=0.3,linestyle="dashed", where="mid")
    ax.step(range(5), k_sv_ci0[1,:], color="yellow", alpha=0.3,linestyle="dashed", where="mid", label = "k_ci")
    ax.step(range(5), k_sv_ci0[0,:], color="yellow", alpha=0.3, linestyle="dashed", where="mid")
    ax.legend()

    # fig = pltf.plots1(meta, sv_true, k_sv, pb_sv, r_sv)
    title = f"{scenario['type']}, n {n}"
    fig.suptitle(title)

    k,p,r = fn.get_metrics1(
        sv_true_lst0,
        k_sv_lst0,
        k_sv_ci_lst0,
        pb_sv_lst0,
        pb_sv_hdi_lst0,
        pb_sv_ci_lst0,
        r_sv_lst0,
        r_sv_ci_lst0
    )
    # cens = np.array([m[0] for m in meta_lst]).mean()
    cens = np.array(cens_lst).mean()
    # return meta_lst, (strt, end), cens, k,p,r, fig
    return (strt, end), cens, k,p,r, fig

def iter_simulation_2s(iters, n, seed_addl, scenario, SPLIT_RULES, model_dict, sampler_dict, plot_all=False):
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
    # meta_lst = []
    cens_lst = []
    sv_true_lst0 = []
    k_sv_lst0 = []
    k_sv_ci_lst0 = []
    pb_sv_lst0 = []
    pb_sv_ci_lst0 = []
    pb_sv_hdi_lst0 = []
    r_sv_lst0 = []
    r_sv_ci_lst0 = []

    sv_true_lst1 = []
    k_sv_lst1 = []
    k_sv_ci_lst1 = []
    pb_sv_lst1 = []
    pb_sv_ci_lst1 = []
    pb_sv_hdi_lst1 = []
    r_sv_lst1 = []
    r_sv_ci_lst1 = []
    figs = []

    strt = n*seed_addl
    end = strt + iters
    for i in range(strt, end):
        print(f"ITERATION************** {i}")
        odict = fn.sim_2s(i, n, scenario, SPLIT_RULES, model_dict, sampler_dict)
        # {"cens_perc":cens_perc, 
        # "uniq_t":uniq_t, 
        # "sv_true_u":(sv_true_r0, sv_true_r1),
        # "pb_sv":pb_sv, 
        # "r_sv":r_sv, 
        # "k_sv":(k_sv1, k_sv2)
        # }

        sv_t_0 = odict["sv_true_u"][0]
        sv_t_1 = odict["sv_true_u"][1]

        k_sv_0 = odict["k_sv"][0][0]
        k_sv_1 = odict["k_sv"][1][0]
        k_sv_ci0 = odict["k_sv"][0][1]
        k_sv_ci1 = odict["k_sv"][1][1]
 
        p_sv_0 = odict["pb_sv"][0][0]
        p_sv_1 = odict["pb_sv"][1][0]
        p_sv_ci0 = odict["pb_sv"][0][1]
        p_sv_ci1 = odict["pb_sv"][1][1]
        p_sv_hdi0 = odict["pb_sv"][0][2]
        p_sv_hdi1 = odict["pb_sv"][1][2]

        r_sv_0 = odict["r_sv"][0][0]
        r_sv_1 = odict["r_sv"][1][0]
        r_sv_ci0 = odict["r_sv"][0][1]
        r_sv_ci1 = odict["r_sv"][1][1]



        # meta_lst.append(meta)
        cens_lst.append(odict["cens_perc"])
        sv_true_lst0.append(sv_t_0)
        k_sv_lst0.append(k_sv_0)
        k_sv_ci_lst0.append(k_sv_ci0)
        pb_sv_lst0.append(p_sv_0)
        pb_sv_ci_lst0.append(p_sv_ci0)
        pb_sv_hdi_lst0.append(p_sv_hdi0)
        r_sv_lst0.append(r_sv_0)
        r_sv_ci_lst0.append(r_sv_ci0)

        sv_true_lst1.append(sv_t_1)
        k_sv_lst1.append(k_sv_1)
        k_sv_ci_lst1.append(k_sv_ci1)
        pb_sv_lst1.append(p_sv_1)
        pb_sv_ci_lst1.append(p_sv_ci1)
        pb_sv_hdi_lst1.append(p_sv_hdi1)
        r_sv_lst1.append(r_sv_1)
        r_sv_ci_lst1.append(r_sv_ci1)


    # if plot_all==False:
        # fig = pltf.plots2(meta, sv_true, k_sv, pb_sv, r_sv)
    fig, ax = plt.subplots(1,2, figsize=(10,10))
    ax[0].step(odict["uniq_t"], odict["sv_true_u"][0], color= "black", where="mid", label = "true")
    ax[1].step(odict["uniq_t"], odict["sv_true_u"][1], color = "black", where= "mid")
    ax[0].step(odict["uniq_t"], odict["k_sv"][0][1][0,:], color = "yellow", where="mid", alpha=0.5, linestyle="dashed", label="k_ci")
    ax[0].step(odict["uniq_t"], odict["k_sv"][0][1][1,:], color = "yellow", where="mid", alpha=0.5, linestyle="dashed")
    ax[1].step(odict["uniq_t"], odict["k_sv"][1][1][0,:], color = "yellow", where="mid", alpha=0.5, linestyle="dashed")
    ax[1].step(odict["uniq_t"], odict["k_sv"][1][1][1,:], color = "yellow", where="mid", alpha=0.5, linestyle="dashed")
    ax[0].step(odict["uniq_t"], odict["r_sv"][0][1][0,:], color = "blue", where="mid", alpha=0.5, linestyle="dashed", label = "r_ci")
    ax[0].step(odict["uniq_t"], odict["r_sv"][0][1][1,:], color = "blue", where="mid", alpha=0.5, linestyle="dashed")
    ax[1].step(odict["uniq_t"], odict["r_sv"][1][1][0,:], color = "blue", where="mid", alpha=0.5, linestyle="dashed")
    ax[1].step(odict["uniq_t"], odict["r_sv"][1][1][1,:], color = "blue", where="mid", alpha=0.5, linestyle="dashed")
    ax[0].step(odict["uniq_t"], odict["pb_sv"][0][1][0,:], color = "red", where="mid", alpha=0.5, linestyle="dashed", label="p_ci")
    ax[0].step(odict["uniq_t"], odict["pb_sv"][0][1][1,:], color = "red", where="mid", alpha=0.5, linestyle="dashed")
    ax[1].step(odict["uniq_t"], odict["pb_sv"][1][1][0,:], color = "red", where="mid", alpha=0.5, linestyle="dashed")
    ax[1].step(odict["uniq_t"], odict["pb_sv"][1][1][1,:], color = "red", where="mid", alpha=0.5, linestyle="dashed")
    ax[0].step(odict["uniq_t"], odict["pb_sv"][0][2][0,:], color = "green", where="mid", alpha=0.5, linestyle="dashed", label="r_ci")
    ax[0].step(odict["uniq_t"], odict["pb_sv"][0][2][1,:], color = "green", where="mid", alpha=0.5, linestyle="dashed")
    ax[1].step(odict["uniq_t"], odict["pb_sv"][1][2][0,:], color = "green", where="mid", alpha=0.5, linestyle="dashed")
    ax[1].step(odict["uniq_t"], odict["pb_sv"][1][2][1,:], color = "green", where="mid", alpha=0.5, linestyle="dashed")
    ax[0].legend()
    title = f"{scenario['type']} {n}"
    fig.suptitle(title)
    figs.append(fig)
    

    k, p, r = fn.get_metrics2(
        sv_true_lst0, sv_true_lst1,
        k_sv_lst0, k_sv_lst1,
        k_sv_ci_lst0, k_sv_ci_lst1,
        pb_sv_lst0, pb_sv_lst1,
        pb_sv_ci_lst0, pb_sv_ci_lst1,
        pb_sv_hdi_lst0, pb_sv_hdi_lst1,
        r_sv_lst0, r_sv_lst1,
        r_sv_ci_lst0, r_sv_ci_lst1
    )
    # cens = np.array([m[0] for m in meta_lst]).mean()
    cens = np.array(cens_lst).mean(0)
    
    # if plot_all:
        # return meta_lst, (strt,end), cens, k,p,r, figs
    return (strt,end), cens, k,p,r, figs
    # return meta_lst, (strt,end), cens, k,p,r, fig

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