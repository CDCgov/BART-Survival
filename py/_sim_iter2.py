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
            if i < strt+10:
                ttl = scenario["type"]
                namesss = f"../figs/{ttl}_{i}.png"
                fig.savefig(namesss)
    
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
    cens1 = [m["cens_perc"] for m in meta_lst]
    cens2 = [m["cens_perc2"] for m in meta_lst]
    # cens1 = np.array([m["cens_perc"] for m in meta_lst]).mean()
    # cens2 = np.array([m["cens_perc2"] for m in meta_lst]).mean()
    
    return meta_lst, (strt, end), (cens1,cens2), c,p,r, fig_l


# this is for continuous complex
def iter_simulation_complex3(iters, n, seed_addl, scenario, SPLIT_RULES, model_dict, sampler_dict, plot_all):
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
        # new simulation 3_2 is for the complex regression
        # changes the decile selection
        meta, sv_true, cph_sv, pb_sv, r_sv, sv_true_tst, cph_sv_tst, pb_sv_tst = fn.sim_3_2s(seed=i, n=n, scenario_=scenario, SPLIT_RULES=SPLIT_RULES, model_dict=model_dict, sampler_dict=sampler_dict)

        uniq_t = meta["qnt_t"][0]
        uniq_idx = meta["qnt_t"][1]

        true_t = meta["true_t"]
        assert((true_t[uniq_idx] == uniq_t).all)

        sv_t_0 = sv_true["sv_true"][:,uniq_idx] # this is obs
        sv_t_tst_0 = sv_true_tst["sv_true"][:,uniq_idx] # this is obs
        cph_sv_0 = cph_sv # this is obs
        cph_sv_tst_0 = cph_sv_tst
        p_sv_0 = pb_sv # this is already the mean, this is obs
        p_sv_tst_0 = pb_sv_tst
        r_sv_0 = r_sv # this is already the mean, this is obs
        assert(sv_t_0.shape == cph_sv_0.shape == p_sv_0.shape == r_sv_0.shape)    

        # meta_lst.append(meta)
        # sv_true_lst0.append(sv_t_0)
        # cph_sv_lst0.append(cph_sv_0)
        # pb_sv_lst0.append(p_sv_0)
        # r_sv_lst0.append(r_sv_0)
        figs = pltf.plots3_3(
            sv_t_0, cph_sv_0, p_sv_0, r_sv_0, sv_t_tst_0, cph_sv_tst_0, p_sv_tst_0,
            qs=1, qe=-3
        )
        tttl = scenario["type"]
        namesss = f"../sv_data/{tttl}_{n}_{i}.png"
        figs.savefig(namesss)

        df = pd.DataFrame(
            np.hstack(
            [
                sv_t_0[:,1:-1].flatten().reshape(-1,1),
                cph_sv[:,1:-1].flatten().reshape(-1,1),
                pb_sv[:,1:-1].flatten().reshape(-1,1),
                r_sv[:,1:-1].flatten().reshape(-1,1),
                sv_t_tst_0[:,1:-1].flatten().reshape(-1,1),
                cph_sv_tst[:,1:-1].flatten().reshape(-1,1),
                pb_sv_tst[:,1:-1].flatten().reshape(-1,1)
            ]
            ),
            columns = ["sv_t", "cph","pb","r","sv_tst","cph_tst","pb_tst"]
        )
        tttl = scenario["type"]
        namesss = f"../sv_data/{tttl}_{n}_{i}.csv"
        df.to_csv(namesss)


        # get metrics
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
        
        qs,qe = 2,-4
        c_rsqr = rsqr(sv_t_0[:,qs:qe], cph_sv_0[:,qs:qe])
        p_rsqr = rsqr(sv_t_0[:,qs:qe], p_sv_0[:,qs:qe])
        r_rsqr = rsqr(sv_t_0[:,qs:qe], r_sv_0[:,qs:qe])
        c_tst_rsqr = rsqr(sv_t_tst_0[:,qs:qe], cph_sv_tst[:,qs:qe])
        p_tst_rsqr = rsqr(sv_t_tst_0[:,qs:qe], p_sv_tst_0[:,qs:qe])
        
        c_medabs = medabs(sv_t_0[:,qs:qe], cph_sv_0[:,qs:qe])
        p_medabs = medabs(sv_t_0[:,qs:qe], p_sv_0[:,qs:qe])
        r_medabs = medabs(sv_t_0[:,qs:qe], r_sv_0[:,qs:qe])
        c_tst_medabs = medabs(sv_t_tst_0[:,qs:qe], cph_sv_tst[:,qs:qe])
        p_tst_medabs = medabs(sv_t_tst_0[:,qs:qe], p_sv_tst_0[:,qs:qe])

        c_msqr = msqr(sv_t_0[:,qs:qe], cph_sv_0[:,qs:qe])
        p_msqr = msqr(sv_t_0[:,qs:qe], p_sv_0[:,qs:qe])
        r_msqr = msqr(sv_t_0[:,qs:qe], r_sv_0[:,qs:qe])
        c_tst_msqr = msqr(sv_t_tst_0[:,qs:qe], cph_sv_tst[:,qs:qe])
        p_tst_msqr = msqr(sv_t_tst_0[:,qs:qe], p_sv_tst_0[:,qs:qe])

        c = {"q":(qs,qe), "r2":c_rsqr.tolist(), "mabs":c_medabs.tolist(), "rmse":c_msqr.tolist()}
        p = {"q":(qs,qe), "r2":p_rsqr.tolist(), "mabs":p_medabs.tolist(), "rmse":p_msqr.tolist()}
        r = {"q":(qs,qe), "r2":r_rsqr.tolist(), "mabs":r_medabs.tolist(), "rmse":r_msqr.tolist()}
        c_tst = {"q":(qs,qe), "r2":c_tst_rsqr.tolist(), "mabs":c_tst_medabs.tolist(), "rmse":c_tst_msqr.tolist()}
        p_tst = {"q":(qs,qe), "r2":p_tst_rsqr.tolist(), "mabs":p_tst_medabs.tolist(), "rmse":p_tst_msqr.tolist()}

        cens1 = meta["cens_perc"]
        cens2 = meta["cens_perc2"]
        # if plot_all:
        #     fig = pltf.plots3(uniq_t, sv_t_0, cph_sv_0, p_sv_0, r_sv_0)
        #     title = f"{scenario['type']}, n {n}"
        #     fig.suptitle(title)
        #     fig_l.append(fig)
        #     if i < strt+10:
        #         ttl = scenario["type"]
        #         namesss = f"../figs/{ttl}_{i}.png"
        #         fig.savefig(namesss)
    
    # if not plot_all:
    #     fig = pltf.plots3(meta, sv_t_0, cph_sv_0, p_sv_0, r_sv_0)
    #     title = f"{scenario['type']}, n {n}"
    #     fig.suptitle(title)
    #     fig_l.append(fig)

    # c,p,r = fn.get_metrics3(
    #     sv_true_lst0,
    #     cph_sv_lst0,
    #     pb_sv_lst0,
    #     r_sv_lst0,
    # )
    # cens1 = [m["cens_perc"] for m in meta_lst]
    # print(f"censs {cens1}")
    # cens2 = [m["cens_perc2"] for m in meta_lst]
    # cens1 = np.array([m["cens_perc"] for m in meta_lst]).mean()
    # cens2 = np.array([m["cens_perc2"] for m in meta_lst]).mean()
    
    return meta_lst, (strt, end), (cens1,cens2), c,p,r, c_tst, p_tst

