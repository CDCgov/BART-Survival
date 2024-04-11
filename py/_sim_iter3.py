import pandas as pd
# import numpy as np
# import lifelines as ll
# import sksurv as sks
import matplotlib.pyplot as plt
# from bart_survival import surv_bart as sb
# from bart_survival import simulation as sm
# import lifelines as ll
# from lifelines import KaplanMeierFitter
import subprocess
import importlib
import _functions1 as fn
import _conditions1 as cn
import _plot_fx as pltf
plt.ioff()
importlib.reload(pltf)
importlib.reload(fn)
importlib.reload(cn)

# parameter check
def iter_simulation_prm1(iters, n, seed_addl, scenario, SPLIT_RULES, model_dict, sampler_dict):
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
    cov_hdi = []
    cov_ci = []
    iv_hdi = []
    iv_ci = []
    rmse = []
    bias = []
    cov_hdi_tst = []
    cov_ci_tst = []
    iv_hdi_tst = []
    iv_ci_tst = []
    rmse_tst = []
    bias_tst = []
    figs = []
    seeds = []

    strt = n*seed_addl
    end = strt + iters
    for i in range(strt, end):
        print(f"ITERATION************** {i}")
        seeds.append(i) # seeds should be the same for each setting
        # only need pb_sv and pb_sv_ci
        sv_true, sv_true_tst, pb_sv, pb_sv_tst, uniq_t, stats, stats_tst, intvs, intvs_tst = fn.sim_prm_1s(seed=i, n=n, scenario_=scenario, SPLIT_RULES=SPLIT_RULES, model_dict=model_dict, sampler_dict=sampler_dict)


        cov_hdi.append(stats["cov_hdi"])
        cov_ci.append(stats["cov_ci"])
        iv_hdi.append(stats["iv_hdi"])
        iv_ci.append(stats["iv_ci"])
        rmse.append(stats["rmse"])
        bias.append(stats["bias"])

        cov_hdi_tst.append(stats_tst["cov_hdi"])
        cov_ci_tst.append(stats_tst["cov_ci"])
        iv_hdi_tst.append(stats_tst["iv_hdi"])
        iv_ci_tst.append(stats_tst["iv_ci"])
        rmse_tst.append(stats_tst["rmse"])
        bias_tst.append(stats_tst["bias"])
        
        
        if i < strt + 2:
            sv_truet = sv_true["sv_true"][:,(uniq_t-1).astype("int")]
            sv_truet_tst = sv_true_tst["sv_true"][:,(uniq_t-1).astype("int")]
            pb_svm = pb_sv["sv"].mean(0)
            pb_svm_tst = pb_sv_tst["sv"].mean(0)
            fig,ax = plt.subplots(1,4, figsize=(40,8))
            for j in range(100):
                if j == 0:
                    ax[0].step(uniq_t, sv_truet[j,:], color="black", label = "true", alpha=0.1)
                    ax[1].step(uniq_t, sv_truet[j,:], color="black", label= "true", alpha=0.1)

                    ax[0].step(uniq_t, pb_svm[j,:], color = "green", alpha=0.4, label="pred")
                    ax[1].step(uniq_t, pb_svm[j,:], color = "blue", alpha=0.3, label = "pred")

                    ax[0].step(uniq_t, intvs[0][j, :, 0], color="lightgreen", alpha=0.1, label = "95 hdi")
                    ax[0].step(uniq_t, intvs[0][j, :, 1], color= "lightgreen", alpha=0.1)
                    
                    ax[1].step(uniq_t, intvs[1][0, j, :], color="lightblue", alpha=0.1, label = "95 ci")
                    ax[1].step(uniq_t, intvs[1][1, j, :], color= "lightblue", alpha=0.1)
                    ax[0].legend()
                    ax[1].legend()

                    # tst
                    ax[2].step(uniq_t, sv_truet_tst[j,:], color="black", label = "true", alpha=0.1)
                    ax[3].step(uniq_t, sv_truet_tst[j,:], color="black", label= "true", alpha=0.1)

                    ax[2].step(uniq_t, pb_svm_tst[j,:], color = "green", alpha=0.4, label="pred")
                    ax[3].step(uniq_t, pb_svm_tst[j,:], color = "blue", alpha=0.3, label = "pred")

                    ax[2].step(uniq_t, intvs_tst[0][j, :, 0], color="lightgreen", alpha=0.1, label = "95 hdi")
                    ax[2].step(uniq_t, intvs_tst[0][j, :, 1], color= "lightgreen", alpha=0.1)
                    
                    ax[3].step(uniq_t, intvs_tst[1][0, j, :], color="lightblue", alpha=0.1, label = "95 ci")
                    ax[3].step(uniq_t, intvs_tst[1][1, j, :], color= "lightblue", alpha=0.1)
                    ax[2].legend()
                    ax[3].legend()
                else:
                    ax[0].step(uniq_t, sv_truet[j,:], color="black", alpha=0.1)
                    ax[1].step(uniq_t, sv_truet[j,:], color="black", alpha=0.1)

                    ax[0].step(uniq_t, pb_svm[j,:], color = "green", alpha=0.4)
                    ax[1].step(uniq_t, pb_svm[j,:], color = "blue", alpha=0.3)

                    ax[0].step(uniq_t, intvs[0][j, :, 0], color="lightgreen", alpha=0.1)
                    ax[0].step(uniq_t, intvs[0][j, :, 1], color= "lightgreen", alpha=0.1)
                    
                    ax[1].step(uniq_t, intvs[1][0, j, :], color="lightblue", alpha=0.1)
                    ax[1].step(uniq_t, intvs[1][1, j, :], color= "lightblue", alpha=0.1)

                    #tst
                    ax[2].step(uniq_t, sv_truet_tst[j,:], color="black", alpha=0.1)
                    ax[3].step(uniq_t, sv_truet_tst[j,:], color="black", alpha=0.1)

                    ax[2].step(uniq_t, pb_svm_tst[j,:], color = "green", alpha=0.4)
                    ax[3].step(uniq_t, pb_svm_tst[j,:], color = "blue", alpha=0.3)

                    ax[2].step(uniq_t, intvs_tst[0][j, :, 0], color="lightgreen", alpha=0.1)
                    ax[2].step(uniq_t, intvs_tst[0][j, :, 1], color= "lightgreen", alpha=0.1)
                    
                    ax[3].step(uniq_t, intvs_tst[1][0, j, :], color="lightblue", alpha=0.1)
                    ax[3].step(uniq_t, intvs_tst[1][1, j, :], color= "lightblue", alpha=0.1)
            # ttl = scenario["type"]
            tune = sampler_dict["tune"]
            draw = sampler_dict["draws"]
            trees = model_dict["trees"]
            namesss = f"../figs/prm2/prm_tune_{tune}_draw_{draw}_trees_{trees}_{i}.png"
            fig.savefig(namesss)
    return cov_hdi, cov_ci, iv_hdi, iv_ci, rmse, bias, cov_hdi_tst, cov_ci_tst, iv_hdi_tst, iv_ci_tst, rmse_tst, bias_tst, seeds
    
