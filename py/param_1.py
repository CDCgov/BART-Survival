# test the params
# trees, alpha, beta, tune draws

import sys
sys.path.append("../py/")

import numpy as np
import subprocess
import importlib
import mlflow as ml

# import _conditions2 as cn
# import _functions1 as fn
import _sim_iter3 as si
import _loggers2 as lo
# import _param2 as prm
import time

EXP_NAME = "param_eval1"
RUN_NAME = "param_run2_f2"
N = 1000
TREES = [20,50,100,200]
TUNE = [100,1000]
DRAW = [500, 1000, 2000]
CHAINS = 5
SEED_ADDL=99
ITERS = 1

# SCENARIO = {
#     "type": "param_check_1",
#     "x_vars": 10, 
#     "VAR_CLASS": [2,2,1,1,1],
#     "VAR_PROB":[.5,.5,None,None,None],
#     # "scale_f": "2.5*x_mat[:,0]",
#     "scale_f": "10 + 0.8*x_mat[:,0] + .8*x_mat[:,0]*x_mat[:,1] -.4*x_mat[:,1] + .2*x_mat[:,2] + 1.2 * x_mat[:,3] + .2 * x_mat[:,4]",
#     "shape_f": "3 - 1.5 * x_mat[:,0] + .2 * x_mat[:,1]",
#     "cens_scale":None
# }

SCENARIO = {
    "type": "param_check_1",
    "x_vars": 10, 
    "VAR_CLASS": [2,2,1,1,1],
    "VAR_PROB":[.5,.5,None,None,None],
    # "scale_f": "10+ 2.5*x_mat[:,0]",
    "scale_f": "20 + 4*x_mat[:,0] + 5*x_mat[:,1] + (1.4*x_mat[:,2])**2 - 2*x_mat[:,3]*x_mat[:,4]",
    "shape_f": "1.6 + 1.2 * x_mat[:,0] ",
    "cens_scale":None
}

SPLT_R = [
    "pmb.ContinuousSplitRule()", # time
    "pmb.OneHotSplitRule()",
    "pmb.OneHotSplitRule()",
    "pmb.ContinuousSplitRule()",
    "pmb.ContinuousSplitRule()",
    "pmb.ContinuousSplitRule()", 
    "pmb.ContinuousSplitRule()", # addl
    "pmb.ContinuousSplitRule()", 
    "pmb.ContinuousSplitRule()", 
    "pmb.ContinuousSplitRule()", 
    "pmb.ContinuousSplitRule()", 
]


def main():
    ##########################
    # set up 
    ml.set_tracking_uri("../mlruns")
    exp_name = EXP_NAME # need
    if ml.get_experiment_by_name(exp_name) == None:
        exp_id = ml.create_experiment(exp_name)
        exp_id = ml.set_experiment(exp_name)
        print("created experiment")
    else:
        exp_id = ml.set_experiment(exp_name)
    print(exp_id)
    
    ##############################
    # start run
    ml.start_run(run_name=RUN_NAME)
    ml.log_dict({"iters":ITERS}, "iters.json")
    ml.log_dict({"seed_addl":SEED_ADDL}, "seed_addl.json")
    for tree in TREES:
        for tune in TUNE:
            for draw in DRAW:
                
                MDICT = {
                    "trees":tree,
                    "split_rules": SPLT_R
                }
                SDICT = {
                    "draws": draw,
                    "tune": tune,
                    "cores": CHAINS,
                    "chains": CHAINS,
                    "compute_convergence_checks":False
                }

                if True:
                    strt_time = time.time()
                    # cov_hdi, cov_ci, iv_hdi, iv_ci, rmse, bias, seeds = si.iter_simulation_prm1(
                    cov_hdi, cov_ci, iv_hdi, iv_ci, rmse, bias, cov_hdi_tst, cov_ci_tst, iv_hdi_tst, iv_ci_tst, rmse_tst, bias_tst, seeds = si.iter_simulation_prm1(
                        iters=ITERS, 
                        n=N,
                        seed_addl=SEED_ADDL,
                        scenario= SCENARIO, 
                        SPLIT_RULES=SPLT_R, 
                        model_dict= MDICT, 
                        sampler_dict=SDICT,
                    )

                    # namesss = f"../figs/prm_tune_{tune}_draw_{draw}_trees_{trees}_{i}.png"
                    oname = f"prm_tune_{tune}_draw_{draw}_trees_{tree}"

                    lo.log_params(oname, N, SCENARIO, MDICT, SDICT, seeds)
                    lo.log_stats(oname, N, cov_hdi, cov_ci, iv_hdi, iv_ci, rmse, bias)
                    lo.log_stats(oname, N, cov_hdi_tst, cov_ci_tst, iv_hdi_tst, iv_ci_tst, rmse_tst, bias_tst, tst=True)

                    end_time = time.time()-strt_time
                    tname = f"{oname}_{N}_time.json"
                    ml.log_dict({"time":end_time}, tname)
                    stm = f"Done {tree}, {tune}, {draw}"
    ml.end_run()

if __name__ == "__main__":
    main()