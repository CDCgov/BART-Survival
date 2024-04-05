import sys
sys.path.append("../py/")

import numpy as np
import subprocess
import importlib
import mlflow as ml

import _conditions2 as cn
import _functions1 as fn
import _sim_iter2 as si
import _loggers1 as lo
import _param2 as prm
import time

# TODO  add in the figures
# add in the other settings
# test w/ multiple iterations

def main():
    ##########################
    # set up 
    ml.set_tracking_uri("../mlruns")
    exp_name = prm.EXP_NAME
    if ml.get_experiment_by_name(exp_name) == None:
        exp_id = ml.create_experiment(exp_name)
        exp_id = ml.set_experiment(exp_name)
        print("created experiment")
    else:
        exp_id = ml.set_experiment(exp_name)
    print(exp_id)
    
    ##############################
    # start run
    ml.start_run(run_name=prm.RUN_NAME_cmplx3)
    ml.log_dict({"iters":prm.ITERS_cmplx3}, "iters.json")
    ml.log_dict({"seed_addl":prm.SEED_ADDL}, "seed_addl.json")

    # should only be one N loop as N==400
    for N in prm.N_cmplx3:
    # for N in [200]:
        # complex 3
        if True:
            strt_time = time.time()
            meta, seeds, cens,c,p,r, c_tst, p_tst = si.iter_simulation_complex3(
                iters=prm.ITERS_cmplx3, 
                n=N,
                seed_addl=prm.SEED_ADDL,
                scenario= cn.complex_3, 
                SPLIT_RULES=prm.SPLIT_RULES3, 
                model_dict=prm.MODEL_DICT3,
                sampler_dict=prm.SAMPLER_DICT3,
                plot_all=prm.PLOT_ALL
            )
            lo.log_params("cmplx3", N, cn.complex_3, prm.MODEL_DICT3, prm.SAMPLER_DICT3, seeds)
            lo.log_mets3("cmplx3", N, cens, c, p, r, c_tst, p_tst)

            # probs don't need to log
            # if prm.PLOT_ALL:
            #     for f in range(len(fig)):
            #         if f in [0,1,2,3]:
            #             lo.log_figures(f"cmplx_3_{f}", N, fig[f])
            # else:
            #     lo.log_figures("cmplx_3",N,fig[0])

            end_time = time.time()-strt_time
            tname = f"cmplx_3_time_{N}.json"
            ml.log_dict({"time":end_time}, tname)
            print("DONE complex 3")

    ml.end_run()

if __name__ == "__main__":
    main()