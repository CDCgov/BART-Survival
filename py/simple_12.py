import sys
sys.path.append("../py/")

import numpy as np
import subprocess
import importlib
import mlflow as ml

import _conditions1 as cn
import _functions1 as fn
import _sim_iter1 as si
import _loggers1 as lo
import _param1 as prm
import time


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
    ml.start_run(run_name=prm.RUN_NAME)
    ml.log_dict({"iters":prm.ITERS}, "iters.json")
    ml.log_dict({"seed_addl":prm.SEED_ADDL}, "seed_addl.json")

    for N in prm.N:
    # for N in [200]:
        ############################################################        
        # cond 1_1
        if False:
            strt_time = time.time()
            meta, seeds, cens, k,p,r,fig = si.iter_simulation_1s(
                iters=prm.ITERS, 
                n=N,
                seed_addl=prm.SEED_ADDL,
                scenario= cn.simple_1_1, 
                SPLIT_RULES=prm.SPLIT_RULES1, 
                model_dict=prm.MODEL_DICT1, 
                sampler_dict=prm.SAMPLER_DICT1
            )
            lo.log_params("1_1", N, cn.simple_1_1, prm.MODEL_DICT1, prm.SAMPLER_DICT1, seeds)
            lo.log_mets("1_1", N, cens, k, p, r)
            lo.log_figures("1_1",N,fig)
            end_time = time.time()-strt_time
            tname = f"time_1_{N}.json"
            ml.log_dict({"time":end_time}, tname)
            print("DONE SIMPLE_1_1")
        ############################################################
        # cond 1_2
        if False:
            strt_time = time.time()
            meta, seeds, cens,k,p,r, fig= si.iter_simulation_1s(
                iters=prm.ITERS, 
                n=N,
                seed_addl=prm.SEED_ADDL,
                scenario= cn.simple_1_2, 
                SPLIT_RULES=prm.SPLIT_RULES1, 
                model_dict=prm.MODEL_DICT1, 
                sampler_dict=prm.SAMPLER_DICT1
            )
            lo.log_params("1_2", N, cn.simple_1_2, prm.MODEL_DICT1, prm.SAMPLER_DICT1, seeds)
            lo.log_mets("1_2", N, cens, k, p, r)
            lo.log_figures("1_2",N,fig)
            end_time = time.time()-strt_time
            tname = f"time_2_{N}.json"
            ml.log_dict({"time":end_time}, tname)
            print("DONE SIMPLE_1_2")
        ############################################################
        # cond 2_1
        if True:
            strt_time = time.time()
            meta, seeds, cens,k,p,r,fig = si.iter_simulation_2s(
                iters=prm.ITERS, 
                n=N,
                seed_addl=prm.SEED_ADDL,
                scenario= cn.simple_2_1, 
                SPLIT_RULES=prm.SPLIT_RULES2, 
                model_dict=prm.MODEL_DICT2,
                sampler_dict=prm.SAMPLER_DICT2,
                plot_all=prm.PLOT_ALL
            )
            lo.log_params("2_1", N, cn.simple_2_1, prm.MODEL_DICT2, prm.SAMPLER_DICT2, seeds)
            lo.log_mets("2_1", N, cens, k, p, r)

            if prm.PLOT_ALL:
                for f in range(len(fig)):
                    if f in [0,1,2,3]:
                        lo.log_figures(f"2_1_{f}", N, fig[f])
            else:
                lo.log_figures("2_1",N,fig)

            end_time = time.time()-strt_time
            tname = f"time_3_{N}.json"
            ml.log_dict({"time":end_time}, tname)
            print("DONE SIMPLE_2_1")
        ############################################################
        #cond 2_2
        if True:
            strt_time = time.time()
            meta, seeds, cens,k,p,r,fig = si.iter_simulation_2s(
                iters=prm.ITERS, 
                n=N,
                seed_addl=prm.SEED_ADDL,
                scenario= cn.simple_2_2, 
                SPLIT_RULES=prm.SPLIT_RULES2, 
                model_dict=prm.MODEL_DICT2, 
                sampler_dict=prm.SAMPLER_DICT2,
                plot_all=prm.PLOT_ALL
            )
            lo.log_params("2_1", N, cn.simple_2_2, prm.MODEL_DICT2, prm.SAMPLER_DICT2, seeds)
            lo.log_mets("2_2", N, cens, k, p, r)
            
            if prm.PLOT_ALL:
                for f in range(len(fig)):
                    if f in [0,1,2,3]:
                        lo.log_figures(f"2_2_{f}", N, fig[f])
            else:
                lo.log_figures("2_2",N,fig)
                
            end_time = time.time()-strt_time
            tname = f"time_4_{N}.json"
            ml.log_dict({"time":end_time}, tname)
            print("DONE SIMPLE_2_2")

    ml.end_run()

if __name__ == "__main__":
    main()