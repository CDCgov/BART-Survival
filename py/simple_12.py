import sys
sys.path.append("../py/")

import numpy as np
import subprocess
import importlib
import mlflow as ml

import _conditions1 as cn
import _functions1 as fn
import _sim_iter1 as si

def log_params(name, n, scenario, model_dict, sampler_dict):
    n_dict = {"N":n}
    oname = f"{name}_{n}_n.json"
    ml.log_dict(n_dict, oname)
    oname = f"{name}_{n}_scenario.json"
    ml.log_dict(scenario, oname)
    oname = f"{name}_{n}_model_dict.json"
    ml.log_dict(model_dict, oname)
    oname = f"{name}_{n}_sampler_dict.json"
    ml.log_dict(sampler_dict, oname)


def log_mets(name, n, cens, k, p, r):
    cens = {"cens_retrieved":cens}
    oname = f"{name}_{n}_cens.json"
    ml.log_dict(cens, oname)
    k = dict([(i,k[i].tolist()) for i in k.keys()])
    oname = f"{name}_{n}_met_k.json"
    ml.log_dict(k, oname)
    p = dict([(i,p[i].tolist()) for i in p.keys()])
    oname = f"{name}_{n}_met_p.json"
    ml.log_dict(p, oname)
    r = dict([(i,r[i].tolist()) for i in r.keys()])
    oname = f"{name}_{n}_met_r.json"
    ml.log_dict(r, oname)


def log_figures(name, n, fig):
    oname = f"{name}_{n}.png"
    ml.log_figure(fig, oname)


def main():
    ml.set_tracking_uri("../mlruns")
    exp_name = "test2"
    if ml.get_experiment_by_name(exp_name) == None:
        exp_id = ml.create_experiment(exp_name)
        print("created experiment")
    else:
        exp_id = ml.set_experiment(exp_name)
    print(exp_id)
    
    iters = 2
    # Cond 1
    SPLIT_RULES1 =  [
        "pmb.ContinuousSplitRule()", 
    ]
    model_dict1 = {"trees": 20,
        "split_rules": SPLIT_RULES1
    }
    sampler_dict1 = {
        "draws": 100,
        "tune": 25,
        "cores": 7,
        "chains": 7,
        "compute_convergence_checks": False
    }
    # Cond 2
    SPLIT_RULES2 =  [
        "pmb.ContinuousSplitRule()", 
        "pmb.OneHotSplitRule()"
    ]
    model_dict2 = {"trees": 20,
        "split_rules": SPLIT_RULES2
    }
    sampler_dict2 = {
        "draws": 100,
        "tune": 25,
        "cores": 7,
        "chains": 7,
        "compute_convergence_checks": False
    }

    ml.start_run(run_name="run2")
    ml.log_dict({"iters":iters}, "iters.json")

    for N in cn.N:
    # for N in [100]:        
        # cond 1_1
        meta,cens, k,p,r,fig = si.iter_simulation_1s(
            iters=iters, 
            n=N,
            scenario= cn.simple_1_1, 
            SPLIT_RULES=SPLIT_RULES1, 
            model_dict=model_dict1, 
            sampler_dict=sampler_dict1
        )
        log_params("1_1", N, cn.simple_1_1, model_dict1, sampler_dict1)
        log_mets("1_1", N, cens, k, p, r)
        log_figures("1_1",N,fig)

        # cond 1_2
        meta, cens,k,p,r, fig= si.iter_simulation_1s(
            iters=iters, 
            n=N,
            scenario= cn.simple_1_2, 
            SPLIT_RULES=SPLIT_RULES1, 
            model_dict=model_dict1, 
            sampler_dict=sampler_dict1
        )
        log_params("1_2", N, cn.simple_1_2, model_dict1, sampler_dict1)
        log_mets("1_2", N, cens, k, p, r)
        log_figures("1_2",N,fig)
        # cond 2_1
        meta, cens,k,p,r,fig = si.iter_simulation_2s(
            iters=iters, 
            n=N,
            scenario= cn.simple_2_1, 
            SPLIT_RULES=SPLIT_RULES2, 
            model_dict=model_dict2, 
            sampler_dict=sampler_dict2
        )
        log_params("2_1", N, cn.simple_2_1, model_dict2, sampler_dict2)
        log_mets("2_1", N, cens, k, p, r)
        log_figures("2_1",N,fig)

        #cond 2_2
        meta, cens,k,p,r,fig = si.iter_simulation_2s(
            iters=iters, 
            n=N,
            scenario= cn.simple_2_2, 
            SPLIT_RULES=SPLIT_RULES2, 
            model_dict=model_dict2, 
            sampler_dict=sampler_dict2
        )
        log_params("2_1", N, cn.simple_2_2, model_dict2, sampler_dict2)
        log_mets("2_2", N, cens, k, p, r)
        log_figures("2_2",N,fig)

    ml.end_run()

if __name__ == "__main__":
    main()