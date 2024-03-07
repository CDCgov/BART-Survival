import sys
sys.path.append("../py/")

import numpy as np
import subprocess
import importlib
import mlflow as ml

import _conditions1 as cn
import _functions1 as fn
import _sim_iter1 as si

from threading import active_count

SPLIT_RULES1 =  [
    "pmb.ContinuousSplitRule()", 
]
model_dict1 = {"trees": 10,
    "split_rules": SPLIT_RULES1
}
sampler_dict1 = {
            "draws": 200,
            "tune": 10,
            "cores": 7,
            "chains": 7,
            "compute_convergence_checks": False
        }

def main():
    rng = np.random.default_rng(99)
    thr = active_count()
    
    for i in range(10):
        print(f"\n ACTIVE COUNT {thr} \n")
        scenario, x_mat, event_dict, sv_true, sv_scale_true = fn.get_sim(rng, 100, **cn.simple_1_1)
        pb_sv = fn.get_py_bart_surv(x_mat, event_dict, model_dict1, sampler_dict1)
    return None

if __name__ == "__main__":
    main()