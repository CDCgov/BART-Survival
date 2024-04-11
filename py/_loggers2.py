import mlflow as ml

def log_params(name, n, scenario, model_dict, sampler_dict, seeds):
    # n_dict = {"N":n}
    # oname = f"{name}_{n}_n.json"
    
    outdict = {
        "N":n,
        "SCENARIO":scenario,
        "MODEL_DICT":model_dict,
        "SAMPLER_DICT":sampler_dict,
        "SEEDS": seeds
    }
    oname = f"{name}_{n}_params.json"
    ml.log_dict(outdict, oname)

    # ml.log_dict(n_dict, oname)
    # oname = f"{name}_{n}_scenario.json"
    # ml.log_dict(scenario, oname)
    # oname = f"{name}_{n}_model_dict.json"
    # ml.log_dict(model_dict, oname)
    # oname = f"{name}_{n}_sampler_dict.json"
    # ml.log_dict(sampler_dict, oname)
    # oname = f"{name}_{n}_seeds.json"
    # seeds = {"seeds":seeds}
    # ml.log_dict(seeds, oname)

def log_stats(name, n, hc, cc, hi, ci, rmse, bias, tst=False):
    if tst:
        oname = f"{name}_{n}_stats_tst.json"
    else:
        oname = f"{name}_{n}_stats.json"

    odict = {
        "cov_hdi":hc,
        "cov_ci":cc,
        "iv_hdi":hi,
        "iv_ci":ci,
        "rmse":rmse,
        "bias":bias
    }
    ml.log_dict(odict, oname)


    # oname = f"{name}_{n}_cov_hdi.json"
    # ml.log_dict(hc, oname)
    # oname = f"{name}_{n}_cov_ci.json"
    # ml.log_dict(cc, oname)
    # oname = f"{name}_{n}_iv_hdi.json"
    # ml.log_dict(hi, oname)
    # oname = f"{name}_{n}_iv_ci.json"
    # ml.log_dict(ci, oname)
    # oname = f"{name}_{n}_rmse.json"
    # ml.log_dict(rmse, oname)
    # oname = f"{name}_{n}_bias.json"
    # ml.log_dict(bias, oname)
    
