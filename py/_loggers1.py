import mlflow as ml

def log_params(name, n, scenario, model_dict, sampler_dict, seeds):
    n_dict = {"N":n}
    oname = f"{name}_{n}_n.json"
    ml.log_dict(n_dict, oname)
    oname = f"{name}_{n}_scenario.json"
    ml.log_dict(scenario, oname)
    oname = f"{name}_{n}_model_dict.json"
    ml.log_dict(model_dict, oname)
    oname = f"{name}_{n}_sampler_dict.json"
    ml.log_dict(sampler_dict, oname)
    onname = f"{name}_{n}_seeds.json"
    seeds = {"seeds":seeds}
    ml.log_dict(seeds, oname)

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