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
    oname = f"{name}_{n}_seeds.json"
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

def log_mets2(name, n, cens, k, p, r):
    cens = {"cens_retrieved":cens}
    oname = f"{name}_{n}_cens.json"
    ml.log_dict(cens, oname)
    k = dict([(i,k[i].tolist()) for i in k.keys()])
    oname = f"{name}_{n}_met_c.json"
    ml.log_dict(k, oname)
    p = dict([(i,p[i].tolist()) for i in p.keys()])
    oname = f"{name}_{n}_met_p.json"
    ml.log_dict(p, oname)
    r = dict([(i,r[i].tolist()) for i in r.keys()])
    oname = f"{name}_{n}_met_r.json"
    ml.log_dict(r, oname)

def log_mets3(name, n, cens, c, p, r, c_tst, p_tst):
    cens = {"cens_retrieved":cens}
    oname = f"{name}_{n}_cens.json"
    ml.log_dict(cens, oname)
    c = dict([(i,c[i]) for i in c.keys()])
    oname = f"{name}_{n}_met_c.json"
    ml.log_dict(c, oname)
    p = dict([(i,p[i]) for i in p.keys()])
    oname = f"{name}_{n}_met_p.json"
    ml.log_dict(p, oname)
    r = dict([(i,r[i]) for i in r.keys()])
    oname = f"{name}_{n}_met_r.json"
    ml.log_dict(r, oname)
    c_tst = dict([(i,c_tst[i]) for i in c_tst.keys()])
    oname = f"{name}_{n}_met_c_tsts.json"
    ml.log_dict(c_tst, oname)
    p_tst = dict([(i,p_tst[i]) for i in p_tst.keys()])
    oname = f"{name}_{n}_met_p_tst.json"
    ml.log_dict(p_tst, oname)

def log_figures(name, n, fig):
    oname = f"{name}_{n}.png"
    ml.log_figure(fig, oname)