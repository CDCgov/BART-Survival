"""Integration test for bart_survival.
"""
from bart_survival import simulation
from bart_survival import surv_bart as sb
import numpy as np


def test_surv_bart():
    rng = np.random.default_rng(1)
    x_mat = simulation.get_x_matrix(
        N=100,
        x_vars=1,
        VAR_CLASS=[2],
        VAR_PROB=[.5],
        rng = rng
    )
    event_dict, sv_true, sv_scale_true = simulation.simulate_survival(
        x_mat = x_mat,
        scale_f = "np.exp(2 + .4*x_mat[:,0])",
        shape_f = "1",
        eos = 20,
        cens_scale=None,
        time_scale=5,
        true_only=False,
        rng = rng
    )
    # prepare data
    t_scale = sb.get_time_transform(event_dict["t_event"], time_scale = 5)
    y_sk = sb.get_y_sklearn(event_dict["status"], t_scale)
    trn = sb.get_surv_pre_train(y_sk, x_mat, weight=None)
    post_test = sb.get_posterior_test(y_sk=y_sk, x_test = x_mat)

    SPLIT_RULES =  [
        "pmb.ContinuousSplitRule()", # time
        "pmb.OneHotSplitRule", # ccsr_ind_p2
    ]
    model_dict = {"trees": 50,
        "split_rules": SPLIT_RULES
    }
    sampler_dict = {
                "draws": 100,
                "tune": 100,
                "cores": 8,
                "chains": 8,
                "compute_convergence_checks": False
            }

    BSM = sb.BartSurvModel(model_config=model_dict, sampler_config=sampler_dict)

    BSM.fit(
        y =  trn["y"],
        X = trn["x"],
        weights=trn["w"],
        coords = trn["coord"],
        random_seed=5
    )

    post1 = BSM.sample_posterior_predictive(X_pred=post_test["post_x"], coords=post_test["coords"])
    sv_prob = sb.get_sv_prob(post1)
    # allow 10% difference
    assert(.1>np.abs((sv_prob["sv"].mean(0)-sv_scale_true["sv_true"]).mean()))

    # pdp
    pdp1 = sb.get_pdp(x_mat, var_col = [0], values = [[0,1]], sample_n = None)
    pdp_tst = sb.get_posterior_test(y_sk, pdp1[0])
    pdp_post = BSM.sample_posterior_predictive(pdp_tst["post_x"], pdp_tst["coords"])

    sv_prob = sb.get_sv_prob(pdp_post)
    msk_1 = pdp1[1]["coord"] == 1
    HR = (sv_prob["prob"][:,msk_1,:].mean(1) / sv_prob["prob"][:,~msk_1,:].mean(1)).mean(0)
    HR_QT = np.quantile((sv_prob["prob"][:,msk_1,:].mean(1) / sv_prob["prob"][:,~msk_1,:].mean(1)),[0.025,0.975])
    
    # allow 20% difference
    assert(.2 > np.abs(np.exp(-.4) - HR.mean()))

    print(HR)
    print(HR_QT)

if __name__ == "__main__":
    test_surv_bart()
    print("Everything passed")