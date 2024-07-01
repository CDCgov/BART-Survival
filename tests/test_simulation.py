from bart_survival import simulation
import numpy as np

def test_weibull_survival_components():
    iws = simulation.inverse_weibull_survival(scale = 1,shape =2, u = np.array([.1,.5,.9]))
    wh = simulation.weibull_hazard(scale=1, shape=2, t=np.array([1,2]))
    ws = simulation.weibull_survival(scale=1, shape=2, t=np.array([1,2]))

    assert (np.allclose(iws, np.array([[1.51742713],[0.83255461],[0.32459285]])))
    assert (np.allclose(wh, np.array([[2., 4.]])))
    assert (np.allclose(ws, np.array([[0.36787944, 0.01831564]])))

def test_check_inputs():
    try:
        simulation.check_inputs(scale = -1, shape = -1, u = -0.1, t=-1)
    except Exception as e:
        error = str(e)
    assert (error == "['u must be between 0 and 1', 'scale must be > 0', 'shape must be > 0', 't must be > 0']")

def test_get_x_matrix():
    rng = np.random.default_rng(1)
    x_mat = simulation.get_x_matrix(
        N = 2,
        x_vars = 5, 
        VAR_CLASS = [2,2,10,100], 
        VAR_PROB = [.6, .8, None, None],
        rng = rng
    )
    assert(np.allclose(x_mat, [[1.00000000e+00, 1.00000000e+00, 3.11800000e+00, 8.27700000e+01, 5.49593688e-01],[0.00000000e+00, 0.00000000e+00, 4.23300000e+00, 4.09200000e+01, 2.75591132e-02]]))

def test_simulate_survival_1():
    #test that downscale time works
    rng = np.random.default_rng(2)
    # x_mat = np.array([[1],[1],[1],[1]])
    x_mat = np.repeat([1,0],10).reshape(-1,1)

    event_dict, sv_true, sv_scale_true = simulation.simulate_survival(
        x_mat = x_mat,
        scale_f = "np.exp(4 + 1.5*x_mat[:,0])",
        shape_f = "1",
        eos = 180,
        cens_scale=None,
        time_scale=30,
        true_only=False,
        rng = rng
    )
    # eos set to 180
    assert(event_dict["t_event"].max() == 180)
    ev_msk = event_dict["t_event"]==180
    assert(all(event_dict["status"][ev_msk] == False) == True)
    # check scaling worked
    tdiff = sv_scale_true["true_times"][1] - sv_scale_true["true_times"][0] 
    assert(tdiff == 30)
    assert(sv_scale_true["scale_times"].shape[0] == 180/30)

def test_simulate_survival_2():
    #test that survival and hz generated survival align
    rng = np.random.default_rng(2)
    x_mat = np.repeat([1,0],10).reshape(-1,1)

    event_dict, sv_true, sv_scale_true = simulation.simulate_survival(
        x_mat = x_mat,
        scale_f = "np.exp(1 + 1.5*x_mat[:,0])",
        shape_f = "1",
        eos = None,
        cens_scale=None,
        time_scale=None,
        true_only=False,
        rng = rng
    )
    assert (
        np.allclose(sv_true["sv_true"][0], np.exp(-np.cumsum(sv_true["hz_true"], axis = 1))[0])
    )

def test_simulate_survival_3():
    #test that expected Hazard Ratio and Generated Hazard Ratio are equivalent
    rng = np.random.default_rng(2)
    # x_mat = np.array([[1],[1],[1],[1]])
    x_mat = np.repeat([1,0],10).reshape(-1,1)
    event_dict, sv_true, sv_scale_true = simulation.simulate_survival(
        x_mat = x_mat,
        scale_f = "np.exp(1 + 1.5*x_mat[:,0])",
        shape_f = "1",
        eos = None,
        cens_scale=None,
        time_scale=None,
        true_only=False,
        rng = rng
    )
    hz = sv_true["hz_true"]
    hr = hz[1,:]/hz[10,:]
    # expected HR as derived from the scale_f parameter
    exp_hr = np.power(np.exp(-1.5), 1)
    # print(exp_hr)
    assert(np.allclose(hr[0],exp_hr))





if __name__ == "__main__":
    test_check_inputs()
    test_weibull_survival_components()
    test_get_x_matrix()
    test_simulate_survival_1()
    test_simulate_survival_2()
    test_simulate_survival_3()
    print("Everything passed")
    