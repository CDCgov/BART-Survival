import numpy as np
import sksurv as sks
from bart_survival import surv_bart as sb 
from bart_survival import simulation as sm
import matplotlib.pyplot as plt

def main():
	rng = np.random.default_rng(1)

	# create the covariate matrix
	# - 100 obs, 5 vars, [binary, binary, 0-5, 0-1, 0-1]
	x_mat = sm.get_x_matrix(
		N=1000,
		x_vars=5,
		VAR_CLASS=[2,2,1],
		VAR_PROB=[.5, .7, None],
		rng = rng
	)

	x_mat[0:10,:]
	event_dict, sv_true, sv_scale_true = sm.simulate_survival(
		x_mat = x_mat,
		scale_f = "np.exp(4 + .4*x_mat[:,0] + .1*x_mat[:,1] + .01*x_mat[:,2])", # note that x_mat[:,3] and x_mat[:,4] are not included
		shape_f = "1", # hazard is constant
		eos = 180,
		cens_scale=None,
		time_scale=60,
		true_only=False,
		rng = rng
	)

	t_scale = sb.get_time_transform(event_dict["t_event"], time_scale = 60)
	y_sk = sb.get_y_sklearn(event_dict["status"], t_scale)
	trn = sb.get_surv_pre_train(y_sk, x_mat, weight=None)
	post_test = sb.get_posterior_test(y_sk=y_sk, x_test = x_mat)

	SPLIT_RULES =  [
		"pmb.ContinuousSplitRule()", # time
		"pmb.OneHotSplitRule", # x_mat[:,0]
		"pmb.OneHotSplitRule", # x_mat[:,1]
		"pmb.ContinuousSplitRule()", # x_mat[:,2]
		"pmb.ContinuousSplitRule()", # x_mat[:,3]
		"pmb.ContinuousSplitRule()", # x_mat[:,4]
	]

	model_dict = {"trees": 40,
		"split_rules": SPLIT_RULES
	}
	sampler_dict = {
				"draws": 200,
				"tune": 200,
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

if __name__ == "__main__":
	main()