EXP_NAME = "test3"
RUN_NAME = "run_t2_200_1"

###########################
# ITERATIONS
ITERS = 200
SEED_ADDL=13
N = [200, 400]
###########################
# Cond 1
SPLIT_RULES1 =  [
	"pmb.ContinuousSplitRule()", 
]
MODEL_DICT1 = {"trees": 15,
	"split_rules": SPLIT_RULES1
}
SAMPLER_DICT1 = {
	"draws": 500,
	"tune": 100,
	"cores": 4,
	"chains": 4,
	"compute_convergence_checks": False
}

###########################
# Cond 2
SPLIT_RULES2 =  [
	"pmb.ContinuousSplitRule()", 
	# "pmb.OneHotSplitRule()"	
	"pmb.ContinuousSplitRule()"
]
MODEL_DICT2 = {"trees": 60,
	"split_rules": SPLIT_RULES2,
	# "split_prior":[10,10]
}
SAMPLER_DICT2 = {
	"draws": 900,
	"tune": 10,
	"cores": 5,
	"chains": 5,
	"compute_convergence_checks": False
}
PLOT_ALL=True