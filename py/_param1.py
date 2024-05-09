EXP_NAME = "simple1"
# RUN_NAME = "simple_12_200_iters_400_n"
RUN_NAME = "test_simple2"

###########################
# ITERATIONS
ITERS = 2
SEED_ADDL=13
# N  = [200, 400]
N  = [400]

###########################
# Cond 1
SPLIT_RULES1 =  [
	"pmb.ContinuousSplitRule()", 
]
MODEL_DICT1 = {"trees": 20,
	"split_rules": SPLIT_RULES1
}
SAMPLER_DICT1 = {
	"draws": 400,
	"tune": 100,
	"cores": 5,
	"chains": 5,
	"compute_convergence_checks": False
}


###########################
# Cond 2
SPLIT_RULES2 =  [
	"pmb.ContinuousSplitRule()", 
	"pmb.OneHotSplitRule()"	
	# "pmb.ContinuousSplitRule()"
]
MODEL_DICT2 = {"trees": 20,
	"split_rules": SPLIT_RULES2,
	# "split_prior":[10,10]
}
SAMPLER_DICT2 = {
	"draws": 500,
	"tune": 100,
	"cores": 5,
	"chains": 5,
	"compute_convergence_checks": False
}
PLOT_ALL=True