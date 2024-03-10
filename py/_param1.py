EXP_NAME = "test3"
RUN_NAME = "run1_100"

###########################
# ITERATIONS
ITERS = 100
SEED_ADDL=2
###########################
# Cond 1
SPLIT_RULES1 =  [
	"pmb.ContinuousSplitRule()", 
]
MODEL_DICT1 = {"trees": 40,
	"split_rules": SPLIT_RULES1
}
SAMPLER_DICT1 = {
	"draws": 620,
	"tune": 30,
	"cores": 3,
	"chains": 3,
	"compute_convergence_checks": False
}

###########################
# Cond 2
SPLIT_RULES2 =  [
	"pmb.ContinuousSplitRule()", 
	"pmb.OneHotSplitRule()"
]
MODEL_DICT2 = {"trees": 40,
	"split_rules": SPLIT_RULES2
}
SAMPLER_DICT2 = {
	"draws": 620,
	"tune": 30,
	"cores": 3,
	"chains": 3,
	"compute_convergence_checks": False
}
