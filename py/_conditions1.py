#simple
# N = [200,400]


# simple_1_1 = {
#     "type": "Simple, 1 group, 20% cens",
#     "x_vars": 1, 
#     "VAR_CLASS": [2],
#     "VAR_PROB":[1],
#     "scale_f": "2.5*x_mat[:,0]",
#     "shape_f": ".8",
#     "cens_scale":3.3
# }
# simple_1_2 = {
#     "type": "Simple, 1 group, 50% cens",
#     "x_vars": 1, 
#     "VAR_CLASS": [2],
#     "VAR_PROB":[1],
#     "scale_f": "2.5*x_mat[:,0]",
#     "shape_f": ".8",
#     "cens_scale":.5,
# }

simple_1_1 = {
    "type": "Simple, 1 group, 20% cens",
    "x_vars": 1, 
    "VAR_CLASS": [2],
    "VAR_PROB":[1],
    "scale_f": "7*x_mat[:,0]",
    "shape_f": "3",
    "cens_scale":7
	# "cens_scale":1.3
}

simple_1_2 = {
    "type": "Simple, 1 group, 50% cens",
    "x_vars": 1, 
    "VAR_CLASS": [2],
    "VAR_PROB":[1],
    "scale_f": "7*x_mat[:,0]",
    "shape_f": "3",
    # "cens_scale":7
	"cens_scale":1.3
}



# simple_2_1 = {
#     "type": "Simple, 2 group, 20% cens",
#     "x_vars": 1, 
#     "VAR_CLASS": [2],
#     "VAR_PROB":[.5],
#     "scale_f": "2.5 + 1.05*x_mat[:,0]",
#     "shape_f": ".8 + .5*x_mat[:,0]",
#     "cens_scale":3
# }

# simple_2_2 = {
#     "type": "Simple, 2 group, 50% cens",
#     "x_vars": 1, 
#     "VAR_CLASS": [2],
#     "VAR_PROB":[.5],
#     "scale_f": "2.5 + 1.05*x_mat[:,0]",
#     "shape_f": ".8 + .5*x_mat[:,0]",
#     "cens_scale":.7
# }

simple_2_1 = {
    "type": "Simple, 2 group, 20% cens",
    "x_vars": 1, 
    "VAR_CLASS": [2],
    "VAR_PROB":[.5],
    "scale_f": "12 - .5 *x_mat[:,0]",
    "shape_f": "1.2 + 1 *x_mat[:,0]",
    "cens_scale":5.9
	# "cens_scale":1.1
}
simple_2_2 = {
    "type": "Simple, 2 group, 50% cens",
    "x_vars": 1, 
    "VAR_CLASS": [2],
    "VAR_PROB":[.5],
    "scale_f": "12 - .5 *x_mat[:,0]",
    "shape_f": "1.2 + 1 *x_mat[:,0]",
    # "cens_scale":5.9
	"cens_scale":1.1
}