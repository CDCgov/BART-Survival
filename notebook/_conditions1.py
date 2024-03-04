#simple
N = [100, 200,400]

simple_1_1 = {
    "type": "Simple, 1 group, 50% cens",
    "x_vars": 1, 
    "VAR_CLASS": [2],
    "VAR_PROB":[1],
    "scale_f": "2.5*x_mat[:,0]",
    "shape_f": ".8",
    "cens_scale":3.3,
}
simple_1_2 = {
    "type": "Simple, 1 group, 50% cens",
    "x_vars": 1, 
    "VAR_CLASS": [2],
    "VAR_PROB":[1],
    "scale_f": "2.5*x_mat[:,0]",
    "shape_f": ".8",
    "cens_scale":.5,
}

simple_2_1 = {
    "type": "Simple, 2 group, 50% cens",
    "x_vars": 1, 
    "VAR_CLASS": [2],
    "VAR_PROB":[.5],
    "scale_f": "2.5 + 1.05*x_mat[:,0]",
    "shape_f": ".8 + .5*x_mat[:,0]",
    "cens_scale":3
}

simple_2_2 = {
    "type": "Simple, 2 group, 50% cens",
    "x_vars": 1, 
    "VAR_CLASS": [2],
    "VAR_PROB":[.5],
    "scale_f": "2.5 + 1.05*x_mat[:,0]",
    "shape_f": ".8 + .5*x_mat[:,0]",
    "cens_scale":.7
}
