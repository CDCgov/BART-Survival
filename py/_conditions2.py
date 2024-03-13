#simple
# N = [200,400]

complex_1 = {
    "type": "Complex, binary, prop, 20% cens",
    "x_vars": 9, 
    "VAR_CLASS": [2,2,2,2,2,2,2,2,2],
    "VAR_PROB":[.5,.5,.5,.5,.5,.5,.5,.5,.5],
    # "scale_f": "2.5*x_mat[:,0]",
    "scale_f": "np.exp(3 + 0.1*(x_mat[:,0] + x_mat[:,1] + x_mat[:,2] + x_mat[:,3] + x_mat[:,4] + x_mat[:,5]) + x_mat[:,6])",
    "shape_f": "2",
    "cens_scale":None
}

complex_2 = {
    "type": "Complex, binary, nonprop, 20% cens",
    "x_vars": 9, 
    "VAR_CLASS": [2,2,2,2,2,2,2,2,2],
    "VAR_PROB":[.5,.5,.5,.5,.5,.5,.5,.5,.5],
    "scale_f": "20 + 5*(x_mat[:,0] + x_mat[:,1] + x_mat[:,2] + x_mat[:,3] + x_mat[:,4] + x_mat[:,5] + 10*x_mat[:,6])",
    "shape_f": "0.7 + 1.3*x_mat[:,6]",
    "cens_scale":3.9
}

complex_3 = {
    "type": "Complex, continuous, 20% cens",
    "x_vars": 10, 
    "VAR_CLASS": [1,1,1,1,1,1,1,1,1,1],
    "VAR_PROB":[None,None,None,None,None,None,None,None,None,None],
    "scale_f" : "np.exp(3 + 0.5*np.sin(np.pi * x_mat[:,0] * x_mat[:,1]) + np.power((x_mat[:,2]-0.5), 2) + 0.5* x_mat[:,3] + 0.25 * x_mat[:,4])",
    # "scale_f": "20 + 5*(x_mat[:,0] + x_mat[:,1] + x_mat[:,2] + x_mat[:,3] + x_mat[:,4] + x_mat[:,5] + 10*x_mat[:,6])",
    "shape_f": "2",
    "cens_scale":3.7
}
