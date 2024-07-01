def get_case_cohort(y_sk, x_sk, prop = 1):
    if type(x_sk).__module__ != "numpy":
        print("x_sk needs to be a np array")
        return
    if prop == 1:
        weights = np.ones(y_sk.shape[0])
        return(y_sk, x_sk, weights)

    case_mask = (y_sk["Status"] == True)
    case_y = y_sk[case_mask]
    case_x = x_sk[case_mask]

    PRCNT = prop
    N = x_sk.shape[0]
    NSMP = int(N*PRCNT)
    sample_mask = np.random.choice(np.arange(0,x_sk.shape[0]), NSMP, replace=False)

    sub_x = x_sk[sample_mask,:]
    sub_y = y_sk[sample_mask]

    # join cohort
    coh_x = np.vstack([case_x, sub_x])
    coh_y = np.concatenate([case_y, sub_y])

    cntrl_msk = np.where(coh_y["Status"]==False)
    weight = np.ones(coh_y.shape)
    weight[cntrl_msk] = (1/PRCNT)

    return coh_y, coh_x, weight

