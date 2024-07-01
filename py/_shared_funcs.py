def save_to_csv(event_dict, x_mat, file="exp1_tmp.csv"):
    """helper function that passes the simulated event dataset to a csv
    - Used for R-bart

    Args:
        event_dict (_type_): _description_
        x_mat (_type_): _description_
        file (str, optional): _description_. Defaults to "exp1_tmp.csv".
    """
    col = ["t","s"] + ["x"+str(i) for i in range(x_mat.shape[1])]
    df = pd.DataFrame(np.hstack(
        [
            event_dict["t_event"],
            event_dict["status"],
            x_mat
        ]
    ),columns= col)
    exp_name = file
    path = f"../data/{exp_name}"
    with open(path, 'w') as f:
        df.to_csv(f, index=False)
    # df.to_csv(path,index = False)
    

def get_sim(rng, N, type, x_vars, VAR_CLASS, VAR_PROB, scale_f, shape_f, cens_scale):
    """Generates simulation dataest

    Args:
        rng (_type_): if int, generate new rng seeded as int value.
        N (_type_): _description_
        type (_type_): _description_
        x_vars (_type_): _description_
        VAR_CLASS (_type_): _description_
        VAR_PROB (_type_): _description_
        scale_f (_type_): _description_
        shape_f (_type_): _description_
        cens_scale (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_mat = sm.get_x_matrix(
        N = N,
        x_vars = x_vars, 
        VAR_CLASS=VAR_CLASS,
        VAR_PROB= VAR_PROB,
        rng = rng
    )

    event_dict, sv_true, sv_scale_true = sm.simulate_survival(
        x_mat = x_mat,
        scale_f=scale_f,
        shape_f=shape_f,
        cens_scale=cens_scale,
        rng = rng
    )
    return type, x_mat, event_dict, sv_true, sv_scale_true


def get_quant_events(qnt_t, event):
	q = np.array(qnt_t)
	et_ = event["t_event"].copy()
	et_out = event["t_event"].copy()
	es_out = event["status"].copy()
	for i in range(q.shape[0]):
		if i == 0:
			msk = et_<=q[i]
			et_out[msk] = q[i]
			# print(np.unique(et_out))
		else:
			msk = (q[i-1] < et_) & (et_ <= q[i])
			et_out[msk] = q[i]
			# print(np.unique(et_out))
			if i == q.shape[0]-1:
				msk = et_ > q[i]
				et_out[msk] = q[i]
				es_out[msk] = 0

	return {"t_event":et_out, "status":es_out}

