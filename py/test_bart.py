import pymc as pm
import pymc_bart as pmb
import numpy as np

def main():
	coal = np.loadtxt(pm.get_data("coal.csv"))
	# discretize data
	years = int(coal.max() - coal.min())
	bins = years // 4
	hist, x_edges = np.histogram(coal, bins=bins)
	# compute the location of the centers of the discretized data
	x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
	# xdata needs to be 2D for BART
	x_data = x_centers[:, None]
	# express data as the rate number of disaster per year
	y_data = hist
	with pm.Model() as model_coal:
		μ_ = pmb.BART("μ_", X=x_data, Y=np.log(y_data), m=20)
		μ = pm.Deterministic("μ", pm.math.exp(μ_))
		y_pred = pm.Poisson("y_pred", mu=μ, observed=y_data)
		idata_coal = pm.sample(random_seed=99)

if __name__ == "__main__":
	main()