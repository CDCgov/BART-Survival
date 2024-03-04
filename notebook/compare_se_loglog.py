# t = get_sim(rng, N[0], **simple_2_1)
kpm = ll.KaplanMeierFitter()
kpm.fit(durations=t[2]["t_event"][t[1]==1], event_observed=t[2]["status"][t[1]==1])
# t[1]==1
k1 = kpm.survival_function_

var_1 = kpm._cumulative_sq_.values[:,None]
# se_1 = np.sqrt(var/np.power(np.log(k1),2))


k1_ci = kpm.confidence_interval_
kpm.fit(durations=t[2]["t_event"][t[1]==0], event_observed=t[2]["status"][t[1]==0])
k2 = kpm.survival_function_
k2_ci = kpm.confidence_interval_
var_2 = kpm._cumulative_sq_.values[:,None]
# se_2 = np.sqrt(var/np.power(np.log(k2),2))


t_max = t[2]["t_event"].max()
kpm.survival_function_at_times(np.arange(t_max))

print(k1)
print(k2)

pd.DataFrame([t[1].reshape(-1), t[2]["status"].reshape(-1), t[2]["t_event"].reshape(-1)]).T.to_csv("../data/sv_time1.csv")

print(np.exp(-np.exp(np.log(-np.log(k1.values)) - 1.96 * np.sqrt(var_1)/ np.log(k1.values))))
np.sqrt(var_1)

# print(var_1)
# print(var_2)
print(np.sqrt(var_2)/np.log(k2.values))
print(np.sqrt(var_1)/np.log(k1.values))
print(np.sqrt(var_2))
# print(se_1)
# print(se_2)