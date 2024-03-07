library(BART)
# get dat file name arg
print(getwd())

#read in csv
df = read.csv("../data/exp1_tmp.csv")
times <- df$t
delta <- df$s

# generate the pre df (only will be on times)
pre <- surv.pre.bart(times=times, delta=delta)##(, K=50)
post <- mc.surv.bart(x.train=pre$tx.train, y=pre$y.train,
                    x.test = pre$tx.test,
                     mc.cores=6, seed=99)

# get sv
sv.mu = post$surv.test.mean
sv.cil = apply(post$surv.test, 2, quantile, probs=0.025)
sv.cih = apply(post$surv.test, 2, quantile, probs=0.975)

df_out = data.frame(sv.mu, sv.cil, sv.cih)
write.csv(df_out, "../data/exp1_tmp_out2.csv", row.names=F)

