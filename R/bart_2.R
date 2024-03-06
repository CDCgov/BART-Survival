library(BART)
# get dat file name arg
print(getwd())

#read in csv
df = read.csv("../data/exp2_tmp.csv")
# df = read.csv("data/exp2_tmp.csv")
x = matrix(df$x0)
x_test = unique(x)
x_test = matrix(x_test[nrow(x_test):1,])
times <- df$t
delta <- df$s

# generate the pre df (only will be on times)
pre <- surv.pre.bart(times=times, delta=delta, x.train=x, x.test=x_test)##(, K=50)

post <- mc.surv.bart(x.train=pre$tx.train, y=pre$y.train,
                    x.test = pre$tx.test,
                     mc.cores=6, seed=99)
sv_mu = post$surv.test.mean
sv_cil = apply(post$surv.test, 2, quantile, probs=0.025)
sv_cih = apply(post$surv.test, 2, quantile, probs=0.975)
n = length(sv_mu)/2
x_out = c(rep(x_test[1],n), rep(x_test[2],n))
df_out = data.frame(x_out, sv_mu, sv_cil, sv_cih)


write.csv(df_out, "../data/exp2_tmp_out2.csv", row.names=F)
# write.csv(df_out, "data/exp2_tmp_out2.csv", row.names=F)

