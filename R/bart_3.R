library(BART)
# get dat file name arg
print(getwd())

#read in csv
df = read.csv("../data/exp3_tmp.csv")
# df = read.csv("data/exp3_tmp.csv")

# df[,-c(1,2)]
x = as.matrix(df[,-c(1,2)]) # removes first two columns
# s.matrix(df[,-c(1,2)])
# x_test = unique(x)
# x_test = matrix(x_test[nrow(x_test):1,])
x_test = x
times <- df$t
delta <- df$s

# generate the pre df (only will be on times)
pre <- surv.pre.bart(times=times, delta=delta, x.train=x, x.test=x_test)##(, K=50)

post <- mc.surv.bart(x.train=pre$tx.train, y=pre$y.train,
                    x.test = pre$tx.test,
                     mc.cores=6, seed=99)
sv_mu = post$surv.test.mean
N = nrow(x)
T = post$K

sv_mu = t(matrix(sv_mu, ncol = N, nrow = T))

sv_cil = apply(post$surv.test, 2, quantile, probs=0.025)
sv_cil = t(matrix(sv_cil, ncol=N, nrow=T))
sv_cih = apply(post$surv.test, 2, quantile, probs=0.975)
sv_cih = t(matrix(sv_cih, ncol=N, nrow=T))

# n = length(sv_mu)/2
# x_out = c(rep(x_test[1],n), rep(x_test[2],n))
df_out_mu = data.frame(sv_mu)
# str(df_out_mu)
write.csv(df_out_mu, "../data/exp3_tmp_out3_mu.csv", row.names=F)
df_out_cil = data.frame(sv_cil)
# str(df_out_cil)
write.csv(df_out_cil, "../data/exp3_tmp_out3_cil.csv", row.names=F)
df_out_cih = data.frame(sv_cih)
# str(df_out_cih)
write.csv(df_out_cih, "../data/exp3_tmp_out3_cih.csv", row.names=F)
# write.csv(df_out, "data/exp2_tmp_out2.csv", row.names=F)

