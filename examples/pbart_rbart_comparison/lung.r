# #################################
# This example is pulled from the suv.bart documentation from the R-based BART package.
# Example is repeated using our python-based BART-Survival package.
# #################################
install.packages("BART")
library(BART)

## load survival package for the advanced lung cancer example
data(lung)
N <- length(lung$status)

table(lung$ph.karno, lung$pat.karno)

## if physician's KPS unavailable, then use the patient's
h <- which(is.na(lung$ph.karno))
lung$ph.karno[h] <- lung$pat.karno[h]

times <- lung$time
delta <- lung$status-1 ##lung$status: 1=censored, 2=dead
##delta: 0=censored, 1=dead

## this study reports time in days rather than weeks or months
## coarsening from days to weeks or months will reduce the computational burden
times <- ceiling(times/30) # months
# times <- ceiling(times/7)  ## weeks

table(times)
table(delta)

## matrix of observed covariates
x.train <- cbind(lung$sex, lung$age, lung$ph.karno)




## lung$sex:        Male=1 Female=2
## lung$age:        Age in years
## lung$ph.karno:   Karnofsky performance score (dead=0:normal=100:by=10)
##                  rated by physician

dimnames(x.train)[[2]] <- c('M(1):F(2)', 'age(39:82)', 'ph.karno(50:100:10)')

## ***************  save as csv for comparable python implmenetation
out_train = data.frame(x.train)

out_train["times"] = times
out_train["delta"] = delta
str(out_train)


write.csv(out_train, "lung.csv")
## 

table(x.train[ , 1])
summary(x.train[ , 2])
table(x.train[ , 3])

##test BART with token run to ensure installation works
# set.seed(99)
# post <- surv.bart(x.train=x.train, times=times, delta=delta,
#                   nskip=1, ndpost=1, keepevery=1)

## Not run: 
## run one long MCMC chain in one process
## set.seed(99)
## post <- surv.bart(x.train=x.train, times=times, delta=delta, x.test=x.test)

## in the interest of time, consider speeding it up by parallel processing
## run "mc.cores" number of shorter MCMC chains in parallel processes
post <- mc.surv.bart(x.train=x.train, times=times, delta=delta,
                     mc.cores=8, seed=99)

# used to create the partial dependence dataset
pre <- surv.pre.bart(times=times, delta=delta, x.train=x.train,
                     x.test=x.train)

K <- pre$K
M <- nrow(post$yhat.train)
str(post)

# essential creating two copies of the PAD used to create the PDAD
pre$tx.test <- rbind(pre$tx.test, pre$tx.test)
pre$tx.test[ , 2] <- c(rep(1, N*K), rep(2, N*K)) # partial dependence of sex = 1(M) and sex=2(F)
## sex pushed to col 2, since time is always in col 1

pred <- predict(post, newdata=pre$tx.test, mc.cores=8)




M <- post$ndpost # set M as number of draws

# get the marginal mean for sex =1 and sex=2
pd <- matrix(nrow=M, ncol=2*K)
for(j in 1:K) { # for times 1,..,31
    h <- seq(j, N*K, by=K) # seq (1,32,63,...)+J-1
    ps <- apply(pred$surv.test[ , h], 1, mean)
    # print(str(ps))
    pd[ , j] <- apply(pred$surv.test[ , h], 1, mean)
    pd[ , j+K] <- apply(pred$surv.test[ , h+N*K], 1, mean)
}
str(pd)


pd.mu  <- apply(pd, 2, mean)
pd.025 <- apply(pd, 2, quantile, probs=0.025)
pd.975 <- apply(pd, 2, quantile, probs=0.975)

males <- 1:K
females <- males+K


plot(c(0, pre$times), c(1, pd.mu[males]), type='s', col='blue',
     ylim=0:1, ylab='S(t, x)', xlab='t (months)',
     main=paste('Advanced Lung Cancer ex. (BART::lung)',
                "Friedman's partial dependence function",
                'Male (blue) vs. Female (red)', sep='\n'))
lines(c(0, pre$times), c(1, pd.025[males]), col='blue', type='s', lty=2)
lines(c(0, pre$times), c(1, pd.975[males]), col='blue', type='s', lty=2)
lines(c(0, pre$times), c(1, pd.mu[females]), col='red', type='s')
lines(c(0, pre$times), c(1, pd.025[females]), col='red', type='s', lty=2)
lines(c(0, pre$times), c(1, pd.975[females]), col='red', type='s', lty=2)

# save the pd to share with python
str(pd)
f1 = data.frame(pd[,females])
names(f1) = gsub("X", "month", names(f1))
f1["sex"] = "female"
str(f1)

m1 = data.frame(pd[,males])
names(m1) = gsub("X", "month", names(m1))
m1["sex"] = "male"
str(f1)
f1[1:10,1:4]
m1[1:10,1:4]

out = (rbind(f1, m1))

write.csv(out, "r_lung_pdp_sex.csv")
# , pd[,males]))


