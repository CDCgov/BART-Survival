install.packages("survival")
library(survival)

getwd()
sv = read.csv("data/sv_time1.csv")
sv

km <- survfit(Surv(X2, X1) ~ X0, data = sv, conf.type = "log-log")

km
summ = summary(km)
summ
summ$std.err ^ 2

km$std.err
se = km$std.err/log(km$surv)
se

exp(-exp(log(-log(km$surv)) - se * qnorm(.975)))

s4$surv[1]

?survfit.formula

km$time
km$std.err
