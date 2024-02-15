# Abstract Outline

1. Simulations
   1. Models = BART, rBART, COXPH(non-optimized), RSF
   2. Simple 1
      1. N=500
      2. Conditions
         1. Censor 0 - Proporitonal Hazard
         2. Censor 20 - Proporitonal Hazard
         3. Censor 80 - Proporitonal Hazard
         4. Censor 0 - Non-Proportional Hazard
         5. Censor 20 - Non-Proportional Hazard
         6. Censor 80 - Non-Proportional Hazard
      3. Metrics
         1. RMSE
         2. BIAS
         3. AUC-T
         4. Brier Score
         5. C-Index
   3. Simple 2
      1. N=2000
      2. Conditions
         1. Censor 0 - Proporitonal Hazard
         2. Censor 20 - Proporitonal Hazard
         3. Censor 80 - Proporitonal Hazard
         4. Censor 0 - Non-Proportional Hazard
         5. Censor 20 - Non-Proportional Hazard
         6. Censor 80 - Non-Proportional Hazard
      3. Metrics
         1. RMSE
         2. BIAS
         3. AUC-T
         4. Brier Score
         5. C-Index
   4. Complex 1
      1. N=500
      2. Conditions
         1. Censor 0 - Proporitonal Hazard
         2. Censor 20 - Proporitonal Hazard
         3. Censor 80 - Proporitonal Hazard
         4. Censor 0 - Non-Proportional Hazard
         5. Censor 20 - Non-Proportional Hazard
         6. Censor 80 - Non-Proportional Hazard
      3. Metrics
         1. RMSE
         2. BIAS
         3. AUC-T
         4. Brier Score
         5. C-Index
   5. Complex 2
      1. N=2000
      2. Conditions
         1. Censor 0 - Proporitonal Hazard
         2. Censor 20 - Proporitonal Hazard
         3. Censor 80 - Proporitonal Hazard
         4. Censor 0 - Non-Proportional Hazard
         5. Censor 20 - Non-Proportional Hazard
         6. Censor 80 - Non-Proportional Hazard
      3. Metrics
         1. RMSE
            1. SV 
            2. HR
         2. BIAS
            1. SV
            2. HR
         3. AUC-T
         4. Brier Score
         5. C-Index
2. Simulations 2 - focus on parameters
   1. Sample Conditions
      1. N = 1000
      2. Iter = 1000 (maybe 500)
   2. Parameter Conditions
      1. Tree
         1. 25, 50, 75, 100, 200
      2. Tune
         1. 200, 400, 800, 1600, 3200, 6400 12800 / per chain
         2. w/ draws 1000, w/ trees 50
      3. Draws
         1. 200, 400, 800, 1600, 3200, 6400 12800 / per chain
         2. w/ tune 1000, w/ trees 50
    3. Model Conditions
       1. Simple
          1. 1, 5, 20, 50 variables
    4. Metrics
       1. ESS
       2. RHAT
       3. RMSE
       4. BIAS
       5. Tree particulars evaluation 
       6. AUC-T
       7. Brier Score
       8. C-Index
       9. Credible Interval Length
       10. Coverage Probability
3.  Simulation 3 - focus on Treatment Effect
    1.  Sample Conditions
        1.  N = 1000
        2.  Iter = 500
    2.  Model Conditions
        1.  Complex
4.  Simulation 4 - focus on marginal sample size
    1.  Sample Conditions
        1.  N = 1000
        2.  iter = 500
    2.  Model Conditions
        1.  simple


1. Actual Data
   1. ... maybe covid
   2. lung
   3. python
   4. 


