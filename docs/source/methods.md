---
title: 'BART-Survival: A Bayesian machine learning approach to survival analyses in Python'
tags:
  - Python
  - Bayesian 
  - Machine Learning
  - Survival
  - Time to Event
authors:
  - name: Jacob Tiegs
    orcid: 0009-0001-6265-913X
    affiliation: "1, 2" 
  - name: Julia Raykin
    orcid: 0009-0006-1840-6991
    corresponding: true 
    affiliation: 1
  - name: Ilia Rochlin
    orcid: 0000-0001-7680-6965
    corresponding: true 
    affiliation: 1
affiliations:
 - name: Inform and Disseminate Division, Office of Public Health Data, Surveillance, and Technology, Centers for Disease Control and Prevention, Atlanta, GA, USA
   index: 1
 - name: Metas Solutions, Atlanta, Georgia
   index: 2

date: 08 August 2024
bibliography: paper.bib

---


# Methods

The following sections provides details on the methods employed by the `BART-Survival` library, focusing specifically on the discrete-time survival algorithm used. For review of the BART algorithm we refer to associated `PyMC-BART` publication [@quiroga2023].

## Background
The `BART-Survival` package provides a discrete-time survival method which aims to model survival as a function of a series of probabilities that indicate the probability of event occurrence at each discrete time. For clarity, the probability of event occurrence at each discrete time will be referred to as the risk probability.

In combination with a structural configuration of the data, the discrete-time algorithm allows for flexible modeling of the risk probabilities as a non-parametric function of time and observed covariates. The series of probability risks can then be used to derive the survival probabilities, along with other estimates of interest.


The foundation of the method is simple and is based off the well-defined Kaplan-Meier Product-Limit estimator. While a full review of the Kaplan-Meier method can be found elsewhere [@stel2011], the following example demonstrates the fundamental concepts of discrete-time survival analysis that motivates its application within the `BART-Survival` method.

1. Starting with a simple event-time dataset, create a sequence of time intervals that represent the unique discrete-time intervals observed in the data. Each interval is represented as a $t_j$, where $j = {1,...,k}$ and $k$ is the length of the set of unique observed times.



    For example if the observed event-time data is:

    $$
    \begin{matrix} 
    \text{event status} &\text{event time} \\
    --- &---\\
        1 & 1 \\
        0 & 2 \\
        1 & 2 \\
        1 & 4 \\
        1 & 4 \\
        1 & 5 \\
    \end{matrix}
    $$

    Then the set of unique observed times is $[1,2,4,5]$ with $k = 4$ indices and the corresponding $t_j$ intervals are:

    $$
    \begin{matrix} 
    \text{index} &\text{interval} \\
    --- &---\\
        t_1 & (0,1] \\
        t_2 & (1,2] \\
        t_3 & (2,4] \\
        t_4 & (4,5] \\
    \end{matrix}
    $$
    
    In the table above the intervals are denoted with the properties, "$($" indicating exclusive times and "$]$" indicating inclusive times in an interval. Additionally, the **event status** column is defined as $(1 = \text{event}, 0 = \text{censor})$


2. Then with the constructed time intervals $t_1,...,t_k$, tally the following for each interval:
    - the number of observations with an event
    - the number of observation censored 
    - the total number of observations eligible to have an event  (at risk) at the start of the interval

    Continuing the above example, the corresponding frequencies for each interval are:

    $$
    \begin{matrix} 
    \text{index} &\text{interval} &\text{event} &\text{censor} & \text{at risk}\\
    --- &---&---&---&---\\
        t_1 & (0,1] & 1  &   & 6 \\
        t_2 & (1,2] &  1 & 1 & 5 \\
        t_3 & (2,4] &  2 &   & 3 \\
        t_4 & (4,5] &  1 &   & 1 \\
    \end{matrix}
    $$


3. Finally, the risk of event occurrence within each interval $t_j$ can simply be derived as: 

    $$
    P_{t_j} = \frac {\text{n events}_{t_j}} {\text{n at risk}_{t_j}} \text{,}
    $$ 

    and the survival probability at a time-index $j$, can be derived as:

    $$
    S(t_j) = \prod_{q=1}^{j} (1-P_{t_q}) \text{ .}
    $$

    Applied to our example the risks of event $P_{t_j}$ are:

    $$
    \begin{matrix} 
    \text{index} &\text{interval} &\text{event} &\text{censor} & \text{at risk} & P_{t_j} \\
    --- &---&---&---&---&---    \\
    t_1 & (0,1] & 1  &   & 6 & 0.167 \\
    t_2 & (1,2] &  1 & 1 & 5 &  0.2 \\
    t_3 & (2,4] &  2 &   & 3 &  0.667 \\
    t_4 & (4,5] &  1 &   & 1 &  1.0 \\
    \end{matrix}
    $$

    And the corresponding survival estimates at times $[1,2,4,5]$ are:

    $$
    \begin{matrix} 
    \text{index} &\text{time} & \text{survival} \\
    ---&--- &---\\
    t_1  &  1 &  .83\\
    t_2  &  2 &  .67 \\
    t_3  &  4 &  .22\\
    t_4  &  5 &  0 \\
    \end{matrix}
    $$    

`BART-Survival` builds off this simple foundation by replacing each naively derived $P_{t_j}$ with a probability risk prediction, $p_{t_{j}|x_{ij}}$ yielded from the BART regression model. The $x_{ij}$ values are the associated covariate values for each observation $i$ at each discrete time $j$. The predicted values $p_{t_j|x_{ij}}$ can be further transformed into survival probability estimates or other estimands of interest. Formally, the survival probability estimate for a single observation $i$ at time-index $j$ can be derived as:

$$
S(t_j|x_i) = \prod_{q=1}^{j} (1-p_{t_q|x_{iq}}) \text{.}
$$ 


## Data Preparation  
As displayed in the above example, survival data is typically given as paired (*event status*, *event time*) outcomes, along with a set of covariates $x_i$ for each observation. In this setup, *event status* is typically a binary variable, with $1=$ **event** and  $0=$ **censored**.

Obtaining the $p_{t_j|x_{ij}}$ values and subsequent survival estimates is a two-step process. Each step of the process requires generating a specialized augmented datasets from the generic survival data. The first step in the process involves training the $\text{BART}$ model which requires the **training augmented dataset** (**TAD**). The second step involves generating the $p_{t_j|x_{ij}}$ predictions from the trained $\text{BART}$ model, which requires the **predictive augmented dataset** (**PAD**).

The **TAD** is created by transforming each single observation row into a series of rows representing the observation over a time series. When transformed a single observation's information is represented as a series of tuples, each tuple containing the augmented values of *status* and *time*, as well as replicates of the covariates $x$. The series of tuples is constructed using the $k$ distinct times from the generic dataset. These times are represented by $t$, which can be indexed by $j$, where $j$ is in the set $1,...,k$. Then for each observation's *event time*$_i$ there is an index $j$ in $1,...,k$ where $t_{j}$ $=$ *event time*$_i$ . This special index $j$ is denoted as $m_i$ and can be defined as:

$$ t_{m_i} = t_j = \text{event time}_i \text{.}$$

For each observation, a set of tuples is created containing a total of $m_i$ tuples which are used to represent that observation's information. Each tuple is created for a specific time $t_j$ where $j = 1,...,m_i$. Additionally, for all tuples with times $t_j < t_{m_i}$ the value of *status* is set to $0$ and for the time $t_j = t_{m_i}$ the value of *status* is set to the value *event status*$_i$. Covariate information is simply replicated across tuples.

For example if the survival data is:

$$
\left[
\begin{matrix} 
\text{observation} &\text{status} &\text{time} \\
--- &--- &---\\
    1& 1 & 4\\
    2& 0 & 6 \\
    3& 1 & 6 \\
    4& 1 & 7 \\
    5& 1 & 8 \\
    \pmb{6}& \pmb{0} & \pmb{8} \\
    \pmb{7} &\pmb{1} & \pmb{12} \\
    8 &1 & 14 \\
\end{matrix}
\right]
\space \space \text{with} \space \space
\begin{matrix}
\text{unique times (k=6)}\\
---\\
4\\6\\7\\8\\12\\14
\end{matrix}
$$

Then the transformation for observation $7$ would be represented in the *augmented* dataset as the sequence of rows:

$$
\begin{matrix} 
\text{observation} & \text{status} &\text{time} \\
---& --- &---\\
    7 & 0 & 4\\
    7 & 0 & 6 \\
    7 & 0 & 7 \\
    7 & 0 & 8 \\
    7 & 1 & 12 \\
\end{matrix}
$$

Similarly, the transformation for observation $6$ would be represented as:

$$
\begin{matrix} 
\text{observation} & \text{status} &\text{time} \\
---& --- &---\\
    6 & 0 & 4\\
    6 & 0 & 6 \\
    6 & 0 & 7 \\
    6 & 0 & 8 \\
\end{matrix}
$$

It is important to reiterate that for each observation $i$, the new set of rows created only include the time points $t_j \le t_{m_i}$. For observation $\pmb{7}$, the $t_{m_i} = 12$ and the corresponding times in the **TAD** are $4,6,7,8,12$. For observation $\pmb{6}$, the $t_{m_i} = 8$ and the corresponding times in the **TAD** are $4,6,7,8$. 

The utility of the **TAD** is that it unlinks *event time* from the *event status*. In this setup, the constructed *status* variable (which is a simple binary variable taking values $1$ or $0$) represents the outcome and *time* is a covariate. Then treating each row of the **TAD** as an independent observation, the outcome can be modeled as a binary regression of *status* over *time* and any additional covariates $x$. The trained regression model from this dataset can be used to yield probability predictions for each time $t_j$ for $j$ in $1,...,k$. These probability predictions hold the interpretation as being the probability risk of event occurrence at each time conditional on the event not having already occurred at a previous time. The predictions for each observation $i$ and time $t_j$ can be used to yield the various analytic targets, such as survival probabilities. 

To generate the predictions of probability risks over the observed times, the **PAD** dataset is used. The **PAD** transformation is similar to the **TAD**, but is simpler in that for each observation a set of size $k$ tuples is created. Each tuple in the set contains the time $t_j$ for $j=1,...,k$ and any additional covariates $x$.

Continuing the examples above, there are $k=6$ the unique times. For $j=1,...,k$ the resulting distinct times are $t_j = 4,6,7,8,12,14$ and the **PAD**s for observations $6$, and $7$ would be:

$$
\begin{matrix} 
\text{observation} & \text{time} \\
---& --- \\
    6 &  4\\
    6 &  6 \\
    6 &  7 \\
    6 &  8 \\
    6 &  12 \\
    6 &  14 \\
    \\
    7 &  4\\
    7 &  6 \\
    7 &  7 \\
    7 &  8 \\
    7 &  12 \\
    7 &  14 \\
\end{matrix}
$$

Notably the **TAD** and **PAD** lengths can differ. **PAD** length is simple to calculate. If $n$ is the number of observations in the generic dataset and $k$ is the number of the unique times observed in the generic dataset, then the length of the **PAD** is simply $n * k$. In the example data above it would be $(8*6)=48$.

The **TAD** length can be calculated using each observation's $m_i$, which again is defined as being the index that resolves $t_{m_i}=$ *event time*$_i$. The **TAD** length can be calculated as:

$$ \pmb{TAD}_{\text{length}} = \sum_{i=1}^{n}{m_i} \text{.}$$

This can be demonstrated with the example dataset, where the **TAD** length can be found as the sum of the $m_i$ column which resolves to a **TAD** length of $27$:

$$
    \begin{matrix}
    \text{observation} & \text{event time}_i & m_i\\
    ---&---&---\\
    1 & 4 & 1\\
    2 & 6 & 2\\
    3 & 6 & 2\\
    4 & 7 & 3\\
    5 & 8 & 4\\
    6 & 8 & 4\\
    7 & 12 & 5\\
    8 & 14 & 6\\
    &&---\\
    &&27
    \end{matrix} 
$$



While the **PAD** and **TAD** lengths are not important for generating the models, they are helpful to keep in mind when completing an analysis. Specifically, the fact that the **TAD** and **PAD** can be far larger in length than the generic dataset. For example the generic dataset used above is only $8$ rows in length, but the **TAD** and **PAD** dataset are $27$ and $48$ rows respectively. 

When the length of the two datasets become large enough to make computation difficult it is recommended to downscale the *event time* values, which will reduce the length of the two augmented datasets. The downscaling algorithm we provide in the library takes the *event time*, divides by a scaling factor and then takes the "ceiling" truncation to create the new *scaled event time*. Below the downscaling algorithm is demonstrated with the example dataset using a scaling factor of $4$:

$$
    \begin{matrix}
    \text{ time} & \text{downscaled time} & \text{rescaled time} \\ 
    ---&---&---\\
    4 & 1 & 4\\
    6 & 2 & 8\\
    6 & 2 & 8\\
    7 & 2 & 8\\
    8 & 2 & 8\\
    8 & 2 & 8\\
    12 & 3 & 12\\
    14 & 4 & 16\\
    \end{matrix} 
$$

Downscaling the *event time* values causes loss of granularity in the set of discrete times analyzed. As shown in the example above, after downscaling (and then rescaling) the analytic targets can only be derived for the times $4,8,12$ and $16$, which is reduced from the original set of times, $4,6,7,8,12,14$. While the analytic targets cannot be returned for the times $6,7,14$, the training information contributed by the observations with these *event times* is maintained through subsequent contribution at the downscaled times. Since there is no loss of event information, the probability of event at the downscaled times $4,8,12,16$ will be equal to the probability of event from a non-downscaled dataset at the select time $4,8,12,16$.  

Downscaling can be safely applied to datasets without loss of precision of estimates at the downscaled times. The only consideration required when using a downscaling procedure is to ensure that the downscaled times granularity fulfills the needs of the analysis. For example, in a multi-year health-outcomes study, downscaling from days to months could significantly reduce computational burden without a meaningful loss of information in the outcome. However, if that same study reduced time from days to years, this could lead to loss of meaningful information in the outcome. The considerations for downscaling need to be made on a study-to-study basis and no general recommendation on selecting a downscaling factor can be provided.

## Model

The `BART-Survival` algorithm is two-step algorithm. First, the **TAD** is used to train a non-parametric regression model of *status* on *time* and covariates. Then the **PAD** is used to yield predicted probabilities at each discrete time. These probability of event predictions can be used generate all additional estimates useful for statistical inference. 

To motivate the use of the **TAD** in training the regression model, an example **TAD** is displayed below. Here, *status* can be represented as the outcome vector of $y_{ij}$ values and *time* is represented by the $t_{ij}$ values. Additional covariates are represented by a vector of $x_{ij}$ values. The subscript $i$ refers to the $i^{th}$ observation of the generic dataset and the subscript $j$ refers to the $j^{th}$ index of the unique set of discrete times previously defined as $t$. In the **TAD** and **PAD**, the $i$, $j$ indices can be thought of as multi-indices over the rows. 

$$
    \begin{matrix}
    i & j & y & t & x \\ 
    -&-&-&- &- &\\
    1 & 1 & 0 & 3 & x_{11} \\
    1 & 2 & 0 & 4 & x_{12} \\
    1 & 3 & 1 & 6& x_{13}  \\
    2 & 1 & 0 & 3 & x_{21} \\
    2 & 2 & 1 & 4 & x_{22} \\
    3 & 1 & 0 & 3 & x_{31} \\
    3 & 2 & 0 & 4 & x_{32} \\
    3 & 3 & 1 & 6 & x_{33} \\
    \end{matrix} 
$$

When using the **TAD**, each $y_{ij}$ value can then be treated as independent draw of a $Bernoulli$ distribution parameterized with the probability of event $p_{ij}$ for observation $i$ at time index $j$. The $p_{ij}$ values are collected as the latent values yielded from a $probit$ regression of $y$ on $t$ and $x$. Formally the model is defined as:

$$
\begin{aligned}
    y_{ij} | p_{ij} \sim Bernoulli(p_{ij}) \text{,}\\
    p_{ij} | \mu_{ij} = \Phi(\mu_{ij}) \text{,}\\
    \mu_{ij} \sim \text{BART}(t_{ij},x_{ij}) \text{,}\\
\end{aligned}
$$

where $\Phi$ is the normal cumulative distribution function and $\text{BART}$ is the ensemble of regression trees that yield the $\mu_{ij}$ value for a given $t_{ij}$, $x_{ij}$ combination.

Regarding the $\text{BART}$ algorithm, in brief $\text{BART}$ is a Bayesian approach to the ensemble regression trees class of non-parametric models. $\text{BART}$ uses a combination of Bayesian priors placed on the tree generating components and a Markov Chain Monte Carlo sampling algorithm to iteratively generate tree ensembles. These tree ensemble are collected as samples of the posterior distribution. The samples of the posterior distribution are subsequently used to generate distinct posterior predictive distributions for a given set of observations.

This implies that for each $t_{ij}$, $x_{ij}$ combination, the algorithm first returns a distribution of $p_{ij}$ values, denoted as $p_{ij_{dist}}$. Each value in the $p_{ij_{dist}}$ is a prediction mapped from a single tree ensemble of the posterior distribution. Point estimates and uncertainty intervals can be easily obtained from the $p_{ij_{dist}}$ as simple estimates from that distribution. For example the mean can be derived as the empirical average of $p_{ij_{dist}}$. Similarly, the even-tailed $95\%$ credible interval can be derived as the $5^{th}$ and $95^{th}$ percentiles of the $p_{ij_{dist}}$. Of note, any reference to $p_{ij}$ used in this article is referring to the mean point estimate of the $p_{ij_{dist}}$.
 
Further details regarding the $\text{BART}$ implementation used in the `BART-Survival` algorithm can be found in `PyMC-BART` repository and the accompanying publication [@quiroga2023].

## Survival 
A trained `BART-Survival` model is used along with the **PAD** to generate the vector of $p_{ij}$ predictions. To generate survival estimates the $p_{ij}$ vector is grouped by the $i$ indices. Then using the following equation the survival probability at discrete times can be constructed:

$$
S(t_{q}| x_{iq}) = \prod_{j=1}^{q} (1-p_{ij}) \text{,}
$$ 

where $q$ is the index of the time for the desired estimate. As described in the previous section a full posterior predictive distribution is returned for each $S_{t_q,x_{iq}}$. Point estimates and credible intervals can be obtained through the empirical mean and percentile functions of the returned posterior predictive distribution for each $S(t_q|x_{iq})$.

## Marginal Effects

`BART-Survival` provides capabilities to evaluate covariate effects through use of partial dependence functions which yield marginal effect estimates. The partial dependence function method involves generating predictions of $p$ for the observations in a **partial dependence augmented dataset** (**PDAD**). Using variations of the **PDAD** yields different sets of predictions $p$. These sets can then be contrasted to yield marginal effect estimates, including marginal Hazard Ratios which have similar interpretation as the Cox Proportional Hazard Model's conditional Hazard Ratios.

**PDAD**s can be created through further augmentation of the previously described **PAD**. As a reminder, the **PAD** contains the generated time covariate $t$ and $k$ replicates of each $x_i$ from the generic dataset, where $k$ is the length of the uniquely observed *event times*. Importantly $x_{i}$ can be defined generically as the set of covariates $\{x_{1_i}, x_{2_i}, .., x_{m_i}\}$, where $m$ is the index of the last included covariate.

Then, starting with the **PAD**, the **PDAD** is generated through selection of a specific covariate $x_{[I]}$, from the set of covariates $\{x_{1}, x_{2}, ..., x_{m}\}$ and setting that covariate to a deterministic value for all observation $i$ and evaluated times $j$. The bracketed notation ${[I]}$ is used to define the index of the covariate selected and convey that the selected covariate is set to a deterministic value. Expanding the notation, $x_{[I] = v}$ indicates that covariate of index $I$ is selected and deterministically set to value $v$.

In the following example the second covariate from a set of covariates $\{x_{1}, x_{2}, x_{3}\}$ is of interest for evaluating marginal effects. This covariate is binary and takes the values ${0,1}$. Evaluation of this covariate can be completed by creating **PDAD**s for possible values of the selected covariate. In each **PDAD** all of the observations and times are set to a deterministic values selected from the range of the covariate. The two generated **PDAD**s can be defined by the covariate index and set values: $x_{[2] = 0}$ and $x_{[2]=1}$.



$$
\text{PAD:} \space\space\space\space
\begin{matrix}
i & j & t & x_1 & x_2 & x_3 \\ 
-&-&-&-&-&-&\\
1 & 1 & 2 & 1.2 & 0 & 10\\
1 & 2 & 3 & 1.2 & 0 & 10\\
1 & 3 & 5 & 1.2 & 0 & 10\\
2 & 1 & 2 & 2.4 & 1 & 12\\
2 & 2 & 3 & 2.4 & 1 & 12\\
2 & 3 & 5 & 2.4 & 1 & 12\\
3 & 1 & 2 & 1.9 & 0 & 3\\
3 & 2 & 3 & 1.9 & 0 & 3\\
3 & 3 & 5 & 1.9 & 0 & 3\\
\end{matrix} 
$$

$$
    \text{PDAD}_{x_{[2]=1}} \space\space\space\space
    \begin{matrix}
    i & j & t & x_1 & \pmb{x_2} & x_3 \\ 
    -&-&-&-&-&-&\\
    1 & 1 & 2 & 1.2 & \pmb{1} & 10\\
    1 & 2 & 3 & 1.2 & \pmb{1} & 10\\
    1 & 3 & 5 & 1.2 & \pmb{1} & 10\\
    2 & 1 & 2 & 2.4 & \pmb{1} & 12\\
    2 & 2 & 3 & 2.4 & \pmb{1} & 12\\
    2 & 3 & 5 & 2.4 & \pmb{1} & 12\\
    3 & 1 & 2 & 1.9 & \pmb{1} & 3\\
    3 & 2 & 3 & 1.9 & \pmb{1} & 3\\
    3 & 3 & 5 & 1.9 & \pmb{1} & 3\\
    \end{matrix} 
$$
$$
    \text{PDAD}_{x_{[2]=0}} \space\space\space\space
    \begin{matrix}
    i & j & t & x_1 & \pmb{x_2} & x_3 \\ 
    -&-&-&-&-&-&\\
    1 & 1 & 2 & 1.2 & \pmb{0} & 10\\
    1 & 2 & 3 & 1.2 & \pmb{0} & 10\\
    1 & 3 & 5 & 1.2 & \pmb{0} & 10\\
    2 & 1 & 2 & 2.4 & \pmb{0} & 12\\
    2 & 2 & 3 & 2.4 & \pmb{0} & 12\\
    2 & 3 & 5 & 2.4 & \pmb{0} & 12\\
    3 & 1 & 2 & 1.9 & \pmb{0} & 3\\
    3 & 2 & 3 & 1.9 & \pmb{0} & 3\\
    3 & 3 & 5 & 1.9 & \pmb{0} & 3\\
    \end{matrix} 
$$

From the two **PDAD**s the predicted probabilities, $p_{x_{[2]=1}}$, $p_{x_{[2]=0}}$ and survival probabilities $S_{x_{[2]=1}}$, $S_{x_{[2]=0}}$, can be generated:

$$
    \begin{matrix}
    i & j & p_{x_{[2]=1}} & S_{x_{[2]=1}} & p_{x_{[2]=0}} & S_{x_{[2]=0}}\\
    -&-&-&-&-&-\\
    1 & 1 & .20 & .80 &.10 & .90\\
    1 & 2 & .25 & .60 &.15 & .77\\
    1 & 3 & .18 & .49 &.08 & .70\\
    2 & 1 & .10 & .90 &.08 & .92 \\
    2 & 2 & .13 & .78 &.11 & .81\\
    2 & 3 & .12 & .69 &.10 & .74\\
    3 & 1 & .20 & .80 &.15 & .92\\
    3 & 2 & .28 & .58 &.23 & .82\\
    3 & 3 & .23 & .44 &.18 & .74\\
    \end{matrix}
$$

The marginal expectations of $p_{x_{[I]}}$ and $S_{x_{[I]}}$ at a specific time can be further derived by taking the average of the estimates over observations $i$ for the specified time $t$ indexed by $j$: 

$$
E_{i}[p_{x_{[I]}}|t_j] = {\frac 1 n} \sum_{i=1}^n {p_{ij,x_{[I]}}} \text{,}
$$
$$
E_{i}[S_{x_{[I]}}(t_j)] = {\frac 1 n} \sum_{i=1}^n {S_{i,x_{[I]}}(t_j)} \text{,}
$$

where $E_{i}$ is the expectation over $i$,...,$n$ observations. From the above example, this yields the expectations for time indices $j=$ $1$,$2$,$3$.

$$
    \begin{matrix}
    j & t & E_i[p_{x_2=1}] & E_i[S_{x_2=1}] & E_i[p_{x_2=0}] & E_i[S_{x_2=0}]\\
    -&-&-&-&-&-\\
    1 & 2 & 0.17 & 0.83 & 0.11 & 0.91 \\ 
    2 & 3 & 0.22 & 0.65 & 0.16 & 0.8 \\
    3 & 5 & 0.18 & 0.54 & 0.12 & 0.73\\
    \end{matrix}
$$

These expectations can be further used to make comparisons between the evaluation of various values of $x_{[I]}$ across multiple **PDAD**s. Common marginal effect estimates derived from these predicted values include:

- Marginal difference is survival probability at time $t$:


$$
\text{Surv. Diff.}_{marg}(t_j) = E_{i}[S_{x_{[I] = v_2}}(t_j)] - E_{i}[S_{x_{[I] = v_1}}(t_j)] \text{.}
$$

- Marginal Risk Ratio at time $t$:

$$
\text{RR}_{marg}(t_j) = \frac {E_{i}[p_{j,x_{[I] = v_{2}}}]} {E_{i}[p_{j,x_{[I] = v_1}}]} \text{.}
$$

- Marginal Hazard Ratio (expectation over $i$ and times up to $t$):

$$
\text{HR}_{marg}(t_{j}) = \frac {E_{ij}[p_{x_{[I]=v_2}}]} {E_{ij}[p_{x_{[I] = v_1}}]} \text{.}
$$

Continuing the example, the marginal effect of $x_2$ as measured by difference in survival probability when $x_{[2]} = 1$ and $x_{[2]} =0$ at the times $2,3,5$ can be examined below: 

$$
    \begin{matrix}
    j & t & \text{Surv.Diff.}\\
    -&-&-&\\
    1 & 2 &-0.08 \\
    2 & 3 &-0.15 \\
    3 & 5 &-0.19 \\
    \end{matrix}
$$

These results can be conveniently interpreted. For example, at time $t=5$ the increase of $x_2$ from $0$ to $1$ leads to an average change in the survival probability of $-0.19$, where survival probability is in the range $0$-$1$. Additionally, as mentioned in the previous sections, all estimates of the model will first be yielded as posterior predictive distributions. The empirical mean and percentile function of this distribution yield the point estimates described above and their respective credible intervals.
