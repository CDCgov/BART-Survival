
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
    corresponding: true 
    affiliation: 1
  - name: Ilia Rochlin
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

# Summary

`BART-Survival` is a Python package that allows time-to-event (survival) analyses in discrete-time using the non-parametric machine learning algorithm, Bayesian Additive Regression Trees (BART). `BART-Survival` combines the performance of the BART algorithm with the complementary data and model structural formatting required to complete the survival analyses. The library contains a convenient application programming interface (API) that allows a simple approach when using the library for survival analyses, while maintaining capabilities for added complexity when desired. The package is intended for analysts exploring use of flexible non-parametric alternatives to traditional (semi-)parametric survival analyses. 

# Statement of need

Survival analyses are a cornerstone of public health and clinical research in such diverse fields as cancer, cardiovascular disease, and infectious diseases [@altman1998; @bradburn2003]. Traditional parametric and semi-parametric statistical methods, such as the Cox proportional hazards model, are commonly employed for survival analyses [@cox1972]. However, these methods have several limitations, particularly when applied to complex data. One major issue is the need for restrictive assumptions, such as proportional hazards and predefined functional forms, which may not hold true in complex, real-world healthcare data [@ishwaran2008; @harrell2015]. Additionally, these methods often struggle with high-dimensional datasets, leading to problems with overfitting, multicollinearity, and dealing with complex interactions [@ishwaran2008; @joffe2013].

More recently, non-parametric machine learning approaches have been introduced to address these limitations by reducing the need for restrictive assumptions and providing increased capabilities for more accurately modeling underlying distributions and complex interactions [@ishwaran2008; @harrell2015]. BART is one such machine learning method that has demonstrated utility in the survival setting through its performance in identifying underlying statistical distributions [@chipman2010; @sparapani2021]. BART offers flexibility in modeling complex relationships and interactions within the data without requiring the specification of a particular functional form [@sparapani2016]. 

Currently, the only BART survival algorithm readily available exists as part of the `BART` R package, which contains a library of various BART-based approaches in addition to a BART survival analysis application [@sparapani2016; @sparapani2021]. BART-Survival package described here combines the survival analysis approach outlined in the `BART` R package with the foundational Python-based probabilistic programming language library, `PyMC` [@abril-pla2023], and the accompanying BART algorithm from the `PyMC-BART` library [@quiroga2023].
Our aim in developing `BART-Survival` is to provide accessibility to the BART survival algorithm within the Python programing language. This contribution is beneficial for analysts when Python is the preferred programming language, the analytic workflow is Python-based, or when the R language is unavailable for analyses. Additionally, `BART-Survival` package abstracts away the complexities of the `PyMC` and `PyMC-BART` libraries through use of a pre-specified core model and generalized functionality that can accommodate analyses across various survival settings. The  `BART-Survival` package is intended for public health and clinical professionals and students who are looking for non-parametric alternatives to traditional (semi-)parametric survival analysis, especially for use in large, complex healthcare data and machine learning applications. 

# Methods

The following sections provides details on the methods employed by the BART-Survival library, focusing specifically on the discrete-time Survival algorithm used. For review of the BART algorithm we refer to associated PyMC-BART publication [@quiroga2023].

## Background
The `BART-Survival` package provides a discrete-time Survival method which aims to model Survival as a function of a series of probabilities (indicating risk of event) that can be determined from the sequence of discrete-time intervals examined. 
In combination with a structural configuration of the data, the discrete-time algorithm allows for flexible modeling of the risk probabilities as a non-parametric function of  time and observed covariates. The series of probability risks can then be used in deriving Survival probabilities, along with other estimates of interest.


The foundation of the method is simple.  



1. Starting with a simple event-time dataset, create a sequence of time intervals that represent the unique discrete-time intervals observed in the data. Each intervals are represented as a $t_j$, where $j = {1,...,k}$ and $k$ is the length of set of unique observed times.



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
        t_1 & [0,1) \\
        t_2 & [1,2) \\
        t_3 & [2,4) \\
        t_4 & [4,5) \\
    \end{matrix}
    $$
    
    of note:
    - "$[$" is exclusive and "$)$" is inclusive times in the interval
    - for **event status**: $(1 = \text{event}, 0 = \text{censor})$


2. For each interval tally the number of observations with an event, the number of observation censored and the total number of observations eligible to have an event at the start of each interval $t_j$.

    Continuing the above example the corresponding frequencies for each interval are:

    $$
    \begin{matrix} 
    \text{index} &\text{interval} &\text{event} &\text{censor} & \text{at risk}\\
    --- &---&---&---&---\\
        t_1 & [0,1) & 1  &   & 6 \\
        t_2 & [1,2) &  1 & 1 & 5 \\
        t_3 & [2,4) &  2 &   & 3 \\
        t_4 & [4,5) &  1 &   & 1 \\
    \end{matrix}
    $$


3. Finally, the risk of event occurrence within each interval $t_j$ can naively be derived as: 

    $$
    P_{t_j} = \frac {\text{n events}_{t_j}} {\text{n at risk}_{t_j}}
    $$

    and the Survival probability at a time-index $q$, can be derived as:

    $$
    S(t_q) = \prod_{j=1}^{q} (1-P_{t_j}) 
    $$

    Applied to our example the risks of event $P_{t_j}$ are:

    $$
    \begin{matrix} 
    \text{index} &\text{interval} &\text{event} &\text{censor} & \text{at risk} & P_{t_j} \\
    --- &---&---&---&---&---    \\
        t_1 & [0,1) & 1  &   & 6 & 0.167 \\
        t_2 & [1,2) &  1 & 1 & 5 &  0.2 \\
        t_3 & [2,4) &  2 &   & 3 &  0.667 \\
        t_4 & [4,5) &  1 &   & 1 &  1.0 \\
    \end{matrix}
    $$

    And the corresponding Survival estimates at times $[1,2,4,5]$ are:
    $$
    \begin{matrix} 
    \text{index} &\text{time} & \text{Survival} \\
    ---&--- &---\\
    t_1  &  1 &  .83\\
    t_2  &  2 &  .67 \\
    t_3  &  4 &  .22\\
    t_4  &  5 &  0 \\
    \end{matrix}
    $$    

`BART-Survival` builds off this simple foundation by replacing $P_t$ with a probability risk estimate, $p_{t_j|x_i}$ yielded from the BART regression model, where $x_i$ is the covariate vector for the $i^{th}$ observation in the sample and $t_j$ is an indices of the observed time intervals as described above. The predicted values $p_{t_j|x_i}$ are generated for each observation, at each time interval and downstream targets can then be derived from these predicted values. For example the Survival probability estimate for a single observation at time-index $q$ can be derived as:

$$
S(t_q|x_i) = \prod_{j=1}^{q} (1-p_{t_j|x_i})
$$ 


## Data Preparation  
As displayed in the above example, Survival data is typically given as paired (*event status*, *event time*) outcomes, along with a set of covariates $x_i$ for each observation. In this setup, *event status* is typically a binary variable, $(1=\text{event; } 0=\text{censored})$ and *event time* is a representation of time.

Obtaining the $p_{t_j|x_i}$ values and subsequent Survival estimates is two-step process, which requires two additional augmented datasets created from the generic Survival data. The first step in the process involves training the $\text{BART}$ model which requires the **training augmented dataset** (**TAD**). The second step involves generating $p_{t_j|x_i}$ predictions from the trained $\text{BART}$ model, which requires the **predictive augmented dataset** (**PAD**).The **TAD** and **PAD** are similar in construction and both are created from an initial generic Survival dataset. 

The **TAD** is created by transforming the generic dataset's paired outcomes (*event status*, *event time*) into sets of outcomes, one set for each distinct observation in the generic dataset. Each new outcome set represents the event status for the observation over the series of discrete-time intervals up to and including each observation's *event time* as provided in the generic dataset. In the **TAD** *event status* is still binary, however the interpretation changes slightly where $1=\text{event}$ and $0=\text{no event}$. Censoring information can be conveyed in the **TAD** through the value of the last *event status* for the observations set, where a value of $0$ indicates censoring at that time interval.



For example if the Survival data is:
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
        \bold{6}& \bold{0} & \bold{8} \\
        \bold{7} &\bold{1} & \bold{12} \\
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
Then the transformation for observation $7$ would be represented in the _augmented_ dataset as the sequence of rows:
$$
\begin{matrix} 
\text{observation} & \text{event status} &\text{time} \\
---& --- &---\\
    7 & 0 & 4\\
    7 & 0 & 6 \\
    7 & 0 & 7 \\
    7 & 0 & 8 \\
    7 & 1 & 12 \\
\end{matrix}
$$
Similarily the transformation for observation $6$ would be represented as:
$$
\begin{matrix} 
\text{observation} & \text{event status} &\text{time} \\
---& --- &---\\
    6 & 0 & 4\\
    6 & 0 & 6 \\
    6 & 0 & 7 \\
    6 & 0 & 8 \\
\end{matrix}
$$

It is important to reiterate that for each observation, the new set of rows created for the **TAD** only include the time points indexed up to and including the original *event status*. For observation $\bold{7}$ in the above example the *event time* in the generic dataset is $12$, so the corresponding times in the **TAD** only include $4,6,7,8,12$ and for observation $\bold{6}$ the *event time* is $8$ and corresponding **TAD** times are $4,6,7,8$. 

The utility of the **TAD** is that each rows of the new dataset can be treated as independent observations which can be used in training a regression model that can yield a latent probability value conditional on time and/or other covariates incorporated in the model. With this model predictions across the range of times can be collected for each patient, which can subsequently used to generate individual, observation level Survival predictions or marginal estimates of other statistical estimands.

To generate the predictions of probability risks over the observed times the **PAD** dataset is used. The **PAD** transformation is similar to the **TAD**, but is simpler in that for each observation in the generic dataset the **PAD** creates a replicate of each observation for every time interval identified from the generic dataset.

Continuing the examples above, where the unique time interval endpoints are $4,6,7,8,12,14$, the **PAD** for observations $6$, and $7$ would be:

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

The **PAD** is required to get predictions for all observed time intervals for each observations. In the Inference section below, derivatives of the **PAD** are used to evaluate marginal estimates from the model.

Notably the **TAD** and **PAD** lengths can differ. **PAD** length is simple to calculate. If $n$ is the number of observations in the generic dataset and $k$ is the number of the unique time intervals observed in the generic dataset, then the length of the **PAD** is simply $n * k$. In the example data above it would be $(8*6)=36$

To calculate the **TAD** length, first the set of unique time intervals must be mapped relative indices. Then the length of the **TAD** is simply the cumulative sum of mapped index values for the observed dataset. Using the above example:
$$
    \begin{matrix}
    \text{unique times (k=6)} & \text{index}\\
    ---&---\\
    4 & 1\\
    6 & 2\\
    7 & 3\\
    8 & 4\\
    12 & 5\\
    14 & 6\\
    \end{matrix} 
    $$
  Then with the example dataset and mapped indices values the **TAD** size can be found as $27$.
$$
      \begin{matrix}
        \text{ time} & \text{index values}\\
    ---&---\\
    4 & 1\\
    6 & 2\\
    6 & 2\\
    7 & 3\\
    8 & 4\\
    8 & 4\\
    12 & 5\\
    14 & 6\\
    &---\\
    &27
    \end{matrix} 
$$

While the **PAD** and **TAD** lengths are not important for generating the models, they are helpful to keep in mind when completing an analysis. Specifically, the fact that the **TAD** and **PAD** can be far larger than the generic dataset (example going from $8$ observations to $27$ and $36$ observations in the **TAD** and **PAD**). When the length of the two datasets become large enough to make computation difficult (specifically when training the model with the **TAD**) it is recommended to downscale the *event time* values, which will reduce the total length of the two augmented datasets.

The downscaling algorithm we provide in the library takes the *event time*, divides by a scaling factor and then takes the "ceiling" truncation to create the new *scaled event time*.
Downscaling the example dataset by a factor of $4$ returns:

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

Downscaling the *event time* values causes loss of granularity within the time intervals. As shown in the example above, after downscaling (and then rescaling) the discrete time intervals on which the predictions can be made is now $4,8,12,16$, rather than the original set of $4,6,7,8,12,14$. While the information at time $6,7,14$ is lost, the evaluation at the downscaled times $4,8,12,16$ will be equivalent to the values of the unadjusted dataset at time $4,8,12,16$. With this equivalency, downscaling can be safely applied with the only remaining concern being what is the granularity of time required for the analysis. For example, in a multi-year health-outcomes study having the time recorded in days may be excessively granular. Using weeks or months could provide sufficient information for the study while significantly reducing the computational burden.




## Model

The aim of the `BART-Survival` algorithm is to train a non-parametric regression model with the **TAD** and yield predicted probabilities across the discrete time intervals with the **PAD** to generate estimates useful for statistical inference. 

Using the **TAD** the *event status* can be defined as the outcome vector of $y_{ij}$ values, *time* is represented by the $t_{ij}$ vector and any additional covariates represented by a vector of $x_{ij}$ values. The subscript $i$ refers to the $i^{th}$ observation of the generic dataset and the subscript $j$ refers to the $j^{th}$ index of the unique set of discrete time intervals previously defined as $t$. In the **TAD** and **PAD**, the $i$, $j$ indices can be thought of as multi-indices over the rows. 

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

Each $y_{ij}$ value can then be treated as independent draws of a $Bernoulli$ distribution parameterized with the probability of event $p_{ij}$ for observation $i$ at time index $j$. The $p_{ij}$ values are collected as the latent values yielded from a $probit$ regression of $y$ on $t$ and $x$. Formally the model is defined as:
$$
\begin{aligned}
    y_{ij} | p_{ij} \sim Bernoulli(p_{ij}) \\
    p_{ij} | \mu_{ij} = \Phi(\mu_{ij})\\
    \mu_{ij} \sim \text{BART}(t_{ij},x_{ij})\\
\end{aligned}
$$

Where $\Phi$ is the normal cumulative distribution function and $\text{BART}$ is the ensemble of regression trees which yield the $\mu_{ij}$ value for a given $t_{ij}$, $x_{ij}$ combination, which is subsequently transformed into the $p_{ij}$ value. 

The above model exists in a Bayesian framework through which the model is trained with an iterative Markov Chain Monte Carlo algorithm. In brief, each iteration of this algorithm results in the proposal of a new collection of regression trees. The proposed collection is either accepted or rejected based on a likelihood criterion. If the proposed collection is accepted, the collection is added as a contributor to the posterior distribution. After all iterations are completed a posterior distribution of $\text{BART}$ functions is constructed. The collection of posterior $\text{BART}$ functions is used to yield the predictive posterior distributions for each  $p_{ij}$. This means that for each $x_{ij},t_{ij}$ combination, $p_{ij}$ is actually a collection of values that construct a probability distribution for that $p_{ij}$ estimate. Point-estimates, like the mean can easily be generated as a empirical average of the posterior predictive distribution. Similarly, credible intervals can be constructed through the $5^{th}$ and $95^{th}$ percentiles of the distribution. More details on the $\text{BART}$ implementation used in the `BART-Survival` algorithm can be found in `PyMC-BART` repository and the accompanying publication [@quiroga2023]. 

## Survival 
A trained `BART-Survival` model can then be used along with the **PAD** to generate the vector of $p_{ij}$ predictions. To generate Survival estimates the $p_{ij}$ vector is regrouped by the $i$ indices. Then using the following equation the Survival probability at discrete times can be constructed:
$$
S(t_{q}| x_{iq}) = \prod_{j=1}^{q} (1-p_{ij})
$$ 

Where $q$ is the index of the unique time intervals from $1$,...,$k$. As described in the previous section a full posterior predictive distribution is returned for each $S_{t_q,x_{iq}}$. Point estimates and credible intervals can be obtained through the empirical mean and percentile functions of the returned posterior predictive distribution for each $S(t_q|x_{iq})$.

## Marginal Effects

Often the goal of Survival analysis is to evaluate the effects of the covariates included in the model. The commonly used Cox Proportional Hazard Model provides accessibility to such effects through the derived conditional Hazard Ratios associated with the covariates of a Cox Proportional Hazard Model.

`BART-Survival` provides capabilities to evaluate covariate effects through use of partial dependence functions which yield marginal effect estimates. The partial dependence function method involves generating predictions of $p$ for the observations in a **partial dependence augmented dataset** (**PDAD**). Using variations of the **PDAD** yields different sets of predictions $p$. These sets can then be contrasted to yield marginal effect estimates, including marginal Hazard Ratios which have similar interpretation as the Cox model's conditional Hazard Ratios.

**PDAD**s can be created through further augmentation of the previously described **PAD**. As a reminder, the **PAD** contains the generated time covariate $t$ and 
$k$ replicates of each $x_i$ from the generic dataset, where $k$ is the length of the uniquely observed *event times*. Starting the **PAD** the **PDAD** is generated through selection a specific variable $x_{[I]}$ from the covariates $x$, and then deterministically setting $x_{[I]}$ to a specific value for all observations. The unselected covariates $x_{[0]}$ that are not augmented and are consistent with the values in the generic dataset. An example of creating a **PDAD**  from the **PAD** is shown below. In this example a baseline **PAD** is used to create two **PDAD** datasets. The selected covariate $x_2$ is deterministically set to the values $0$ or $1$ for all observations in each dataset. 
 


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
    \text{PDAD}_{x_2=1} \space\space\space\space
    \begin{matrix}
    i & j & t & x_1 & \bold{x_2} & x_3 \\ 
    -&-&-&-&-&-&\\
    1 & 1 & 2 & 1.2 & \bold{1} & 10\\
    1 & 2 & 3 & 1.2 & \bold{1} & 10\\
    1 & 3 & 5 & 1.2 & \bold{1} & 10\\
    2 & 1 & 2 & 2.4 & \bold{1} & 12\\
    2 & 2 & 3 & 2.4 & \bold{1} & 12\\
    2 & 3 & 5 & 2.4 & \bold{1} & 12\\
    3 & 1 & 2 & 1.9 & \bold{1} & 3\\
    3 & 2 & 3 & 1.9 & \bold{1} & 3\\
    3 & 3 & 5 & 1.9 & \bold{1} & 3\\
    \end{matrix} 

    \text{PDAD}_{x_2=0} \space\space\space\space
    \begin{matrix}
    i & j & t & x_1 & \bold{x_2} & x_3 \\ 
    -&-&-&-&-&-&\\
    1 & 1 & 2 & 1.2 & \bold{0} & 10\\
    1 & 2 & 3 & 1.2 & \bold{0} & 10\\
    1 & 3 & 5 & 1.2 & \bold{0} & 10\\
    2 & 1 & 2 & 2.4 & \bold{0} & 12\\
    2 & 2 & 3 & 2.4 & \bold{0} & 12\\
    2 & 3 & 5 & 2.4 & \bold{0} & 12\\
    3 & 1 & 2 & 1.9 & \bold{0} & 3\\
    3 & 2 & 3 & 1.9 & \bold{0} & 3\\
    3 & 3 & 5 & 1.9 & \bold{0} & 3\\
    \end{matrix} 
$$

From the two **PDAD**s the predicted probabilities $p_{x_2=1}$, $p_{x_2=0}$ and Survival probabilities $S_{x_2=1}$, $S_{x_2=0}$ can be generated:

$$
    \begin{matrix}
    i & j & p_{x_2=1} & S_{x_2=1} & p_{x_2=0} & S_{x_2=0}\\
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

The marginal expectations of $p_{x_{[i]}}$ and $S_{x_{[i]}}$ at a specific time can be further derived by taking the average of the estimates over observations $i$ for the specified time $t$ indexed by $j$: 

$$E_{i}[p_{x_{[I]}}|t_j] = {\frac 1 n} \sum_{i=1}^n {p_{x_{[I]_{ij}}}} $$
$$E_{i}[S_{x_{[I]}}(t_j)] = {\frac 1 n} \sum_{i=1}^n {S_{x_{[I]_{i}}}(t_j)} $$

Where $E_{i}$ is the expectation over $i$,...,$n$ observations. From the above example this yields the expectations for time indices $j=$ $1$,$2$,$3$.

$$
    \begin{matrix}
    j & t & E_i[p_{x_2=1}] & E_i[S_{x_2=1}] & E_i[p_{x_2=0}] & E_i[S_{x_2=0}]\\
    -&-&-&-&-&-\\
    1 & 2 & 0.17 & 0.83 & 0.11 & 0.91 \\ 
    2 & 3 & 0.22 & 0.65 & 0.16 & 0.8 \\
    3 & 5 & 0.18 & 0.54 & 0.12 & 0.73\\
    \end{matrix}
$$

These expectations can be further used to make comparisons can be made between the evaluation of various values of $x_{[I]}$ across multiple **PDAD**s. Common marginal effect estimates derived from these predicted values include:

- Marginal difference is Survival probability at time $t$:


$$
\text{Surv. Diff.}_{marg}(t_j) = E_{i}[S_{x_{[I]_2}}(t_j)] - E_{i}[S_{x_{[I]_1}}(t_j)]
$$

- Marginal Risk Ratio at time $t$:

$$
\text{RR}_{marg}(t_j) = \frac {E_{i}[p_{x_{[I]_2j}}]} {E_{i}[p_{x_{[I]_1j}}]}
$$

- Marginal Hazard Ratio (expectation over $i$ and times $t$):
$$
\text{HR}_{marg}(t_j) = \frac {E_{it}[p_{x_{[I]_2}j}]} {E_{it}[p_{x_{[I]_1}j}]}
$$

Continuing the example, the marginal effect of $x_2$ as measured by difference in Survival probability when $x_2 = 1$ and $x_2 =0$ at the times $2,3,5$ can be examined below: 
$$
    \begin{matrix}
    j & t & \text{Surv.Diff.}\\
    -&-&-&\\
    1 & 2 &-0.08 \\
    2 & 3 &-0.15 \\
    3 & 5 &-0.19 \\
    \end{matrix}
$$

These results can be conveniently interpreted. For example, at time $t=5$ the increase of $x_2$ from $0$ to $1$ leads to an average change in the Survival probability of $-0.19$, where Survival Probability is in the range $0$-$1$. Additionally, as mentioned in the previous sections, all estimates of the model will first be yielded as posterior predictive distributions. The empirical mean and percentile function of this distribution yield the point estimates described above and their respective credible intervals.

# Demonstration 
The following is a brief demonstration on how to use `BART-Survival`. The *rossi* survival dataset from the lifelines library is used for this example.

```python
from lifelines.datasets import load_rossi
from bart_survival import surv_bart as sb
import numpy as np

######################################
# Load rossi dataset from lifelines
rossi = load_rossi()
names = rossi.columns.to_numpy()
rossi = rossi.to_numpy()
```

After loading the libraries and data, the first step is to generate the **TAD** and **PAD** datasets. In this step, the time (originally in days) is downscaled by a factor of 7, setting time to be measured in weeks.

```python
######################################
# Transform data into 'augmented' dataset
# Requires creation of the training dataset and a predictive dataset for inference
# TAD
trn = sb.get_surv_pre_train(
    y_time=rossi[:,0],
    y_status=rossi[:,1],
    x = rossi[:,2:],
    time_scale=7
)

# PAD
post_test = sb.get_posterior_test(
    y_time=rossi[:,0],
    y_status=rossi[:,1],
    x = rossi[:,2:],
    time_scale=7
)
```

The next step is to initialize the model, which involves setting several parameter values. The key considerations when initializing the models is number of trees and the split rules. The number of trees controls how many regression trees will be used. Typically 50 trees is a good default, but it can be adjusted to assist in model performance. Split rules is a specific `PyMC-BART` parameter and is used to designate the how the regression trees are constructed. The one requirement of the split rules is that the time covariate has to be set as a `pmb.ContinuousSplitRule()`. Otherwise, generally continuous variables can assigned `pmb.ContinuousSplitRule()` and categorical variables assigned `pmb.OneHotSplitRule()`. It is is recommended to review the `PyMC-BART` literature for more information regarding parameterization of the models.

```python
######################################
# Instantiate the BART models
# model_dict is defines specific model parameters
model_dict = {"trees": 50,
    "split_rules": [
        "pmb.ContinuousSplitRule()", # time
        "pmb.OneHotSplitRule()", # fin
        "pmb.ContinuousSplitRule()",  # age
        "pmb.OneHotSplitRule()", # race
        "pmb.OneHotSplitRule()", # wexp
        "pmb.OneHotSplitRule()", # mar
        "pmb.OneHotSplitRule()", # paro
        "pmb.ContinuousSplitRule()", # prio
    ]
}
# sampler_dict defines specific sampling parameters
sampler_dict = {
            "draws": 200,
            "tune": 200,
            "cores": 8,
            "chains": 8,
            "compute_convergence_checks": False
        }
BSM = sb.BartSurvModel(
    model_config=model_dict, 
    sampler_config=sampler_dict
)
```

The model can then be trained with the **TAD** input and predicted $p_{ij}$ values yielded with the PAD input.

```python
#####################################
# Fit Model with TAD
BSM.fit(
    y =  trn["y"], 
    X = trn["x"], 
    weights=trn["w"],
    coords = trn["coord"],
    random_seed=5
)

# Get posterior predictive for evaluation using the PAD
post1 = BSM.sample_posterior_predictive(
    X_pred=post_test["post_x"], 
    coords=post_test["coords"] 
)
```
Finally the Survival probability can derived from the $p_{ij}$ estimates. 
```python
# Convert to SV probability.
sv_prob = sb.get_sv_prob(post1)
print(sv_prob["sv"].shape)
# (1600, 432, 8)
```
The sv_prob object above is a dictionary containing numpy arrays of both the $p_{ij}$ and $s_{ij}$ estimates, labeled "prob" and "sv" respectively. At step of the output the values are held in a three dimensional array to keep the values organsized. The dimensions of the arrays are:
- axis 0 = draws of the posterior predictive distribution: 1600
- axis 1 = observations $i$: 432
- axis 2 = times $j$: 8

These arrays can be easily reduced down to point estimates and credible intervals using basic numpy methods. For example to get the estimate of the mean over all observations, first get the mean over the observations (axis 1) followed by the mean over the posterior draws (axis 0). The results being the estimated mean Survival over the 8 time intervals. 

Similarly the 0.05-0.95 credible interval for the estimated mean Survival can be returned as the quantile evaluations of the same mean-over-axis-1 array. This yields a (2,8) array with the lower and upper bounds (rows) of the credible interval defined for each time point (columns).

```python
# get the mean value across observations for each time within each draw of the posterior predictive distribution
ave_obs = sv_prob["sv"].mean(axis=1)
print(ave_obs.shape)
# (1600, 8)

# get the average across the posterior draws 
ave_obs_draws = ave_obs.mean(0)
print(ave_obs_draws)
# [0.97259158 0.93862734 0.9015028  0.86572104 0.83166388 0.79712983 0.76321937 0.73634161]

# get the .05 and .95 percentiles of the mean across posterior draws
ci_obs_draws = np.quantile(ave_obs, [0.05, 0.95], axis=0)
print(ci_obs_draws)
# lower bound
#[[0.96432971 0.92615903 0.88601098 0.84845099 0.81381525 0.77771897 0.7426666  0.71438202]
# upper bound
# [0.98066217 0.94989412 0.91620586 0.8829086  0.84948681 0.81667259 0.78434529 0.75795991]]
```

Examples of generation of marginal effect estimates can be found in the example notebooks provided in the repository documentation.


# Conclusion

`BART-Survival` provides the computational methods required for completing non-parametric discrete-time Survival analysis. This approach can have several advantages over alternative Survival methods. These advantages include capabilities to incorporate non-linear and interaction effects into the model, naturally ability to regularize the model (which reduces the risk of over-fitting) and of being robust to issues of multi-collinearity. The `BART-Survival` approach is especially useful when the assumptions of alternative Survival methods are violated. 

Our `BART-Survival` algorithm has been tested in a rigorous simulation study, with additional applications to real-world data. While the manuscript for this work is currently under development, the results indicate similar performance as the the R-based BART Survival method across settings of varied complexity and both demonstrate the previously describe advantages over other survival approaches as the complexity of the underlying data increases.

Our library provides a convenient API for completing discrete-time Survival analysis, along with the functionality to customize the methodology as needed. The associated API documentation can be found [here](https://cdcgov.github.io/BART-Survival/build/html/index.html), along with the associated github repository [`BART-Survival`](https://github.com/CDCgov/BART-Survival).




# Acknowledgements

We thank Oscar Rincón-Guevara for helpful suggestions and review. We also thank Tegan Boehmer, Sachin Agnihotri, and Matt Ritchey for supporting the project throughout its development.

# References