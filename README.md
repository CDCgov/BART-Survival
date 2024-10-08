## Overview

``BART-Survival`` is a Python package that supports discrete-time Survival analyses using the non-parametric machine learning algorithm, Bayesian Additive Regression Trees (BART). ``BART-Survival`` combines the performance of the BART algorithm from the [PyMC-BART](https://www.pymc.io/projects/bart/en/latest/) library with the proper structural formatting required to complete the end-to-end Survival analysis. 

``BART-Survival``'s performance is comparative to other Survival regression methods, such as Cox Proportional Hazard and AFT models, in low-complexity settings and can outperform these other models in high-complexity settings were their respective assumptions may fail (i.e. proportional hazard assumption, linearity assumptions). Additionally, the Bayesian framework provides easily accessible uncertainty intervals and capabilities to extensively interrogate trained model.

The ``BART-Survival`` library provides a simple API for completing standard Survival analysis, as well as allowing exposure to the underlying PyMC code providing accessibility to extend the ``BART-Survival`` library when necessary.

This repository contains the source code and documentation for the `BART-Survival` package as well as [example](https://github.com/CDCgov/BART-Survival/tree/main/examples/) notebooks. 

``BART-Survival`` can be installed directly from PyPi 
```python
pip install `BART-Survival`==0.1.1
```
Or as accessed from this repository. 

### API
[BART-Survival API](https://cdcgov.github.io/BART-Survival/build/html/index.html) 



### Background

Survival analysis methods are statistical methods used to describe the risk of an event occurrence over a period of time. The `BART-Survival` package provides a discrete-time Survival method which aims to model Survival as a function of the cumulative risk of event occurrences over the series of discrete time intervals. 

Using discrete-time intervals provides a convenient approach to flexibly model the latent probability of event as a non-parametric function of the distinct time interval and a set of observation covariates. The latent probabilities can then be used for deriving Survival probability or other estimates.


The foundation of the method is simple.  
1. Create a sequence of time intervals, denoted as $t_j$ with ($j = {1,...,k}$), from the range of observed event times. 
2. Then for each interval $t_j$ obtain the number of observations with an event, along with the total number of observations at risk for having an event. 
3. Finally, the risk of event occurrence within each interval $t_j$ can naively be derived as: 

```math
\begin{equation}
P_{t_j} = \frac {\text{n events}_{t_j}} {\text{n at risk}_{t_j}}
\end{equation}
```

and the Survival probability $S(t)$ at a time $q$, can be derived as:

```math
\begin{equation}
S(t_q) = \prod_{j=1}^{q} (1-P_{t_j}) 
\end{equation}
```

`BART-Survival` builds off this simple foundation by replacing $P_t$ with a probability risk estimate, $p_{t_j|x_i}$ yielded from the BART regression model. The predicted values $p_{t_j|x_i}$ are generated for each observation, at each time interval from the set $j$. Downstream targets can be further derived from these predicted values with observation-level Survival derived as:

```math
\begin{equation}
S(t_q|x_i) = \prod_{j=1}^{q} (1-p_{t_j|x_i})
\end{equation}
```
Statistical estimands can also be estimated using the predicted $p_{t_j|x_i}$ through evaluation of marginal functions of the predicted values. 

To properly model $p_{t|x}$ the data requires an transformation from a standard Survival dataset to a _augmented_ dataset. Survival data is typically given as a paired (**event status**, **event time**) outcome, along with a set of covariates for each observation. In this setup **event status** is typically a binary variable (1=event; 0=censored) and **event time** is some continuous representation of time.

The _augmented_ dataset transforms the generic dataset from a single, paired (**event status**, **event time**) outcome per observation, to a sequence of single (**event status**) outcomes over the series of discrete-time intervals, up to the given **event time** for the observation. 

For example if the unique set of a dataset's **event times** is **{4,6,7,8,12,14}**, and a single observation's paired outcome is (**event status** = 1, **event times** = 12), then the single observation will be represented in the _augmented_ dataset as the sequence of rows:

```math
\begin{matrix} 
\text{event status} &\text{time} \\
 --- &---\\
    0 & 4\\
    0 & 6 \\
    0 & 7 \\
    0 & 8 \\
    1 & 12 \\
\end{matrix}
```

Each row in the _augmented_ dataset can then be treated as an independent observation, with **event status** as the outcome $Y$ and the **event time** $T$ as an added covariate. Now each of the original observations are represented by $j$ rows ($j=1,...,i_\text{event time})$ and the corresponding variables can be denoted as ($y_{ij}$, $t_{j}$, $x_{ij}$). 

Using the new _augmented_ dataset, the model is simplified to a probit regression of $y_{ij}$ on time $t_{j}$ and covariates $x_{ij}$, which yields a latent value $p_{ij}$ corresponding to $P(y_{ij} = 1)$. Explicitly the model is defined as:


```math
\begin{align*}
    y_{ij} | p_{ij} \sim Bernoulli(p_{ij}) \\
    p_{ij} | \mu_{ij} = \Phi(\mu_{ij})\\
    \mu_{ij} \sim \text{BART}(j,x_{i})\\
\end{align*}
```

where $\Phi$  is the CDF of the Normal distribution.

A trained `BART-Survival` model can then be used to yield the $p_{ij}$ predictions, which can be used to derive Survival as described above.



#### Inference
A common goal of Survival regression models is to derive a statistical estimate for the adjusted effect of a variable on the outcome. A classic example of these estimates are Hazard Ratios derived from the coefficients of Cox Proportional Hazard Models.

With `BART-Survival`, the underlying BART algorithm does not rely on a linear equation and therefore does not produce coefficient that can be treated as conditional measures of effects. Instead, variable effects are summarized as marginal effect estimates which can derived through use of partial dependence functions. 

The partial dependence function method is relatively simple. It involves generating predictions of $p$ from a trained `BART-Survival` model using a _augmented partial dependence_ (_APD_) dataset as input. 

The _APD_ dataset is generated so that a specific variable $x_{[I]}$ is deterministically set to a specific value for all observations while the other covariates $`x_{[O]_i}`$ remain as observed. Each observation is then expanded over the time-intervals $1,...,j_{T_{max}}$ to create the discrete-time datasets. Here $`j_{T_{max}}`$ is the maximum time across all **event times**.

To estimate the effect of an variable on the outcome, multiple _APD_ datasets are created. Each _APD_ dataset varies the value of the specific variable of interest (i.e. $`x_{[I]_1}`$, $`x_{[I]_2}`$ ), allowing evaluation of the outcome under different conditions of that variable. The $p_{ij}$ values from each predicted dataset ($p_{[1]}$, $p_{[2]}$), can be used to obtain the marginal estimates of a specific outcome. 

Common marginal effect estimates derived from these predicted values include:

- Marginal difference is Survival probability at time $j$:


```math
\begin{equation}
\text{Surv Diff}_{marg} = E_{i}[S_{p_{[2]}}(t_j)] - E_{i}[S_{p_{[1]}}(t_j)]
\end{equation}
```

- Marginal Risk Ratio at time $j$:

```math
\text{RR}_{marg} = \frac {E_{i}[p_{[2]_{j}}]} {E_{i}[p_{[1]_{j}}]}
```


- Marginal Hazard Ratio (assuming constant hazard rates):
```math
\text{HR}_{marg} = \frac {E_{ij}[p_{[2]}]} {E_{ij}[p_{[1]}]}
```

For users familiar with causal inference literature, these partial dependence functions are similar to the g-estimation and counterfactual outcomes methods common to the causal inference field.

In addition to providing point estimates, the `BART-Survival` naturally generates Bayesian credible intervals as a product of the posterior predictive distribution. The credible intervals can provide useful measure of uncertainty and allows for bayesian variants of hypothesis testing and statistical inference to be made on the estimates.


#### Summary

The `BART-Survival` package provide the algorithms necessary to complete the 3 major steps of the Survival analysis.

1. Generate augmented dataset
2. Train-predict-transform the BART model and estimates
3. Generate _augmented partial dependece_ datasets and generating marginal estimates.


### Demonstration

Brief demo of the basic steps.

```python
from lifelines.datasets import load_rossi
from bart_survival import surv_bart as sb
import numpy as np

######################################
# Load rossi dataset from lifelines
rossi = load_rossi()
names = rossi.columns.to_numpy()
rossi = rossi.to_numpy()

######################################
# Transform data into 'augmented' dataset
# Requires creation of the training dataset and a predictive dataset for inference
trn = sb.get_surv_pre_train(
    y_time=rossi[:,0],
    y_status=rossi[:,1],
    x = rossi[:,2:],
    time_scale=7
)

post_test = sb.get_posterior_test(
    y_time=rossi[:,0],
    y_status=rossi[:,1],
    x = rossi[:,2:],
    time_scale=7
)

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
BSM = sb.BartSurvModel(model_config=model_dict, sampler_config=sampler_dict)

#####################################
# Fit Model
BSM.fit(
    y =  trn["y"],
    X = trn["x"],
    weights=trn["w"],
    coords = trn["coord"],
    random_seed=5
)

# Get posterior predictive for evaluation.
post1 = BSM.sample_posterior_predictive(X_pred=post_test["post_x"], coords=post_test["coords"])

# Convert to SV probability.
sv_prob = sb.get_sv_prob(post1)

```




### User Guide/Example Notebooks
https://github.com/CDCgov/`BART-Survival`/blob/main/examples/example1.ipynb

https://github.com/CDCgov/`BART-Survival`/blob/main/examples/example2.ipynb


### Validation study links
In progress...

---
---
# CDCgov General Disclaimers
This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm).  GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise. 

## Related Documents
* [Open Practices](open_practices.md)
* [Rules of Behavior](rules_of_behavior.md)
* [Disclaimer](DISCLAIMER.md)
* [Contribution Notice](CONTRIBUTING.md)
* [Code of Conduct](code-of-conduct.md)
* [Licence](licence)
  
## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice
The repository utilizes code licensed under the terms of the Apache Software
License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](DISCLAIMER.md)
and [Code of Conduct](code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records, but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices
Please refer to [CDC's Template Repository](https://github.com/CDCgov/template) for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/main/CONTRIBUTING.md), [public domain notices and disclaimers](https://github.com/CDCgov/template/blob/main/DISCLAIMER.md), and [code of conduct](https://github.com/CDCgov/template/blob/main/code-of-conduct.md).


