
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07213/status.svg)](https://doi.org/10.21105/joss.07213)


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


