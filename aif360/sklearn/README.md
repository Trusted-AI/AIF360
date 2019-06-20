## `aif360.sklearn`

[![Build Status](https://travis-ci.org/IBM/AIF360.svg?branch=sklearn-compat)](https://travis-ci.org/IBM/AIF360)

This is a wholly separate interface for interacting with data, viewing metrics,
and running debiasing algorithms than the main AIF360 package. The purpose of
this sub-package is to match scikit-learn paradigms/APIs for easier integration
in typical machine learning workflows.

See [Getting Started](examples/Getting%20Started.ipynb) to see `aif360.sklearn`
in action.

To do:

- [x] Reformat datasets as separate X and y (and sample_weight) DataFrame
objects with sample properties (protected attributes) as the index
- [ ] Load included datasets in the above format
  - [x] Use `sklearn.datasets.fetch_openml` to load UCI datasets (#53)
  - [ ] COMPAS
  - [ ] MEPS
- [ ] Implement metrics as individual functions instead of instance methods
  - [x] Make certain metrics compatible as sklearn scorers
  - [x] Use "groups" and "priv_group" keywords to specify protected attributes to
  functions
  - [ ] Generalized confusion matrix
  - [ ] Sample distortion metrics
- [ ] Make inprocessing algorithms compatible as sklearn `Estimator`s
  - [ ] **[External]** `get_feature_names()` from data preprocessing
  steps that would remove DataFrame formatting
    - [ ] [SLEP008](https://github.com/scikit-learn/enhancement_proposals/pull/18 )?
  - [ ] Prejudice remover
  - [ ] Adversarial debiasing
  - [ ] Meta-fair classifier
- [ ] Make preprocessing algorithms compatible as sklearn `Transformer`s
  - [ ] **[External]** Add functionality to modify X and y
    - [ ] [SLEP001](https://github.com/scikit-learn/enhancement_proposals/blob/master/slep001/proposal.rst)
  - [ ] Disparate impact remover
  - [ ] Learning fair representations
  - [ ] Optimized preprocessing
  - [X] Reweighing
    - [X] Use dynamic object to pass sample_weight to estimator, etc. after they
    are fitted (NOTE: does not work with GridSearchCV)
    - [ ] **[External]** SLEP006 - Sample properties
- [ ] Make postprocessing algorithms compatible
  - [ ] **[External]** Allow for `fit(y_true, y_pred)`
    - [ ] New SLEP?
  - [ ] Calibrated equalized odds postprocessing
  - [ ] Equalized odds postprocessing
  - [ ] Reject option classification
- [ ] Miscellaneous:
  - [ ] LIME encoder
  - [ ] Explainers
