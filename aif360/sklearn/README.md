## `aif360.sklearn`

This is a wholly separate interface for interacting with data, viewing metrics,
and running debiasing algorithms than the main AIF360 package. The purpose of
this sub-package is to match scikit-learn paradigms/APIs for easier integration
in typical machine learning workflows.

To do:

- [x] Reformat datasets as separate X and y (and sample_weight) DataFrame
objects with sample properties (protected attributes) as the index
- [ ] Load included datasets in the above format (partially done)
  - [x] Use `sklearn.datasets.fetch_openml` to load UCI datasets (#53)
  - [ ] COMPAS
  - [ ] MEPS
- [ ] Implement metrics as individual functions instead of instance methods
(mostly done)
  - [x] Make certain metrics compatible as sklearn scorers
  - [ ] Generalized confusion matrix
  - [ ] Sample distortion metrics
- [ ] Make inprocessing algorithms compatible as sklearn `Estimator`s
  - [ ] Adversarial debiasing
  - [ ] Meta-fair classifier
  - [ ] Prejudice remover
- [ ] Make preprocessing algorithms compatible as sklearn `Transformer`s
  - [ ] Add functionality to modify X and y (worst case: just `predict()` +
  `transform()` separately)
  - [ ] Disparate impact remover
  - [ ] Learning fair representations
  - [ ] Optimized preprocessing
  - [ ] Reweighing
    - [ ] Use dynamic object to pass sample_weight to estimator, etc. after they
    are fitted
- [ ] Make postprocessing algorithms compatible
  - [ ] Allow `fit()` on `y_true`,`y_pred`
  - [ ] Calibrated equalized odds postprocessing
  - [ ] Equalized odds postprocessing
  - [ ] Reject option classification
- [ ] Miscellaneous:
  - [ ] LIME encoder
  - [ ] Explainers
