## `aif360.sklearn`

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
  - [x] Use "prot_attr" and "priv_group" keywords to specify protected attributes to
  functions
  - [x] Generalized confusion matrix
  - [ ] Sample distortion metrics
- [ ] Make inprocessing algorithms compatible as sklearn `Estimator`s
  - [x] Adversarial debiasing
  - [ ] **[External]** `get_feature_names()` from data preprocessing
  steps that would remove DataFrame formatting
    - [ ] [SLEP007](https://github.com/scikit-learn/enhancement_proposals/pull/17)/[SLEP008](https://github.com/scikit-learn/enhancement_proposals/pull/18) - feature names
  - [ ] Prejudice remover
  - [ ] Meta-fair classifier
- [ ] Make preprocessing algorithms compatible as sklearn `Transformer`s
  - [ ] **[External]** Add functionality to modify X and y
    - [ ] [SLEP005](https://github.com/scikit-learn/enhancement_proposals/pull/15) - Resampler API (see discussion; meta-estimator workaround may be enough)
  - [ ] Disparate impact remover
  - [ ] Learning fair representations
  - [ ] Optimized preprocessing
  - [X] Reweighing
    - [X] Meta-estimator workaround
    - [ ] **[External]** [SLEP006](https://github.com/scikit-learn/enhancement_proposals/pull/16) - Sample properties (meta-estimator works but would be very nice to have)
- [ ] Make postprocessing algorithms compatible
  - [x] Calibrated equalized odds postprocessing
    - [x] Meta-estimator workaround again
  - [ ] Equalized odds postprocessing
  - [ ] Reject option classification
- [ ] Miscellaneous:
  - [ ] Explainers
