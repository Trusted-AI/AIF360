# AI Fairness 360 Examples (Tutorials and Demos)

This directory contains a diverse collection of jupyter notebooks that use [AI Fairness 360](https://github.com/Trusted-AI/AIF360/) in various ways.
Both tutorials and demos illustrate working code using AIF360.  Tutorials provide additional discussion that walks
the user through the various steps of the notebook.

## Tutorials
The [Credit scoring](https://nbviewer.jupyter.org/github/Trusted-AI/AIF360/blob/main/examples/tutorial_credit_scoring.ipynb) tutorial is the recommended first tutorial to get an understanding for how AIF360 works.  It first provides a brief summary of a machine learning workflow and an overview of AIF360.  It then demonstrates the use of one fairness metric (mean difference) and one bias mitigation algorithm (optimized preprocessing) in the context of age bias in a credit scoring scenario using the [German Credit dataset](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29).

The [Medical expenditure](https://nbviewer.jupyter.org/github/Trusted-AI/AIF360/blob/main/examples/tutorial_medical_expenditure.ipynb) tutorial is a comprehensive tutorial demonstrating the interactive exploratory nature of a data scientist detecting and mitigating racial bias in a care management scenario.  It uses a variety of fairness metrics (disparate impact, average odds difference, statistical parity difference, equal opportunity difference, and Theil index) and algorithms (reweighing, prejudice remover, and disparate impact remover). It also demonstrates how explanations can be generated for predictions made by models learned with the toolkit using LIME.
Data from the Medical Expenditure Panel Survey ([2015](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181) and [2016](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192)) is used in this tutorial.

## Demos
Below is a list of additional notebooks that demonstrate the use of AIF360.

**NEW:** [sklearn/demo_new_features.ipynb](sklearn/demo_new_features.ipynb): highlights the
features of the new `scikit-learn`-compatible API

[demo_optim_data_preproc.ipynb](demo_optim_data_preproc.ipynb): demonstrates a generalization of the credit scoring tutorial that  shows the full machine learning workflow for the optimized data pre-processing algorithm for bias mitigation on several datasets

[demo_adversarial_debiasing.ipynb](demo_adversarial_debiasing.ipynb): demonstrates the use of the adversarial debiasing in-processing algorithm to learn a fair classifier

[demo_calibrated_eqodds_postprocessing.ipynb](demo_calibrated_eqodds_postprocessing.ipynb): demonstrates the use of an odds-equalizing post-processing algorithm for bias mitigiation

[demo_disparate_impact_remover.ipynb](demo_disparate_impact_remover.ipynb): demonstrates the use of a disparate impact remover pre-processing algorithm for bias mitigiation

[demo_json_explainers.ipynb](demo_json_explainers.ipynb):

[demo_lfr.ipynb](demo_lfr.ipynb):  demonstrates the use of the learning fair representations algorithm for bias mitigation

[demo_lime.ipynb](demo_lime.ipynb):  demonstrates how LIME - Local Interpretable Model-Agnostic Explanations - can be used with models learned with the AIF 360 toolkit to generate explanations for model predictions

[demo_reject_option_classification.ipynb](demo_reject_option_classification.ipynb): demonstrates the use of the Reject Option Classification (ROC) post-processing algorithm for bias mitigation

[demo_reweighing_preproc.ipynb](demo_reweighing_preproc.ipynb):  demonstrates the use of a reweighing pre-processing algorithm for bias mitigation