###############
Getting Started
###############
Welcome to AI Fairness 360!

AIF360 is an extensible open-source library containing techniques developed by the
research community to help detect and mitigate bias in machine learning models
throughout the AI application lifecycle. This document will provide an overview of
its features and conventions for users of the toolkit.

Definitions of relevant terms may be found in the
`Glossary <https://aif360.res.ibm.com/resources#glossary>`_ page.

Installation
============
To install the latest stable version from PyPI, run::

     pip install aif360

Note: Some algorithms require additional dependencies (although the metrics will
all work out-of-the-box). To install with certain algorithm dependencies
included, run, e.g.::

     pip install 'aif360[LFR,OptimPreproc]'

or, for complete functionality, run::

     pip install 'aif360[all]'

It is recommended to install AIF360 in a virtual environment (e.g., conda).

AIF360 is also available as an R package. To install aif360-r from CRAN::

     install.packages("aif360")

For more details, see the
`AIF360-r README <https://github.com/Trusted-AI/AIF360/aif360/aif360-r/README.md>`_.

.. _sklearn-api:

scikit-learn API
================
AIF360 has two interfaces for handling data formats, a legacy interface and a
scikit-learn-compatible one. The sklearn API is preferred going forward and all
future development will be focused there. Note: there may be slight differences
in results between the two versions due to the implementations but both are
valid.

AIF360 contains four major classes of features: datasets, estimators, metrics,
and detectors.

Datasets
--------
Currently, AIF360 focuses on tabular data which are formatted as pandas
``DataFrames``.

>>> from aif360.sklearn.datasets import fetch_compas
>>> fetch_compas().X.head()
                        sex  age          age_cat              race  juv_fel_count  juv_misd_count  juv_other_count  priors_count c_charge_degree                   c_charge_desc
sex  race
Male Other             Male   69  Greater than 45             Other              0               0                0             0               F    Aggravated Assault w/Firearm
     African-American  Male   34          25 - 45  African-American              0               0                0             0               F  Felony Battery w/Prior Convict
     African-American  Male   24     Less than 25  African-American              0               0                1             4               F           Possession of Cocaine
     Other             Male   44          25 - 45             Other              0               0                0             0               M                         Battery
     Caucasian         Male   41          25 - 45         Caucasian              0               0                0            14               F       Possession Burglary Tools

Per-sample *protected attributes* are stored in the index of the DataFrame. If
multiple protected attributes are present, a single attribute or subset may be
selected for use in a specific algorithm/metric by using the relevant
``prot_attr`` argument.

In some cases, it is possible to use numpy arrays for features and protected
attributes separately by passing an array to ``prot_attr`` such as in metrics.
However, since support for ``predict_params`` in sklearn is sparse, ``prot_attr``
is an ``__init__`` argument which means handling ``fit`` and ``predict`` separately
is difficult.

Estimators
----------
AIF360 estimators are machine learning models which mitigate unfair bias in some
way. They may also be referred to as *algorithms*. AIF360 estimators inherit
from scikit-learn ``Estimators`` and are intended to work in conjunction with each
other. Note: while scikit-learn estimators can generally be used in an AIF360
pipeline, the reverse is not always true due to the data limitations explained
above. This is ongoing work to reduce friction between the libraries.

Algorithms are divided into three classes based on how they mitigate bias. For
guidance on choosing an algorithm, see the
`Resources <https://aif360.res.ibm.com/resources#guidance>`_ page.

Pre-processing
^^^^^^^^^^^^^^
Pre-processors act on data and produce fair representations which are
subsequently used by another machine learning model. Under scikit-learn
conventions, these work like ``Transformers``.

>>> from aif360.sklearn.preprocessing import LearnedFairRepresentations
>>> Xt = LearnedFairRepresentations().fit_transform(X, y)

In some cases, AIF360 pre-processors necessarily break scikit-learn conventions,
e.g., :class:`~aif360.sklearn.preprocessing.Reweighing` and
:class:`~aif360.sklearn.preprocessing.FairAdapt`. This may cause difficulty when
using them within a ``Pipeline``. Typically, a ``MetaEstimator`` may be used to
perform pre-processing and estimation in a single step. See relevant examples for
details.

In-processing
^^^^^^^^^^^^^
In-processors apply mitigation during model training in order to produce a fair
model. Under scikit-learn conventions, these are simply
``Classifiers``/``Regressors``.

>>> from aif360.sklearn.inprocessing import AdversarialDebiasing
>>> y_pred = AdversarialDebiasing().fit(X_train, y_train).predict(X_test)

Post-processing
^^^^^^^^^^^^^^^
Post-processors act on the outputs of a model (and possibly the inputs as well)
and produce new, fairer outputs. These types of models are classified as
``MetaEstimators`` in scikit-learn.

>>> from sklearn.linear_model import LogisticRegression
>>> from aif360.sklearn.postprocessing import RejectOptionClassifierCV, PostProcessingMeta
>>> y_pred = PostProcessingMeta(
...      LogisticRegression(),
...      RejectOptionClassifierCV('sex', scoring='disparate_impact')
... ).fit(X_train, y_train).predict(X_test)

Metrics
-------
Fairness may be defined in many different ways according to different situations
and stakeholders. It is typically defined as a notion of equality within the
population. Fairness metrics measure the deviation from equality (bias) in data
or model outputs. These can be divided into two classes of metrics according to
the definition of fairness they measure: individual or group fairness.

Group fairness
^^^^^^^^^^^^^^
Group fairness metrics compare statistical measures between different
subpopulations of the data divided by protected attributes. Often these are
aggregated into a single score, for example, by taking a difference or ratio
between the unprivileged and privileged group for binary groups.

These functions are similar to scikit-learn metrics with additional arguments
for ``prot_attr`` (index label of ``y_true``/``y_pred`` or explicit array) and
``priv_group`` (if binary groups are required).

>>> from aif360.sklearn.metrics import disparate_impact_ratio
>>> di = disparate_impact_ratio(y_true, y_pred, prot_attr='race' priv_group='White', pos_label=1)

Individual fairness
^^^^^^^^^^^^^^^^^^^
Individual fairness is most commonly defined as "similar individuals are treated
similarly" (`Dwork, et al. 2011 <https://arxiv.org/pdf/1104.3913.pdf>`_). These
functions therefore, require access to the features, ``X``, labels, ``y``, and a
distance function/matrix.

>>> from aif360.sklearn.metrics import consistency_score
>>> cons = consistency_score(X_test, y_pred)

Distributional fairness
^^^^^^^^^^^^^^^^^^^^^^^
A generalized entropy index, or inequality index, measures how unequally the
benefits of a model are distributed amongst the population. A value of zero
indicates equal benefit is given to every individual. It is similar to individual
fairness but because GEI does not take into account individual similarity (i.e.,
"we're all equal") it may be thought of as a separate category, more related to
utility measurement. See
`Mitigating Bias in Machine Learning, Chapter 5 <https://mitigatingbias.ml/#ch_InequalityIndices>`_
for more details.

>>> from aif360.sklearn.metrics import generalized_entropy_error
>>> ent = generalized_entropy_error(y_true, y_pred)

Detectors
---------
The goal of a bias detector is to identify subgroup(s) which are especially
disadvantaged by a model or dataset.

>>> from aif360.sklearn.detectors import bias_scan
>>> subset, score = bias_scan(X, y_true, y_pred)

Future work
-----------
Other data modalities
^^^^^^^^^^^^^^^^^^^^^
AIF360 is currently best suited for tabular data tasks however any model used by
humans could contain bias. In most cases, metrics can still be computed using
only labels and protected attribute data but if you are interested in
contributing an algorithm which does not fit within the scikit-learn interface
style, please submit an issue on
`GitHub <https://www.github.com/Trusted-AI/AIF360/issues>`_ and let's discuss!
