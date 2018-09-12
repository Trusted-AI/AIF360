# AI Fairness 360 (AIF360 v0.1.0)

[![Build Status](https://travis-ci.com/IBM/AIF360.svg?branch=master)](https://travis-ci.com/IBM/AIF360)

Welcome to [AI Fairness 360](http://aif360.mybluemix.net/). We hope you will use it and contribute to it to help engender trust in AI and make the world more equitable for all.

Machine learning models are increasingly used to inform high stakes decisions about people. Although machine learning, by its very nature, is always a form of statistical discrimination, the discrimination becomes objectionable when it places certain privileged groups at systematic advantage and certain unprivileged groups at systematic disadvantage. Biases in training data, due to either prejudice in labels or under-/over-sampling, yields models with unwanted bias ([Barocas and Selbst](http://www.californialawreview.org/2-big-data/)).


The AI Fairness 360 Python package includes a comprehensive set of metrics for datasets and models to test for biases, explanations for these metrics, and algorithms to mitigate bias in datasets and models. The [AI Fairness 360 interactive experience](http://aif360.mybluemix.net/data) provides a gentle introduction to the concepts and capabilities. The [tutorials and other notebooks](./examples) offer a deeper, data scientist-oriented introduction. The complete API is also available.


Being a comprehensive set of capabilities, it may be confusing to figure out which metrics and algorithms are most appropriate for a given use case. To help, we have created some [guidance material](http://aif360.mybluemix.net/resources#guidance) that can be consulted.


We have developed the package with extensibility in mind.  We encourage the contribution of your metrics, explainers, and debiasing algorithms. Please join the community to get started as a contributor. Get in touch with us on [Slack](https://aif360.slack.com) (invitation [here](https://join.slack.com/t/aif360/shared_invite/enQtNDI5Nzg2NTk0MTMyLTU4N2UwODVmMTYxZWMwZmEzZmZkODdjMTk5NWUwZDNhNDhlMzNkZDNhOTYwZDNlODc1MTdjYzY5OTU2OWQ1ZmY))!


## Supported bias mitigation algorithms

* Flavio P. Calmon, Dennis Wei, Bhanukiran Vinzamuri, Karthikeyan Natesan Ramamurthy, and Kush R. Varshney, “[Optimized Pre-Processing for Discrimination Prevention](http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention),” Conference on Neural Information Processing Systems, 2017.


* Michael Feldman, Sorelle A. Friedler, John Moeller, Carlos Scheidegger, and Suresh Venkatasubramanian, “[Certifying and Removing Disparate Impact](https://doi.org/10.1145/2783258.2783311),” ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2015.


* Moritz Hardt, Eric Price, and Nathan Srebro, “[Equality of Opportunity in Supervised Learning](https://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning),” Conference on Neural Information Processing Systems, 2016.


* Faisal Kamiran and Toon Calders, “[Data Preprocessing Techniques for Classification without Discrimination](http://doi.org/10.1007/s10115-011-0463-8),” Knowledge and Information Systems, 2012.


* Faisal Kamiran, Asim Karim, and Xiangliang Zhang, “[Decision Theory for Discrimination-Aware Classification](https://doi.org/10.1109/ICDM.2012.45),” IEEE International Conference on Data Mining, 2012.


* Toshihiro Kamishima, Shotaro Akaho, Hideki Asoh, and Jun Sakuma, “[Fairness-Aware Classifier with Prejudice Remover Regularizer](https://rd.springer.com/chapter/10.1007/978-3-642-33486-3_3),” Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 2012.


* Geoff Pleiss, Manish Raghavan, Felix Wu, Jon Kleinberg, and Kilian Q. Weinberger, “[On Fairness and Calibration](https://papers.nips.cc/paper/7151-on-fairness-and-calibration),” Conference on Neural Information Processing Systems, 2017.


* Richard Zemel, Yu (Ledell) Wu, Kevin Swersky, Toniann Pitassi, and Cynthia Dwork, “[Learning Fair Representations](http://proceedings.mlr.press/v28/zemel13.html),” International Conference on Machine Learning, 2013.


* Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell, “[Mitigating Unwanted Biases with Adversarial Learning](http://www.aies-conference.com/wp-content/papers/main/AIES_2018_paper_162.pdf),” AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society, 2018.

## Supported fairness metrics

* Comprehensive set of group fairness metrics derived from selection rates and error rates


* Comprehensive set of sample distortion metrics


* Till Speicher, Hoda Heidari, Nina Grgic-Hlaca, Krishna P. Gummadi, Adish Singla, Adrian Weller, and Muhammad Bilal Zafar, “[A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual & Group Unfairness via Inequality Indices](https://doi.org/10.1145/3219819.3220046),” ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.


## Setup

Installation is easiest on a Unix system running Python 3. See the additional instructions for [Windows](#windows) and [Python 2](#python-2) as appropriate.

### Linux and MacOS

#### Installation with `pip`

```bash
pip install aif360
```

This package supports both Python 2 and 3. However, for Python 2, the `BlackBoxAuditing` package must be [installed manually](#python-2).

To run the example notebooks, install the additional requirements as follows:

```bash
pip install -r requirements.txt
```

#### Manual installation

Clone the latest version of this repository:

```bash
git clone https://github.com/IBM/AIF360
```

Then, navigate to the root directory of the project and run:

```bash
pip install .
```

### Windows

Follow the same steps above as for Linux/MacOS. Then, follow the [instructions](https://www.tensorflow.org/install/install_windows) to install the appropriate build of TensorFlow which is used by `aif360.algorithms.inprocessing.AdversarialDebiasing`. Note: `aif360` requires version 1.1.0. For example,

```bash
pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.1.0-cp35-cp35m-win_amd64.whl
```

To use `aif360.algorithms.preprocessing.OptimPreproc`, install `cvxpy` by following the [instructions](http://www.cvxpy.org/install/index.html#windows) and be sure to install version 0.4.11, e.g.:

```bash
pip install cvxpy==0.4.11
```

### Python 2

Some additional installation is required to use `aif.algorithms.preprocessing.DisparateImpactRemover` with Python 2:

```bash
git clone https://github.com/algofairness/BlackBoxAuditing
```

In the root directory of `BlackBoxAuditing`, run:

```bash
echo -n $PWD/BlackBoxAuditing/weka.jar > python2_source/BlackBoxAuditing/model_factories/weka.path
echo "include python2_source/BlackBoxAuditing/model_factories/weka.path" >> MANIFEST.in
pip install --no-deps .
```

This will produce a minimal installation which satisfies our requirements.

## Using AIF360


## Citing AIF360

   Please ask in Slack channel.
