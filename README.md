# AI Fairness 360 (AIF360)

[![Continuous Integration](https://github.com/Trusted-AI/AIF360/actions/workflows/ci.yml/badge.svg)](https://github.com/Trusted-AI/AIF360/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/aif360/badge/?version=latest)](https://aif360.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/aif360.svg)](https://badge.fury.io/py/aif360)
[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/aif360)](https://cran.r-project.org/package=aif360)

The AI Fairness 360 toolkit is an extensible open-source library containing techniques developed by the
research community to help detect and mitigate bias in machine learning models throughout the AI application lifecycle. AI Fairness 360 package is available in both Python and R.

The AI Fairness 360 package includes
1) a comprehensive set of metrics for datasets and models to test for biases,
2) explanations for these metrics, and
3) algorithms to mitigate bias in datasets and models.
It is designed to translate algorithmic research from the lab into the actual practice of domains as wide-ranging
as finance, human capital management, healthcare, and education. We invite you to use it and improve it.

The [AI Fairness 360 interactive experience](https://aif360.res.ibm.com/data)
provides a gentle introduction to the concepts and capabilities. The [tutorials
and other notebooks](./examples) offer a deeper, data scientist-oriented
introduction. The complete API is also available.

Being a comprehensive set of capabilities, it may be confusing to figure out
which metrics and algorithms are most appropriate for a given use case. To
help, we have created some [guidance
material](https://aif360.res.ibm.com/resources#guidance) that can be
consulted.

We have developed the package with extensibility in mind. This library is still
in development. We encourage the contribution of your metrics, explainers, and
debiasing algorithms.

Get in touch with us on [Slack](https://aif360.slack.com) (invitation
[here](https://join.slack.com/t/aif360/shared_invite/zt-5hfvuafo-X0~g6tgJQ~7tIAT~S294TQ))!


## Supported bias mitigation algorithms

* Optimized Preprocessing ([Calmon et al., 2017](http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention))
* Disparate Impact Remover ([Feldman et al., 2015](https://doi.org/10.1145/2783258.2783311))
* Equalized Odds Postprocessing ([Hardt et al., 2016](https://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning))
* Reweighing ([Kamiran and Calders, 2012](http://doi.org/10.1007/s10115-011-0463-8))
* Reject Option Classification ([Kamiran et al., 2012](https://doi.org/10.1109/ICDM.2012.45))
* Prejudice Remover Regularizer ([Kamishima et al., 2012](https://rd.springer.com/chapter/10.1007/978-3-642-33486-3_3))
* Calibrated Equalized Odds Postprocessing ([Pleiss et al., 2017](https://papers.nips.cc/paper/7151-on-fairness-and-calibration))
* Learning Fair Representations ([Zemel et al., 2013](http://proceedings.mlr.press/v28/zemel13.html))
* Adversarial Debiasing ([Zhang et al., 2018](https://arxiv.org/abs/1801.07593))
* Meta-Algorithm for Fair Classification ([Celis et al., 2018](https://arxiv.org/abs/1806.06055))
* Rich Subgroup Fairness ([Kearns, Neel, Roth, Wu, 2018](https://arxiv.org/abs/1711.05144))
* Exponentiated Gradient Reduction ([Agarwal et al., 2018](https://arxiv.org/abs/1803.02453))
* Grid Search Reduction ([Agarwal et al., 2018](https://arxiv.org/abs/1803.02453), [Agarwal et al., 2019](https://arxiv.org/abs/1905.12843))
* Fair Data Adaptation ([Plečko and Meinshausen, 2020](https://www.jmlr.org/papers/v21/19-966.html), [Plečko et al., 2021](https://arxiv.org/abs/2110.10200))
* Sensitive Set Invariance/Sensitive Subspace Robustness ([Yurochkin and Sun, 2020](https://arxiv.org/abs/2006.14168), [Yurochkin et al., 2019](https://arxiv.org/abs/1907.00020))

## Supported fairness metrics

* Comprehensive set of group fairness metrics derived from selection rates and error rates including rich subgroup fairness
* Comprehensive set of sample distortion metrics
* Generalized Entropy Index ([Speicher et al., 2018](https://doi.org/10.1145/3219819.3220046))
* Differential Fairness and Bias Amplification ([Foulds et al., 2018](https://arxiv.org/pdf/1807.08362))
* Bias Scan with Multi-Dimensional Subset Scan ([Zhang, Neill, 2017](https://arxiv.org/abs/1611.08292))

## Setup

### R

``` r
install.packages("aif360")
```

For more details regarding the R setup, please refer to instructions [here](aif360/aif360-r/README.md).

### Python

Supported Python Configurations:

| OS      | Python version |
| ------- | -------------- |
| macOS   | 3.8 – 3.11     |
| Ubuntu  | 3.8 – 3.11     |
| Windows | 3.8 – 3.11     |

### (Optional) Create a virtual environment

AIF360 requires specific versions of many Python packages which may conflict
with other projects on your system. A virtual environment manager is strongly
recommended to ensure dependencies may be installed safely. If you have trouble
installing AIF360, try this first.

#### Conda

Conda is recommended for all configurations though Virtualenv is generally
interchangeable for our purposes. [Miniconda](https://conda.io/miniconda.html)
is sufficient (see [the difference between Anaconda and
Miniconda](https://conda.io/docs/user-guide/install/download.html#anaconda-or-miniconda)
if you are curious) if you do not already have conda installed.

Then, to create a new Python 3.11 environment, run:

```bash
conda create --name aif360 python=3.11
conda activate aif360
```

The shell should now look like `(aif360) $`. To deactivate the environment, run:

```bash
(aif360)$ conda deactivate
```

The prompt will return to `$ `.

### Install with `pip`

To install the latest stable version from PyPI, run:

```bash
pip install aif360
```

Note: Some algorithms require additional dependencies (although the metrics will
all work out-of-the-box). To install with certain algorithm dependencies
included, run, e.g.:

```bash
pip install 'aif360[LFR,OptimPreproc]'
```

or, for complete functionality, run:

```bash
pip install 'aif360[all]'
```

The options for available extras are: `OptimPreproc, LFR, AdversarialDebiasing,
DisparateImpactRemover, LIME, ART, Reductions, FairAdapt, inFairness,
LawSchoolGPA, notebooks, tests, docs, all`

If you encounter any errors, try the [Troubleshooting](#troubleshooting) steps.

### Manual installation

Clone the latest version of this repository:

```bash
git clone https://github.com/Trusted-AI/AIF360
```

If you'd like to run the examples, download the datasets now and place them in
their respective folders as described in
[aif360/data/README.md](aif360/data/README.md).

Then, navigate to the root directory of the project and run:

```bash
pip install --editable '.[all]'
```

#### Run the Examples

To run the example notebooks, complete the manual installation steps above.
Then, if you did not use the `[all]` option, install the additional requirements
as follows:

```bash
pip install -e '.[notebooks]'
```

Finally, if you did not already, download the datasets as described in
[aif360/data/README.md](aif360/data/README.md).

### Troubleshooting

If you encounter any errors during the installation process, look for your
issue here and try the solutions.

#### TensorFlow

See the [Install TensorFlow with pip](https://www.tensorflow.org/install/pip)
page for detailed instructions.

Note: we require `'tensorflow >= 1.13.1'`.

Once tensorflow is installed, try re-running:

```bash
pip install 'aif360[AdversarialDebiasing]'
```

TensorFlow is only required for use with the
`aif360.algorithms.inprocessing.AdversarialDebiasing` class.

#### CVXPY

On MacOS, you may first have to install the Xcode Command Line Tools if you
never have previously:

```sh
xcode-select --install
```

On Windows, you may need to download the [Microsoft C++ Build Tools for Visual
Studio 2019](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16).
See the [CVXPY Install](https://www.cvxpy.org/install/index.html#mac-os-x-windows-and-linux)
page for up-to-date instructions.

Then, try reinstalling via:

```bash
pip install 'aif360[OptimPreproc]'
```

CVXPY is only required for use with the
`aif360.algorithms.preprocessing.OptimPreproc` class.

## Using AIF360

The `examples` directory contains a diverse collection of jupyter notebooks
that use AI Fairness 360 in various ways. Both tutorials and demos illustrate
working code using AIF360. Tutorials provide additional discussion that walks
the user through the various steps of the notebook. See the details about
[tutorials and demos here](examples/README.md)

## Citing AIF360

A technical description of AI Fairness 360 is available in this
[paper](https://arxiv.org/abs/1810.01943). Below is the bibtex entry for this
paper.

```
@misc{aif360-oct-2018,
    title = "{AI Fairness} 360:  An Extensible Toolkit for Detecting, Understanding, and Mitigating Unwanted Algorithmic Bias",
    author = {Rachel K. E. Bellamy and Kuntal Dey and Michael Hind and
	Samuel C. Hoffman and Stephanie Houde and Kalapriya Kannan and
	Pranay Lohia and Jacquelyn Martino and Sameep Mehta and
	Aleksandra Mojsilovic and Seema Nagar and Karthikeyan Natesan Ramamurthy and
	John Richards and Diptikalyan Saha and Prasanna Sattigeri and
	Moninder Singh and Kush R. Varshney and Yunfeng Zhang},
    month = oct,
    year = {2018},
    url = {https://arxiv.org/abs/1810.01943}
}
```

## AIF360 Videos

* Introductory [video](https://www.youtube.com/watch?v=X1NsrcaRQTE) to AI
  Fairness 360 by Kush Varshney, September 20, 2018 (32 mins)

## Contributing
The development fork for Rich Subgroup Fairness (`inprocessing/gerryfair_classifier.py`) is [here](https://github.com/sethneel/aif360). Contributions are welcome and a list of potential contributions from the authors can be found [here](https://trello.com/b/0OwPcbVr/gerryfair-development).
