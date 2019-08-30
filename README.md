# AI Fairness 360 (AIF360 v0.1.1)

[![Build Status](https://travis-ci.com/IBM/AIF360.svg?branch=master)](https://travis-ci.com/IBM/AIF360)

The AI Fairness 360 toolkit is an extensible open-source library containg techniques developed by the research community to help detect and mitigate bias in machine learning models throughout the AI application lifecycle.
The AI Fairness 360 Python package includes
 1) a comprehensive set of metrics for datasets and models to test for biases, 2) explanations for these metrics, and 3) algorithms to mitigate bias in datasets and models.  It is designed to translate algorithmic research from the lab into the actual practice of domains as wide-ranging as finance, human capital management, healthcare, and education. We invite you to use it and improve it.

The [AI Fairness 360 interactive experience](http://aif360.mybluemix.net/data) provides a gentle introduction to the concepts and capabilities. The [tutorials and other notebooks](./examples) offer a deeper, data scientist-oriented introduction. The complete API is also available.

Being a comprehensive set of capabilities, it may be confusing to figure out which metrics and algorithms are most appropriate for a given use case. To help, we have created some [guidance material](http://aif360.mybluemix.net/resources#guidance) that can be consulted.

We have developed the package with extensibility in mind. This library is still in development. We encourage the contribution of your metrics, explainers, and debiasing algorithms.

Get in touch with us on [Slack](https://aif360.slack.com) (invitation [here](https://join.slack.com/t/aif360/shared_invite/enQtNDI5Nzg2NTk0MTMyLTU4N2UwODVmMTYxZWMwZmEzZmZkODdjMTk5NWUwZDNhNDhlMzNkZDNhOTYwZDNlODc1MTdjYzY5OTU2OWQ1ZmY))!


## Supported bias mitigation algorithms

* Optimized Preprocessing ([Calmon et al., 2017](http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention))
* Disparate Impact Remover ([Feldman et al., 2015](https://doi.org/10.1145/2783258.2783311))
* Equalized Odds Postprocessing ([Hardt et al., 2016](https://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning))
* Reweighing ([Kamiran and Calders, 2012](http://doi.org/10.1007/s10115-011-0463-8))
* Reject Option Classification ([Kamiran et al., 2012](https://doi.org/10.1109/ICDM.2012.45))
* Prejudice Remover Regularizer ([Kamishima et al., 2012](https://rd.springer.com/chapter/10.1007/978-3-642-33486-3_3))
* Calibrated Equalized Odds Postprocessing ([Pleiss et al., 2017](https://papers.nips.cc/paper/7151-on-fairness-and-calibration))
* Learning Fair Representations ([Zemel et al., 2013](http://proceedings.mlr.press/v28/zemel13.html))
* Adversarial Debiasing ([Zhang et al., 2018](http://www.aies-conference.com/wp-content/papers/main/AIES_2018_paper_162.pdf))
* Meta-Algorithm for Fair Classification ([Celis et al.. 2018](https://arxiv.org/abs/1806.06055))

## Supported fairness metrics

* Comprehensive set of group fairness metrics derived from selection rates and error rates
* Comprehensive set of sample distortion metrics
* Generalized Entropy Index ([Speicher et al., 2018](https://doi.org/10.1145/3219819.3220046))


## Setup

Supported Configurations:

| OS      | Python version |
| ------- | -------------- |
| macOS   | 2.7, 3.5, 3.6  |
| Ubuntu  | 2.7, 3.5, 3.6  |
| Windows | 3.5            |

Installation is easiest on a Unix-like system running Python 3. See the [Troubleshooting](#troubleshooting) section if you have issues with other configurations.

### (Optional) Create a virtual environment

AIF360 requires specific versions of many Python packages which may conflict with other projects on your system. A virtual environment manager is strongly recommended to ensure dependencies may be installed safely. If you have trouble installing AIF360, try this first.

#### Conda

Conda is recommended for all configurations though Virtualenv is generally interchangeable for our purposes ([CVXPY](#cvxpy) may require conda in some cases). Miniconda is sufficient (see [the difference between Anaconda and Miniconda](https://conda.io/docs/user-guide/install/download.html#anaconda-or-miniconda) if you are curious) and can be installed from [here](https://conda.io/miniconda.html) if you do not already have it.

Then, to create a new Python 3.5 environment, run:

```bash
conda create --name aif360 python=3.5
conda activate aif360
```

The shell should now look like `(aif360) $`. To deactivate the environment, run:

```bash
(aif360)$ conda deactivate
```

The prompt will return to `$ `.

Note: Older versions of conda may use `source activate aif360` and `source deactivate` (`activate aif360` and `deactivate` on Windows).

### Install with minimal dependencies

To install the latest stable version from PyPI, run:

```bash
pip install aif360
```

This package supports Python 2.7, 3.5, and 3.6. However, for Python 2, the `BlackBoxAuditing` package must be [installed manually](#blackboxauditing).

Some algorithms require additional dependencies not included in the minimal installation. To use these, we recommend a full installation.

### Full installation

Clone the latest version of this repository:

```bash
git clone https://github.com/IBM/AIF360
```

If you'd like to run the examples, download the datasets now and place them in their respective folders as described in [aif360/data/README.md](aif360/data/README.md).

Then, navigate to the root directory of the project and run:

```bash
pip install .
```

#### Run the Examples

To run the example notebooks, install the additional requirements as follows:

```bash
pip install -r requirements.txt
```

Then, follow the [Getting Started](https://pytorch.org) instructions from PyTorch to download and install the latest version for your machine.

Finally, if you did not already, download the datasets as described in [aif360/data/README.md](aif360/data/README.md) but place them **in the appropriate sub-folder** in `$ANACONDA_PATH/envs/aif360/lib/python3.5/site-packages/aif360/data/raw` where `$ANACONDA_PATH` is the base path to your conda installation (e.g. `~/anaconda`).

### Troubleshooting

If you encounter any errors during the installation process, look for your issue here and try the solutions.

#### TensorFlow

In some cases, the URL is required for installation:

```bat
# WINDOWS
pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.1.0-cp35-cp35m-win_amd64.whl

# MACOS
pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0-py3-none-any.whl

# LINUX
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp36-cp36m-linux_x86_64.whl
```

Substitute Python version numbers for your configuration as appropriate (Note: TensorFlow 1.1.0 only supports Python 3.5 officially on Windows).

TensorFlow is only required for use with the `aif360.algorithms.inprocessing.AdversarialDebiasing` class.

#### CVXPY

On Windows, you may need to download the appropriate [Visual Studio C++ compiler for Python](https://wiki.python.org/moin/WindowsCompilers) and run:

```bat
conda install -c https://conda.anaconda.org/omnia scs
```

Then, rerun:

```bat
pip install cvxpy==0.4.11
```

See the [CVXPY 0.4.11 Installation Instructions](https://www.cvxpy.org/versions/0.4.11/install/index.html#windows-with-anaconda) for details.

CVXPY is only required for use with the `aif360.algorithms.preprocessing.OptimPreproc` class.

#### BlackBoxAuditing

Some additional installation is required to use `aif360.algorithms.preprocessing.DisparateImpactRemover` with Python 2.7. In a directory of your choosing, run:

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

The `examples` directory contains a diverse collection of jupyter notebooks that use AI Fairness 360 in various ways.
Both tutorials and demos illustrate working code using AIF360. Tutorials provide additional discussion that walks the
user through the various steps of the notebook. See the details about [tutorials and demos here](examples/README.md)

## Citing AIF360

A technical description of AI Fairness 360 is available in this [paper](https://arxiv.org/abs/1810.01943).  Below is the bibtex entry for this paper.

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

* Introductory [video](https://www.youtube.com/watch?v=X1NsrcaRQTE) to AI Fairness 360 by Kush Varshney, September 20, 2018 (32 mins)
