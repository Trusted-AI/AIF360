# AI Fairness 360 (AIF360 v0.1.1)

[![Build Status](https://travis-ci.com/IBM/AIF360.svg?branch=master)](https://travis-ci.com/IBM/AIF360)

The AI Fairness 360 toolkit is an open-source library to help detect and remove bias in machine learning models. The AI Fairness 360 Python package includes a comprehensive set of metrics for datasets and models to test for biases, explanations for these metrics, and algorithms to mitigate bias in datasets and models.

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

## Supported fairness metrics

* Comprehensive set of group fairness metrics derived from selection rates and error rates
* Comprehensive set of sample distortion metrics
* Generalized Entropy Index ([Speicher et al., 2018](https://doi.org/10.1145/3219819.3220046))


## Setup

Installation is easiest on a Unix system running Python 3.6. See the [Troubleshooting](#troubleshooting) section if you have issues with other configurations.

### (Optional) Create a Virtualenv environment

AIF360 requires specific versions of many Python packages which may conflict with other projects on your system. Virtualenv creates an isolated virtual Python environment where these dependencies may be installed safely. If you have trouble installing AIF360, try this first.

```bash
mkdir ~/virtualenvs && cd ~/virtualenvs  # this can be wherever you like storing virtualenvs
virtualenv -p python3 aif360             # or substitute your preferred version of Python
source aif360/bin/activate
```

For Windows, this is a little different:

```bat
md C:\virtualenvs             # this can be wherever you like storing virtualenvs
cd C:\virtualenvs
virtualenv -p python3 aif360  # or substitute your preferred version of Python
aif360/Scripts/activate
```

The shell should now look like `(aif360) $`.

Also, upgrade `pip` to be safe:

```bash
(aif360)$ pip install --upgrade pip
```

To deactivate the environment, run

```bash
deactivate
```

The prompt will return to `$ `.

See the [Virtualenv User Guide](https://virtualenv.pypa.io/en/stable/userguide/) for more details.

### Install with minimal dependencies

#### Installation with `pip`

```bash
pip install aif360
```

This package supports Python 2.7, 3.5, and 3.6. However, for Python 2, the `BlackBoxAuditing` package must be [installed manually](#python-2).

#### Manual installation

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

Finally, if you did not already, download the datasets as described in [aif360/data/README.md](aif360/data/README.md) but place them **in the appropriate sub-folder** in `~/virtualenvs/aif360/lib/pythonX.X/site-packages/aif360/data/raw` where *pythonX.X* is the version of python you are using.

### Troubleshooting

If you encounter any errors during the installation process, look for your issue here and try the solutions.

#### Windows

##### TensorFlow

Note: TensorFlow 1.1.0 only supports Python 3.5 officially on Windows. You can get it from [python.org](https://www.python.org/downloads/release/python-352/).

Then, try:

```bat
pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.1.0-cp35-cp35m-win_amd64.whl
```

TensorFlow is only required for use with the `aif360.algorithms.inprocessing.AdversarialDebiasing` class.

##### CVXPY

You may need to download the [Visual Studio C++ compiler for Python](https://www.microsoft.com/en-us/download/details.aspx?id=44266) and rerun

```bat
pip install cvxpy==0.4.11
```

CVXPY is only required for use with the `aif360.algorithms.preprocessing.OptimPreproc` class.

#### Python 2

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

A technical description of AI Fairness 360 is available in this [paper](https://arxiv.org/pdf/1810.01943.pdf).   Below is the bibtex entry for this paper.

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
    year = {2017},
    url = {https://arxiv.org/pdf/1807.01069}
}
```


   