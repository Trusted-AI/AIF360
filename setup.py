from setuptools import setup, find_packages

long_description = """The AI Fairness 360 toolkit is an open-source library to help detect and remove bias in machine
learning models. The AI Fairness 360 Python package includes a comprehensive set of metrics for datasets and models to
test for biases, explanations for these metrics, and algorithms to mitigate bias in datasets and models.

We have developed the package with extensibility in mind. This library is still in development. We encourage the
contribution of your datasets, metrics, explainers, and debiasing algorithms."""

setup(name='aif360',
      version='0.1.1',
      description='IBM AI Fairness 360',
      author='aif360 developers',
      author_email='aif360@us.ibm.com',
      url='https://github.com/IBM/AIF360',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='Apache License 2.0',
      packages=find_packages(),
      # python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <3.7',
      install_requires=[
          'numpy>=1.14,<1.16',
          'scipy',
          'pandas==0.23.3',
          'scikit-learn',
          'numba',
      ],
      package_data={'aif360': ['data/*', 'data/*/*', 'data/*/*/*']},
      include_package_data=True,
      zip_safe=False)
