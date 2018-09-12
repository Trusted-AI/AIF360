from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='aif360',
      version='0.1.0',
      description='IBM AI Fairness 360',
      author='aif360 developers',
      author_email='aif360@us.ibm.com',
      url='https://github.com/IBM/AIF360',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='Apache License 2.0',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'pandas==0.23.3',
          'scikit-learn',
          'cvxpy==0.4.11;platform_system!="windows"',
          'numba',
          'tensorflow==1.1.0;platform_system!="windows"',
          'networkx==1.11',
          'BlackBoxAuditing;python_version>="3"'
      ],
      include_package_data=True,
      zip_safe=False)
