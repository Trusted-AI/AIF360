from setuptools import setup, find_packages

setup(name='aif360',
      version='0.1',
      description='IBM AI Fairness 360',
      author='IBM Corporation',
      author_email='IBM@ibm.com',
      license='Apache 2.0',
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
