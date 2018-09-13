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
      # python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <3.7',
      install_requires=[
          'numpy',
          'scipy',
          'pandas==0.23.3',
          'scikit-learn',
          'numba',
      ],
      include_package_data=True,
      zip_safe=False)
