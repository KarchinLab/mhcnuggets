from setuptools import setup;
from setuptools import find_packages;

def readme():
    with open('README.md') as f:
        return f.read()


setup(name='mhcnuggets',
      version='2.0',
      description='MHCnuggets: Neoantigen peptide MHC binding prediction for class I and II',
      long_description=readme(),
      url='http://karchinlab.org/apps/mhcnuggets.html', # TODO
      author='Rohit Bhattacharya',
      author_email='rohit.bhattachar@gmail.com',
      license='Apache License',
      packages=find_packages(exclude=['tests']),
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'tensorflow',
          'keras'
      ],
      classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Developers/Users',
      'Topic :: Scientific/Engineering :: Bioinformatics'
      ],
      zip_safe=True)
