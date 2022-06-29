from setuptools import setup;
from setuptools import find_packages;

def readme():
    with open('README.md') as f:
        return f.read()


setup(name='mhcnuggets',
      version='2.4',
      description='MHCnuggets: Neoantigen peptide MHC binding prediction for class I and II',
      long_description=readme(),
      url='http://karchinlab.org/apps/mhcnuggets.html', # TODO
      author='Melody Shao',
      author_email='melody.xiaoshan.shao@gmail.com',
      license='Apache License',
      packages=find_packages(exclude=['tests']),
      include_package_data=True,
      install_requires=[
          'numpy==1.22.4',
          'scipy',
          'scikit-learn',
          'pandas',
          'tensorflow',
          'keras',
      ],
      classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Bio-Informatics'
      ],
      zip_safe=True)
