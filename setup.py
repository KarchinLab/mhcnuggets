from setuptools import setup;
from setuptools import find_packages;

def readme():
    with open('README.md') as f:
        return f.read()


setup(name='mhcnuggets',
      version='2.4.0',
      description='MHCnuggets: Neoantigen peptide MHC binding prediction for class I and II',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='http://karchinlab.org/apps/mhcnuggets.html', # TODO
      author='Melody Shao',
      author_email='melody.xiaoshan.shao@gmail.com',
      license='Apache License',
      packages=find_packages(exclude=['tests']),
      include_package_data=True,
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'pandas',
          'keras',
          'tensorflow',
          'varcode',
      ],
      classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Bio-Informatics'
      ],
      zip_safe=True)
