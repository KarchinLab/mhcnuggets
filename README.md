# MHCnuggets

Branch   |[Travis CI build status](https://travis-ci.org)                                                                 
---------|------------------------------------------------------------------------------------------------------------------------------
`master` |[![Build Status](https://travis-ci.org/KarchinLab/mhcnuggets.svg?branch=master)](https://travis-ci.org/KarchinLab/mhcnuggets) 
`develop`|[![Build Status](https://travis-ci.org/KarchinLab/mhcnuggets.svg?branch=develop)](https://travis-ci.org/KarchinLab/mhcnuggets)
`richel` |[![Build Status](https://travis-ci.org/KarchinLab/mhcnuggets.svg?branch=richel)](https://travis-ci.org/KarchinLab/mhcnuggets)

Welcome to MHCnuggets! Presumably you're here to do some
peptide-MHC prediction and not because you were [hungry](https://www.mcdonalds.com/us/en-us/product/chicken-mcnuggets-4-piece.html).

### Usage ###
For an overview of how to use MHCnuggets please refer to the Jupyter notebook
called `user_guide.ipynb` in the repository

If you would like to use MHCnuggets as a docker container, there are several options:

1. [MHCnuggets with command line interface](https://github.com/KarchinLab/mhcnuggets/wiki/Creating-a-mhcnuggets-docker-container-with-command-line-interface)

2. [MHCnuggets with Jupyter Notebook interface](https://github.com/KarchinLab/mhcnuggets/wiki/Creating-a-mhcnuggets-docker-container-with-Jupyter-Notebook-interface)

3. [MHCnuggets container for batch operations](https://github.com/KarchinLab/mhcnuggets/wiki/Creating-and-running-the-MHCnuggets-batch-container)

### Installation ###

MHCnuggets is `pip` installable as:
```bash
pip install mhcnuggets
```

**Required pacakges:**

* numpy
* scipy
* scikit-learn
* pandas
* keras
* tensorflow
* varcode

You might want to check if the Keras backend is configured to use
the Tensorflow backend.
