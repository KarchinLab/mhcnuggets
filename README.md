# MHCnuggets

Welcome to MHCnuggets! Presumably you're here to do some
peptide-MHC prediction and not because you were [hungry](https://www.mcdonalds.com/us/en-us/product/chicken-mcnuggets-4-piece.html).

### Prediction ###
MHCnuggets comes with pre-trained models for class I and class II
prediction that can be used out of the box. The maximum length
for a peptide is 15 residues for class I and 30 for class II.
Prediction is pretty easy and can be done like so

```bash
python scripts/predict.py -a HLA-A0201 -p data/test/test_peptides.peps
```
You can use the `-o` flag to specify an output path, otherwise the output
is written to `stdout`


### Training ###
If you'd like to get fancy and train your own models on your own
datasets, it's pretty simple. Simply invoke the training script
for the respective class of model you're trying to train like so

```bash
python scripts/train.py -a HLA-A0201 -s test/HLA-A0201.h5 -n 100 -d data/production/curated_training_data.csv
```

The recommended number of epochs for training a model from scratch
is 100 epochs. Have a look at `train.py` for additional flags
such as learning rate etc. that you might want to set yourself if
you don't like the defaults

### Transfer learning ###
Transfer learning is just as easy, and is actually integral to
MHCnuggets. All models besides the ones for the alleles with the most
data (HLA-A\*02:01 and HLA-DRB1\*01:01) are trained via transfer
learning like so

```bash
python scripts/train.py -a HLA-DRB10301 -s test/test.h5 -n 100 -s -t test/HLA-DRB10101.h5 -d data/production/curated_training_data.csv
```

```bash
python scripts/train.py -a HLA-A0203 -s test/test.h5 -n 100 -s -t test/HLA-A0201.h5 -d data/production/curated_training_data.csv
```

Remember to do transfer learning only within the same class of allele


### Fine tuning ###
We fine tune models when we see there's a strong relationship between
two MHC alleles i.e. the model of one allele has very high predictive
value for another. In this case we do a slightly special case of
trasnfer learning with less epochs.

```bash
python scripts/train.py -a HLA-A0203 -s test/test.h5 -n 25 -s -t test/HLA-A0201.h5 -d data/production/curated_training_data.csv
```

Have a look at `data/production/mhc_tuning.csv` for a list
of MHC pairs that can be "fine-tuned".


### Evaluation ###
Finally, if you want to know how well your model is doing, you can evaluate it like so

```bash
python scripts/evaluate.py -a HLA-A0201 -s test/HLA-A0201.h5 -d data/production/curated_training_data.csv
```

### Installation ###

A `pip` installable version of MHCnuggets is still under construction.
We should have it up soon, stay tuned!

**Required pacakges:**

* numpy
* scipy
* scikit-learn
* tensorflow
* keras

You might want to check if the Keras backend is configured to use
the Tensforflow backend.