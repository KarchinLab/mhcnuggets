'''
Calculate MHC relations for
transfer learning

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

from __future__ import print_function
from dataset import Dataset
import numpy as np
import os
from models import get_predictions
import models
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from scipy.stats import kendalltau
from keras.optimizers import Adam, SGD
import argparse
import cPickle as pickle


def calculate_relation(mhc, data, model, weights_dir):
    '''
    Tuning protocol
    '''

    print('Calculating tuning MHC for %s' % mhc)

    relations_dict = {}

    # get the allele specific data
    mhc_data = data.get_allele(mhc)
    train_peptides, train_continuous, train_binary = mhc_data.tensorize_keras(embed_type='softhot')
    best_mhc = ''
    best_auc = 0
    num_mhc = len(mhc_data.peptides)

    for tuning_mhc in sorted(set(data.alleles)):

        # don't want to tune with ourselves
        if mhc == tuning_mhc:
            continue

        # define the path to save weights
        try:
            model_path = os.path.join(weights_dir, tuning_mhc + '.h5')
            model.load_weights(model_path)
        except IOError:
            continue
        preds_continuous, preds_binary = get_predictions(train_peptides, model)

        try:
            auc = roc_auc_score(train_binary, preds_continuous)
            if auc > best_auc:
                best_mhc = tuning_mhc
                best_auc = auc
                num_tuning_mhc = len(data.get_allele(tuning_mhc).peptides)
        except ValueError:
            continue

    return best_mhc, best_auc, num_mhc, num_tuning_mhc


def parse_args():
    '''
    Parse user arguments
    '''

    info = "Calculate MHC tuning relations for given data"
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-d', '--data',
                        type=str, default='data/production/curated_training_data.csv',
                        help='Path to data file')

    parser.add_argument('-m', '--model',
                        type=str, required=True,
                        help=('Model we are transfer learning for. Options ' +
                              'are fc, gru, lstm, chunky_cnn, spanny_cnn'))

    parser.add_argument('-w', '--weights_dir',
                        type=str, required=True,
                        help='Path to saved weights per allele')

    parser.add_argument('-a', '--allele',
                        type=str, required=True,
                        help='Allele to calculating tuning for')

    parser.add_argument('-s', '--save_file',
                        type=str, required=True,
                        help='File to which to append the tuning result to')

    args = parser.parse_args()
    return vars(args)


def main():
    '''
    Main function
    '''

    opts = parse_args()
    model = opts['model']

    # load training data
    data = Dataset.from_csv(filename=opts['data'],
                            sep=',',
                            allele_column_name='mhc',
                            peptide_column_name='peptide',
                            affinity_column_name='IC50(nM)')

    if 'gru' or 'lstm' in model:
        data.mask_peptides()
    else:
        data.cut_pad_peptides()

    # create the model
    if model == 'lstm':
        model = models.mhcnuggets_lstm()

    # find ideal tuning allele
    best_mhc, best_auc, num_mhc, num_tuning_mhc = calculate_relation(opts['allele'], data, model, opts['weights_dir'])

    print('Tuning result best MHC, AUC:', best_mhc, best_auc, num_mhc, num_tuning_mhc)

    # factor for the tuning to be valid
    if best_auc > 0.9 and num_tuning_mhc > num_mhc:
        out_file = open(opts['save_file'], 'a')
        out_file.write(','.join((opts['allele'], best_mhc, str(best_auc), str(num_mhc), str(num_tuning_mhc))) + '\n')
        out_file.close()


if __name__ == '__main__':
    main()
