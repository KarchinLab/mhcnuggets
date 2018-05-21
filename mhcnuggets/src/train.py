'''
Script to train a MHCnuggets model
of choice

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

# imports
from __future__ import print_function
from mhcnuggets.src.dataset import Dataset
import numpy as np
import os
from mhcnuggets.src.models import get_predictions, mhcnuggets_lstm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from scipy.stats import kendalltau
from keras.optimizers import Adam
import argparse
from mhcnuggets.src.aa_embeddings import NUM_AAS
from mhcnuggets.src.aa_embeddings import MHCI_MASK_LEN, MHCII_MASK_LEN


def train(class_, data, mhc, save_path, n_epoch,
          model='lstm', lr=0.001, transfer_path=None):
    '''
    Training protocol
    '''

    # store model name
    model_name = model

    # print out options
    print('Training\nMHC: %s\nData: %s\nModel: %s\nSave path: %s\nTransfer: %s' %
          (mhc, data, model, save_path, transfer_path))

    # load training
    train_data = Dataset.from_csv(filename=data,
                                  sep=',',
                                  allele_column_name='mhc',
                                  peptide_column_name='peptide',
                                  affinity_column_name='IC50(nM)')

    # set the length
    if class_.upper() == 'I':
        mask_len = MHCI_MASK_LEN
    elif class_.upper() == 'II':
        mask_len = MHCII_MASK_LEN

    # apply cut/pad or mask to same length
    if 'lstm' in model or 'gru' in model or 'attn' in model:
        train_data.mask_peptides(max_len=mask_len)
    else:
        train_data.cut_pad_peptides()

    # get the allele specific data
    mhc_train = train_data.get_allele(mhc)

    print('Training on %d peptides' % len(mhc_train.peptides))

    # define model
    if model == 'lstm':
        model = mhcnuggets_lstm(input_size=(mask_len, NUM_AAS))

    # check if we need to do transfer learning
    if transfer_path:
        model.load_weights(transfer_path)

    # compile model
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    # get tensorized values for training
    train_peptides, train_continuous, train_binary = mhc_train.tensorize_keras(embed_type='softhot')

    # convergence criterion
    highest_f1 = -1

    for epoch in range(n_epoch):

        # train
        model.fit(train_peptides, train_continuous, epochs=1, verbose=0)
        # test model on training data
        train_preds_cont, train_preds_bin = get_predictions(train_peptides, model)
        train_auc = roc_auc_score(train_binary, train_preds_cont)
        train_f1 = f1_score(train_binary, train_preds_bin)
        train_ktau = kendalltau(train_continuous, train_preds_cont)[0]
        print('epoch %d / %d' % (epoch, n_epoch))
        print('Train AUC: %.4f, F1: %.4f, KTAU: %.4f' %
              (train_auc, train_f1, train_ktau))

        # convergence
        if train_f1 > highest_f1:

            highest_f1 = train_f1
            best_epoch = epoch
            model.save_weights(save_path)

    print('Done!')


def parse_args():
    '''
    Parse user arguments
    '''

    info = 'Train a MHCnuggets model on given data'
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-d', '--data',
                        type=str, default='data/production/curated_training_data.csv',
                        help=('Data file to use for training, should ' +
                              'be csv files w/ formatting similar to ' +
                              'data/production/curated_training_data.csv'))

    parser.add_argument('-a', '--allele',
                        type=str, required=True,
                        help='MHC allele to train on')

    parser.add_argument('-m', '--model',
                        type=str, default='lstm',
                        help=('Type of MHCnuggets model to train' +
                              'options are just lstm for now'))

    parser.add_argument('-c', '--class',
                        type=str, required=True,
                        help='MHC class - options are I or II')

    parser.add_argument('-s', '--save_path',
                        type=str, required=True,
                        help=('Path to which the model is saved'))

    parser.add_argument('-n', '--num_epoch',
                        type=int, required=True,
                        help=('Number of epochs to train for, see README.md ' +
                              'for recommended numbers'))

    parser.add_argument('-l', '--learning_rate',
                        type=float, default=0.001,
                        help='Learning rate')

    parser.add_argument('-t', '--transfer_weights',
                        type=str, default=None,
                        help='Path to transfer weights from')

    args = parser.parse_args()
    return vars(args)


def main():
    '''
    Main function
    '''

    opts = parse_args()
    train(mhc=opts['allele'], data=opts['data'], model=opts['model'],
          class_=opts['class'], save_path=opts['save_path'],
          lr=opts['learning_rate'],
          n_epoch=opts['num_epoch'], transfer_path=opts['transfer_weights'])


if __name__ == '__main__':
    main()
