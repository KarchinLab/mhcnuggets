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
import pandas as pd
import os
import math  #required for Pycharm
from mhcnuggets.src.models import get_predictions, mhcnuggets_lstm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from keras.optimizers import Adam
import argparse
from mhcnuggets.src.aa_embeddings import NUM_AAS
from mhcnuggets.src.aa_embeddings import MHCI_MASK_LEN, MHCII_MASK_LEN


def train(class_, data, mhc, save_path, n_epoch, model='lstm',
          lr=0.001, transfer_path=None, mass_spec=False, ic50_threshold=500, max_ic50=50000):
    '''
    Training protocol
    '''
    # store model name
    model_name = model

    # print out options
    print('Training\nMHC: %s\nData: %s\nModel: %s\nSave path: %s\nTransfer: %s\nMassSpec: %s' %
          (mhc, data, model, save_path, transfer_path, mass_spec))

    # load training
    train_data = Dataset.from_csv(filename=data, ic50_threshold=ic50_threshold, max_ic50=max_ic50,
                                  sep=',', 
                                  allele_column_name='mhc',
                                  peptide_column_name='peptide',
                                  affinity_column_name='IC50(nM)',
                                  type_column_name='measurement_type',
                                  source_column_name='measurement_source'
                                  )

    # set the length
    if class_.upper() == 'I':
        mask_len = MHCI_MASK_LEN
    elif class_.upper() == 'II':
        mask_len = MHCII_MASK_LEN

    train_data.mask_peptides(max_len=mask_len)

    # get the allele specific data 
    mhc_train, n_pos, n_rand_neg, n_real_neg = train_data.get_allele(mhc, mass_spec, ic50_threshold)

    """
    #calculate the composition of the actual training set that will be used
    print('Training on %d peptides' % len(mhc_train.peptides))

    print(str(n_pos) + ' positives ')
    print(str(n_real_neg)  + ' real_negatives ')
    if n_real_neg != 0:
        real_skew = math.fabs(math.log((float(n_pos) / float(n_real_neg))))
    else:
        real_skew = "ND"
    print(str(real_skew) + ' real skew')
    print(str(n_rand_neg) + ' random negatives ')
    n_all_neg = n_real_neg + n_rand_neg
    if n_real_neg + n_rand_neg != 0:
        total_skew = math.fabs(math.log((float(n_pos) / float(n_all_neg))))    #including random negs
    else:
        total_skew = "ND"
    print(str(total_skew) + 'total skew after random negs added')
    """

    # define model
    input_size = (mask_len, NUM_AAS)
    model = mhcnuggets_lstm(input_size)

    # check if we need to do transfer learning
    if transfer_path:
        model.load_weights(transfer_path)

    #select appropriate loss function for binding affinity data (continuous) or mass spec data (binary)    
    if mass_spec:
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
    else:
        model.compile(loss='mse', optimizer=Adam(lr=0.001))

    # convergence criterion
#    highest_f1 = -1
    highest_ppv_top = -1

    # get tensorized values of the whole dataset for epoch training and for testing
    train_peptides, train_continuous, train_binary = mhc_train.tensorize_keras(embed_type='softhot')

    for epoch in range(n_epoch):
        # train
        model.fit(train_peptides, train_continuous, epochs=1, verbose=0)
        # test model on training data
        train_preds_cont, train_preds_bin = get_predictions(train_peptides, model)
        train_auc = roc_auc_score(train_binary, train_preds_cont)
        train_f1 = f1_score(train_binary, train_preds_bin)
        train_ktau = kendalltau(train_continuous, train_preds_cont)[0]
        raveled_train_preds_cont = np.array(train_preds_cont, dtype='float32').ravel()
        train_pearsonr = pearsonr(train_continuous, raveled_train_preds_cont)[0]
        train_ppv = precision_score(train_binary, train_preds_bin, pos_label=1)
        #make train_preds_cont, train_binary and train_preds_bin into a matrix, sort by train_preds_cont, do predicion on the top npos rows only
        np_lists = np.array([raveled_train_preds_cont, train_preds_bin, train_binary])
        columns = ['pred_cont','pred_bin','true_bin']
        dframe = pd.DataFrame(np_lists.T,columns=columns)
        dframe.sort_values('pred_cont',inplace=True, ascending=False)
        dframe_head = dframe.head(n_pos)
        sorted_pred_cont = dframe_head['pred_cont'].tolist()
        sorted_pred_bin = dframe_head['pred_bin'].tolist()
        sorted_true_bin = dframe_head['true_bin'].tolist()
        train_ppv_top = precision_score(sorted_true_bin, sorted_pred_bin, pos_label=1)

        print('epoch %d / %d' % (epoch, n_epoch))

        print('Num pos: %.4f\nTrain AUC: %.4f, F1: %.4f, KTAU: %.4f, PCC: %.4f, PPV: %.4f, PPVtop: %.4f' %
              (n_pos, train_auc, train_f1, train_ktau, train_pearsonr, train_ppv, train_ppv_top))

       # convergence
        if train_ppv_top > highest_ppv_top:
            highest_ppv_top = train_ppv_top
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

    parser.add_argument('-n', '--num_epoch',
                        type=int, required=True,
                        help=('Number of epochs to train for, see README.md ' +
                              'for recommended numbers'))

    parser.add_argument('-y', '--learning_rate',
                        type=float, default=0.001,
                        help='Learning rate')

    parser.add_argument('-t', '--transfer_weights',
                        type=str, default=None,
                        help='Path to transfer weights from')

    parser.add_argument('-s', '--save_path',
                        type=str, required=True,
                        help=('Path to which the model is saved'))

    parser.add_argument('-e', '--mass_spec', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Train on mass spec data if True, binding affinity data if False')

    parser.add_argument('-l', '--ic50_threshold',
                        type=int, default=500,
                        help='Threshold on ic50 (nM) that separates binder/non-binder')

    parser.add_argument('-x', '--max_ic50',
                        type=int, default=50000,
                        help='Maximum ic50 value')

    

    args = parser.parse_args()
    return vars(args)



def main():
    '''
    Main function
    '''

    opts = parse_args()
    train(class_=opts['class'], data=opts['data'], mhc=opts['allele'], 
                 save_path=opts['save_path'], n_epoch=opts['num_epoch'], 
                 model=opts['model'], lr=opts['learning_rate'],
                 transfer_path=opts['transfer_weights'],
                 mass_spec=opts['mass_spec'], ic50_threshold=opts['ic50_threshold'],
                 max_ic50=opts['max_ic50'])


if __name__ == '__main__':
    main()
