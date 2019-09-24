'''
Script to evaluate a MHCnuggets model
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


def test_all(class_, data, model_path_dir, model='lstm', mass_spec=False, ic50_threshold=500, max_ic50=50000):
    '''
    Evaluation protocol
    '''

    # print out options
    print('Testing\n: Data: %s\nModel: %s\nSave path: %s\nMass spec: %s\nIC50 threshold: %s\nMax IC50: %s\n' %
    (data, model, model_path_dir, mass_spec, ic50_threshold, max_ic50))

    # load training
    test_data = Dataset.from_csv(filename=data, ic50_threshold=ic50_threshold, max_ic50=max_ic50, 
                                 sep=',', 
                                 allele_column_name='mhc',
                                 peptide_column_name='peptide',
                                 affinity_column_name='IC50(nM)',
                                 type_column_name='measurement_type',
                                 source_column_name='measurement_source')

    # define model
    if class_.upper() == 'I':
        mask_len = MHCI_MASK_LEN
        input_size=(MHCI_MASK_LEN, NUM_AAS)
    elif class_.upper() == 'II':
        mask_len = MHCII_MASK_LEN
        input_size=(MHCII_MASK_LEN, NUM_AAS)

    model = mhcnuggets_lstm(input_size)
    test_data.mask_peptides(max_len=mask_len)

    all_preds_continuous = []
    all_test_continuous = []
    all_test_binary = []
    all_preds_binary = []
    all_npos = 0
    
    for mhc in sorted(set(test_data.alleles)):

   # get the allele specific data
        mhc_test, npos, nrandneg, nrealneg = test_data.get_allele(mhc, mass_spec, ic50_threshold, length=None) 

        # compile model
        model.load_weights(model_path_dir + '/' + mhc + '.h5')

        if mass_spec:
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
        else:
            model.compile(loss='mse', optimizer=Adam(lr=0.001))


        # get tensorized values for training
        test_peptides, test_continuous, test_binary = mhc_test.tensorize_keras(embed_type='softhot')

        # test
        preds_continuous, preds_binary = get_predictions(test_peptides, model, ic50_threshold=ic50_threshold, max_ic50=max_ic50)
        try:
            test_auc = roc_auc_score(test_binary, preds_continuous)
            test_f1 = f1_score(test_binary, preds_binary)
            test_ktau = kendalltau(test_continuous, preds_continuous)[0]
            raveled_preds_continuous = np.array(preds_continuous, dtype='float32').ravel()
            test_pearsonr = pearsonr(test_continuous, raveled_preds_continuous)[0]
            test_ppv = precision_score(test_binary, preds_binary, pos_label=1)
            #make preds_continuous, test_binary and preds_binary into a matrix, sort by preds_continous, do predicion on the top npos rows only
            np_lists = np.array([raveled_preds_continuous, preds_binary, test_binary])
            columns = ['pred_cont','pred_bin','true_bin']
            dframe = pd.DataFrame(np_lists.T,columns=columns)
            dframe.sort_values('pred_cont',inplace=True, ascending=False)
            dframe_head = dframe.head(npos)
            sorted_pred_cont = dframe_head['pred_cont'].tolist()
            sorted_pred_bin = dframe_head['pred_bin'].tolist()
            sorted_true_bin = dframe_head['true_bin'].tolist()
            test_ppv_top = precision_score(sorted_true_bin, sorted_pred_bin, pos_label=1)
            print('%s AUC: %.4f, F1: %.4f, KTAU: %.4f, PCC: %.4f, PPV: %.4f, PPVtop: %.4f' %
                  (mhc, test_auc, test_f1, test_ktau, test_pearsonr, test_ppv, test_ppv_top))
            all_preds_continuous += list(preds_continuous)
            all_test_continuous += list(test_continuous)
            all_test_binary += list(test_binary)
            all_preds_binary += list(preds_binary)
            all_npos += npos
            
        except:
            print(mhc, "Only one class in y_true")


    #calculations needed to get overall performance metrics
    all_raveled_preds_continuous = np.array(all_preds_continuous, dtype='float32').ravel()
    all_np_lists = np.array([all_raveled_preds_continuous, all_preds_binary, all_test_binary])
    columns = ['pred_cont','pred_bin','true_bin']
    all_dframe = pd.DataFrame(all_np_lists.T,columns=columns)
    all_dframe.sort_values('pred_cont',inplace=True, ascending=False)
    all_dframe_head = all_dframe.head(all_npos)
    all_sorted_pred_cont = all_dframe_head['pred_cont'].tolist()
    all_sorted_pred_bin = all_dframe_head['pred_bin'].tolist()
    all_sorted_true_bin = all_dframe_head['true_bin'].tolist()
    
    
    print("Overall AUC %.4f" % (roc_auc_score(all_test_binary, all_preds_continuous)))
    print("Overall F1 %.4f" % (f1_score(all_test_binary, all_preds_binary)))
    print("Overall KTAU %.4f" % (kendalltau(all_test_continuous, all_preds_continuous)[0]))
    print("Overall PCC %.4f" % (pearsonr(all_test_continuous, all_raveled_preds_continuous)[0]))
    print("Overall PPV %.4f" % (precision_score(all_test_binary, all_preds_binary, pos_label=1)))
    print("Overall PPVtop %.4f" % (precision_score(all_sorted_true_bin, all_sorted_pred_bin, pos_label=1)))
    
    
def parse_args():
    '''
    Parse user arguments
    '''

    info = 'Evaluate a MHCnuggets model on given data'
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-d', '--data',
                        type=str, default='data/production/curated_training_data.csv',
                        help=('Data file to use for evaluation, should ' +
                              'be csv files w/ formatting similar to ' +
                              'data/production/curated_training_data.csv'))

    parser.add_argument('-m', '--model',
                        type=str, default='lstm',
                        help=('Type of MHCnuggets model to evaluate' +
                              'options are just lstm for now'))

    parser.add_argument('-c', '--class',
                        type=str, required=True,
                        help='MHC class - options are I or II')

    parser.add_argument('-s', '--save_path_dir',
                        type=str, required=True,
                        help=('Path to the directory where the model weights are saved'))

    parser.add_argument('-e', '--mass_spec', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Train on mass spec data if True, binding affinity data if False')

    parser.add_argument('-l', '--ic50_threshold',
                        type=int, default=500,
                        help='Threshold on ic50 (nM) that separates binder/non-binder')

    parser.add_argument('-x', '--max_ic50',
                        type=int, default=50000,
                        help='Maximum ic50 value')


    parser.add_argument('-n', '--length',
                        type=int, default=-9999,
                        help='Peptide length')

        
    args = parser.parse_args()
    return vars(args)


def main():
    '''
    Main function
    '''

    opts = parse_args()
    if opts['length'] == -9999:  # don't limit evaluation to peptides of a specific length
        test_all(data=opts['data'],
                 model=opts['model'], class_=opts['class'],
                 model_path_dir=opts['save_path_dir'], mass_spec=opts['mass_spec'],
                 ic50_threshold=opts['ic50_threshold'],
                 max_ic50=opts['max_ic50'])

        
if __name__ == '__main__':
    main()
