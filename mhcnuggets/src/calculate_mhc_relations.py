'''
Calculate MHC relations for
transfer learning

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

from __future__ import print_function
from mhcnuggets.src.dataset import Dataset
import numpy as np
import os
from mhcnuggets.src.models import get_predictions
import mhcnuggets.src.models
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from scipy.stats import kendalltau
from keras.optimizers import Adam, SGD
import argparse
import cPickle as pickle
import pandas as pd
from mhcnuggets.src.aa_embeddings import NUM_AAS, MHCI_MASK_LEN, MHCII_MASK_LEN


def calculate_relation(mhc, data, model, weights_dir, mass_spec, rand_negs, ic50_threshold, max_ic50, binary=False, embed_peptides=False):
    '''
    Training protocol
    '''

    print('Calculating tuning MHC for %s' % mhc)

    relations_dict = {}

    # get the allele specific data
    mhc_data, num_positives, num_random_negatives, num_real_negatives = data.get_allele(mhc, mass_spec, rand_negs, ic50_threshold)
    train_peptides, train_continuous, train_binary = mhc_data.tensorize_keras(embed_type='softhot')
    best_mhc = ''
    best_auc = 0
    best_f1 = 0
    best_ppv_top = 0
    
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
        preds_continuous, preds_binary = get_predictions(train_peptides, model, binary, embed_peptides, ic50_threshold, max_ic50)

        try:
            auc = roc_auc_score(train_binary, preds_continuous)
            f1 = f1_score(train_binary, preds_binary)
                #make preds_continuous, test_binary and preds_binary into a matrix, sort by preds_continous, do predicion on the top npos rows only
            raveled_preds_continuous = np.array(preds_continuous, dtype='float32').ravel()
            np_lists = np.array([raveled_preds_continuous, preds_binary, train_binary])
            columns = ['pred_cont','pred_bin','true_bin']
            dframe = pd.DataFrame(np_lists.T,columns=columns)
            dframe.sort_values('pred_cont',inplace=True, ascending=False)
            dframe_head = dframe.head(num_positives)
            sorted_pred_cont = dframe_head['pred_cont'].tolist()
            sorted_pred_bin = dframe_head['pred_bin'].tolist()
            sorted_true_bin = dframe_head['true_bin'].tolist()
            ppv_top = precision_score(sorted_true_bin, sorted_pred_bin, pos_label=1)

            #print ('MHC: %s, AUC: %.4f, F1: %.4f, KTAU: %.4f' % (tuning_mhc,
            #                                                     auc,
            #                                                     f1,
            #                                                     ktau))
            if auc > best_auc:
                best_auc_mhc = tuning_mhc
                best_auc = auc
            if f1 > best_f1:
                best_f1_mhc = tuning_mhc
                best_f1 = f1
            if ppv_top > best_ppv_top:
                best_ppv_top_mhc = tuning_mhc
                best_ppv_top = ppv_top
            
            adata, num_pos, num_rand_neg, num_real_neg = data.get_allele(tuning_mhc,mass_spec,rand_negs, ic50_threshold)
            num_tuning_mhc = len(adata.peptides)

        except ValueError:
            continue

    return best_auc_mhc, best_auc, best_f1_mhc, best_f1, best_ppv_top_mhc, best_ppv_top, num_mhc, num_tuning_mhc


def parse_args():
    '''
    Parse user arguments
    '''

    info = 'Calculate MHC tuning relations for given data'
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-d', '--data',
                        type=str, default='data/production/curated_training_data.csv',
                        help='Path to data file')

    parser.add_argument('-m', '--model',
                        type=str, required=False, default='lstm',
                        help=('Neural network architecture'))

    parser.add_argument('-w', '--weights',
                        type=str, required=True,
                        help='Path to saved weights per allele')

    parser.add_argument('-c', '--class',
                        type=str, required=True,
                        help='MHC class - options are I or II')

    parser.add_argument('-a', '--allele',
                        type=str, required=True,
                        help='Allele to calculate tuning for')

    parser.add_argument('-s', '--save_file',
                        type=str, required=True,
                        help='File to which to write the tuning result to')

    parser.add_argument('-e', '--mass_spec', 
                        required=True, default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Train on mass spec data if True, binding affinity data if False')

    parser.add_argument('-r', '--random_negs', 
                        required=True, default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Random negative examples included in training if True')

    parser.add_argument('-l', '--ic50_threshold',
                        type=int, default=500,
                        help='Threshold on ic50 (nM) that separates binder/non-binder')

    parser.add_argument('-x', '--max_ic50',
                        type=int, default=50000,
                        help='Maximum ic50 value')

    parser.add_argument('-q', '--embed_peptides',
                        type=bool, default=False,
                        help='Embedding of peptides used')


    parser.add_argument('-B', '--binary_predictions',
                        type=bool, default=False,
                        help='Binary predictions used')

    

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
                            sep=',', ic50_threshold=opts['ic50_threshold'],
                            max_ic50=opts['max_ic50'],
                            allele_column_name='mhc',
                            peptide_column_name='peptide',
                            affinity_column_name='IC50(nM)',
                            type_column_name='measurement_type',
                            source_column_name='measurement_source'
                             )


    if opts['class'] == 'I':
        data.mask_peptides(max_len=MHCI_MASK_LEN)
        input_size=(MHCI_MASK_LEN, NUM_AAS)
    if opts['class'] == 'II':
        data.mask_peptides(max_len=MHCII_MASK_LEN)
        input_size=(MHCII_MASK_LEN, NUM_AAS)

    # create the model
    model = models.mhcnuggets_lstm(input_size)
       
    # find ideal tuning allele
    best_auc_mhc, best_auc, best_f1_mhc, best_f1, best_ppv_top_mhc, best_ppv_top, num_mhc, num_tuning_mhc = \
                                                                     calculate_relation(opts['allele'], data, model,
                                                                     opts['weights'], opts['mass_spec'],
                                                                     opts['random_negs'], opts['ic50_threshold'],
                                                                     opts['max_ic50'],opts['embed_peptides'])

    print('Tuning result best AUC_MHC, AUC, F1_MHC, F1, PPV_TOP_MHC, PPV_TOP:', 
          best_auc_mhc, best_auc, best_f1_mhc, best_f1, best_ppv_top_mhc, best_ppv_top, num_mhc, num_tuning_mhc)

#REWRITE TO CONSIDER ALL
    # factor for the tuning to be valid
    # if best_auc > 0.9 and num_tuning_mhc > num_mhc:
    #     out_file = open(opts['save_file'], 'a')
    #     out_file.write(','.join((opts['allele'], best_auc_mhc, str(best_auc), str(num_mhc), str(num_tuning_mhc))) + '\n')

    #accept the tuning model if it has better PPV_top and more training examples 
    if best_ppv_top > 0.8 and num_tuning_mhc > num_mhc:
        out_file = open(opts['save_file'], 'a')
        out_file.write(','.join((opts['allele'], best_ppv_top_mhc, str(best_ppv_top), str(num_mhc), str(num_tuning_mhc))) + '\n')
        out_file.close()


if __name__ == '__main__':
    main()
