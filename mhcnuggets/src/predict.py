'''
Predict IC50s for a batch of peptides
using a trained model

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

from __future__ import print_function
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
from mhcnuggets.src.models import get_predictions, mhcnuggets_lstm
from mhcnuggets.src.dataset import mask_peptides
from mhcnuggets.src.dataset import tensorize_keras, map_proba_to_ic50

try:
    from keras.optimizers import Adam
except:
    from tensorflow.keras.optimizers import Adam
import argparse
from mhcnuggets.src.find_closest_mhcI import closest_allele as closest_mhcI
from mhcnuggets.src.find_closest_mhcII import closest_allele as closest_mhcII

import os
import sys
import math
MHCNUGGETS_HOME = os.path.join(os.path.dirname(__file__), '..')
from mhcnuggets.src.aa_embeddings import NUM_AAS
from mhcnuggets.src.aa_embeddings import MHCI_MASK_LEN, MHCII_MASK_LEN


def predict(class_, peptides_path, mhc, pickle_path='data/production/examples_per_allele.pkl',
            model='lstm', model_weights_path="saves/production/", output=None,
            mass_spec=False, ic50_threshold=500, max_ic50=50000, embed_peptides=False,
            binary_preds=False, ba_models=False, rank_output=False,
            hp_ic50s_cI_pickle_path='data/production/mhcI/hp_ic50s_cI.pkl',
            hp_ic50s_positions_cI_pickle_path='data/production/mhcI/hp_ic50s_positions_cI.pkl',
            hp_ic50s_hp_lengths_cI_pickle_path='data/production/mhcI/hp_ic50s_hp_lengths_cI.pkl',
            hp_ic50s_first_percentiles_cI_pickle_path='data/production/mhcI/hp_ic50s_first_percentiles_cI.pkl',
            hp_ic50s_cII_pickle_path='data/production/mhcII/hp_ic50s_cII.pkl',
            hp_ic50s_positions_cII_pickle_path='data/production/mhcII/hp_ic50s_positions_cII.pkl',
            hp_ic50s_hp_lengths_cII_pickle_path='data/production/mhcII/hp_ic50s_hp_lengths_cII.pkl',
            hp_ic50s_first_percentiles_cII_pickle_path='data/production/mhcII/hp_ic50s_first_percentiles_cII.pkl'):
    '''
    Prediction protocol
    '''
    # read peptides
    peptides = [p.strip() for p in open(peptides_path)]

    # set the length
    if class_.upper() == 'I':
        mask_len = MHCI_MASK_LEN
        input_size=(MHCI_MASK_LEN, NUM_AAS)
    elif class_.upper() == 'II':
        mask_len = MHCII_MASK_LEN
        input_size=(MHCII_MASK_LEN, NUM_AAS)

    print('Predicting for %d peptides' % (len(peptides)))

    # apply cut/pad or mask to same length
    normed_peptides, original_peptides = mask_peptides(peptides, max_len=mask_len)
    # get tensorized values for prediction
    peptides_tensor = tensorize_keras(normed_peptides, embed_type='softhot')

    # make model
    print('Building model')
    model = mhcnuggets_lstm(input_size)
    if class_.upper() == 'I':
        predictor_mhc = closest_mhcI(mhc,pickle_path)
    elif class_.upper() == 'II':
        predictor_mhc = closest_mhcII(mhc,pickle_path)
    print("Closest allele found", predictor_mhc)

    if model_weights_path != "saves/production/":
        print('Predicting with user-specified model: ' + model_weights_path)
        model.load_weights(model_weights_path)
    elif ba_models:
        print('Predicting with only binding affinity trained models')
        model.load_weights(os.path.join(MHCNUGGETS_HOME,model_weights_path,predictor_mhc+'_BA.h5'))
    elif os.path.isfile(os.path.join(MHCNUGGETS_HOME,model_weights_path,predictor_mhc+'_BA_to_HLAp.h5')):
        print('BA_to_HLAp model found, predicting with BA_to_HLAp model...')
        model.load_weights(os.path.join(MHCNUGGETS_HOME,model_weights_path,predictor_mhc+'_BA_to_HLAp.h5'))
    else:
        print ('No BA_to_HLAp model found, predicting with BA model...')
        model.load_weights(os.path.join(MHCNUGGETS_HOME,model_weights_path,predictor_mhc+'_BA.h5'))

    if mass_spec:
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
    else:
        model.compile(loss='mse', optimizer=Adam(lr=0.001))

    # test model
    preds_continuous, preds_binary = get_predictions(peptides_tensor, model, binary_preds, embed_peptides, ic50_threshold, max_ic50)
    ic50s = [map_proba_to_ic50(p[0], max_ic50) for p in preds_continuous]

    if (rank_output):
        print("Rank output selected, computing peptide IC50 ranks against human proteome peptides...")
        if class_.upper() == 'I':
            hp_ic50_pickle = pickle.load(open(os.path.join(MHCNUGGETS_HOME,
                                                           hp_ic50s_cI_pickle_path), 'rb'))
            ic50_pos_pickle = pickle.load(open(os.path.join(MHCNUGGETS_HOME
                                                            , hp_ic50s_positions_cI_pickle_path), 'rb'))
            hp_lengths_pickle = pickle.load(open(os.path.join(MHCNUGGETS_HOME
                                                              , hp_ic50s_hp_lengths_cI_pickle_path), 'rb'))
            first_percentiles_pickle = pickle.load(open(os.path.join(MHCNUGGETS_HOME
                                                                     , hp_ic50s_first_percentiles_cI_pickle_path), 'rb'))
        elif class_.upper() == 'II':
            hp_ic50_pickle = pickle.load(open(os.path.join(MHCNUGGETS_HOME,
                                                           hp_ic50s_cII_pickle_path), 'rb'))
            ic50_pos_pickle = pickle.load(open(os.path.join(MHCNUGGETS_HOME,
                                                            hp_ic50s_positions_cII_pickle_path), 'rb'))
            hp_lengths_pickle = pickle.load(open(os.path.join(MHCNUGGETS_HOME
                                                              , hp_ic50s_hp_lengths_cII_pickle_path), 'rb'))
            first_percentiles_pickle = pickle.load(open(os.path.join(MHCNUGGETS_HOME
                                                                     , hp_ic50s_first_percentiles_cII_pickle_path), 'rb'))
        ic50_ranks = get_ranks(ic50s,hp_ic50_pickle,hp_lengths_pickle,
                               first_percentiles_pickle,ic50_pos_pickle,predictor_mhc)
        if (output):
            if len(output.split('.')) > 1:
                rank_filehandle = open(''.join(output.split('.')[:-1] + ['_ranks.'] + \
                                               [output.split('.')[-1]]), 'w')
            else:
                rank_filehandle = open(output + '_ranks', 'w')
        else:
            rank_filehandle = sys.stdout

    print("Writing output files...")
    # write out results
    if output:
        filehandle = open(output, 'w')
    else:
        filehandle = sys.stdout

    try:
        print(','.join(('peptide', 'ic50')), file=filehandle)
        for i, peptide in enumerate(original_peptides):
            print(','.join((peptide, str(round(ic50s[i],2)))), file=filehandle)
        if (rank_output):
            print(','.join(('peptide', 'ic50', 'human_proteome_rank')), file=rank_filehandle)
            for i, peptide in enumerate(original_peptides):
                print(','.join((peptide, str(round(ic50s[i],2)), str(round(ic50_ranks[i],4)))), file=rank_filehandle)

    finally:
        if output:
            filehandle.close()



def get_ranks(ic50_list, ic50_pickle, hp_lengths_pickle, first_percentiles_pickle, pos_pickle, mhc):
    """
    Get percentile rank of every ic50 in the given list, when compared to peptides from
    the human proteome.
    """
    rank_list=[]
    first_percentile = first_percentiles_pickle[mhc]
    for ic50 in ic50_list:
        if not math.isnan(ic50):
            if ic50 > first_percentile:
                base_ic50_list = ic50_pickle['downsampled'][mhc]
                closest_ind, exact_match = binary_search(base_ic50_list, 0,
                                                         len(base_ic50_list) - 1,ic50)
                if(exact_match):
                    (first_occ, last_occ) = pos_pickle['downsampled'][mhc][ic50]
                    middle_ind = float(first_occ + last_occ) / 2
                    closest_ind = middle_ind
                percentile=(closest_ind + 1) / float(len(base_ic50_list))
            else:
                base_ic50_list = ic50_pickle['first_percentiles'][mhc]
                hp_length = hp_lengths_pickle[mhc]
                closest_ind, exact_match = binary_search(base_ic50_list, 0,
                                                         len(base_ic50_list) - 1,ic50)
                if(exact_match):
                    (first_occ, last_occ) = pos_pickle['first_percentiles'][mhc][ic50]
                    middle_ind = float(first_occ + last_occ) / 2
                    closest_ind = middle_ind
                percentile=(closest_ind + 1) / float(hp_length)
            rank_list.append(percentile)
        else:
            rank_list.append(float('nan'))
    return rank_list


def binary_search(arr, low, high, x):
    # Check base case
    if high >= low:
        mid = (high + low) // 2

        # If element is present at the middle itself
        if arr[mid] == x:
            exact_match = True
            return float(mid), exact_match

            # If element is smaller than mid, then it can only
            # be present in left subarray
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)

            # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, high, x)
    else:
        # Element is not present in the array
        # Record the position to the left of our value of interest
        exact_match = False
        return float(high), exact_match

def parse_args():
    '''
    Parse user arguments
    '''

    info = 'Predict IC50 for a batch of peptides using a trained model'
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-m', '--model',
                        type=str, default='lstm',
                        help='Type of MHCnuggets model used to predict' +
                              'options are just lstm for now')

    parser.add_argument('-c', '--class',
                        type=str, required=True,
                        help='MHC class - options are I or II')

    parser.add_argument('-s', '--model_weights_path',
                        type=str, required=False, default='saves/production/',
                        help='Path to which the model weights are saved')

    parser.add_argument('-k', '--pickle_path',
                        type=str, required=False, default='data/production/examples_per_allele.pkl',
                        help='Path to which the pickle file is saved')

    parser.add_argument('-p', '--peptides',
                        type=str, required=True,
                        help='New line separated list of peptides')

    parser.add_argument('-a', '--allele',
                        type=str, required=True,
                        help = 'Allele used for prediction')

    parser.add_argument('-e', '--mass_spec', default=False, type=lambda x: (str(x).lower()== 'true'),
                        help='Train on mass spec data if True, binding affinity data if False')

    parser.add_argument('-o', '--output',
                        type=str, default=None,
                        help='Path to output file, if None, ' +
                              'output is written to stdout')

    parser.add_argument('-l', '--ic50_threshold',
                        type=int, default=500,
                        help='Threshold on ic50 (nM) that separates binder/non-binder')

    parser.add_argument('-x', '--max_ic50',
                        type=int, default=50000,
                        help='Maximum ic50 value')

    parser.add_argument('-q', '--embed_peptides',
                        action='store_true', default=False,
                        help='Embedding of peptides used')

    parser.add_argument('-B', '--binary_predictions',
                        action='store_true', default=False,
                        help='Binary prediction')

    parser.add_argument('-M', '--ba_models',
                        action='store_true', default=False,
                        help='Use binding affinity trained models only instead of mass spec trained models')

    parser.add_argument('-r', '--rank_output', type=lambda x: (str(x).lower()== 'true'),
                        default=False,
                        help='Additionally write output files of predicted peptide ic50 binding ' + \
                        'percentiles compared to human proteome peptides')

    args = parser.parse_args()
    return vars(args)


def main():
    '''
    Main function
    '''

    opts = parse_args()
    predict(model=opts['model'], class_=opts['class'],
            peptides_path=opts['peptides'],
            model_weights_path=opts['model_weights_path'], pickle_path=opts['pickle_path'],
            mhc=opts['allele'], output=opts['output'],mass_spec=opts['mass_spec'],
            ic50_threshold=opts['ic50_threshold'],
            max_ic50=opts['max_ic50'], embed_peptides= opts['embed_peptides'],
            binary_preds=opts['binary_predictions'],ba_models=opts['ba_models'],
            rank_output=opts['rank_output'])


if __name__ == '__main__':
    main()
