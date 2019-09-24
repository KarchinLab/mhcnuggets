'''
Predict IC50s for a batch of peptides
using a trained model

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

from __future__ import print_function
import numpy as np
from mhcnuggets.src.models import get_predictions, mhcnuggets_lstm
from mhcnuggets.src.dataset import mask_peptides
from mhcnuggets.src.dataset import tensorize_keras, map_proba_to_ic50

from keras.optimizers import Adam
import argparse
from mhcnuggets.src.find_closest_mhcI import closest_allele as closest_mhcI
from mhcnuggets.src.find_closest_mhcII import closest_allele as closest_mhcII

import os
import sys
MHCNUGGETS_HOME = os.path.join(os.path.dirname(__file__), '..')
from mhcnuggets.src.aa_embeddings import NUM_AAS
from mhcnuggets.src.aa_embeddings import MHCI_MASK_LEN, MHCII_MASK_LEN


def predict(class_, peptides_path, mhc, pickle_path='data/production/examples_per_allele.pkl',
            model='lstm', model_weights_path="saves/production/", output=None,
            mass_spec=False, ic50_threshold=500, max_ic50=50000, embed_peptides=False,
            binary_preds=False, ba_models=False):
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

    # write out results
    if output:
        filehandle = open(output, 'w')
    else:
        filehandle = sys.stdout

    print(','.join(('peptide', 'ic50')), file=filehandle)
    for i, peptide in enumerate(original_peptides):
        print(','.join((peptide, str(round(ic50s[i],2)))), file=filehandle)



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
                        type=bool, default=False,
                        help='Embedding of peptides used')

    parser.add_argument('-B', '--binary_predictions',
                        type=bool, default=False,
                        help='Binary prediction')

    parser.add_argument('-M', '--ba_models', default=False, type=bool,
                        help='Use binding affinity trained models only instead of mass spec trained models')

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
            binary_preds=opts['binary_predictions'],ba_models=opts['ba_models'])


if __name__ == '__main__':
    main()
