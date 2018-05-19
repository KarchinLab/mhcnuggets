'''
Predict IC50s for a batch of peptides
using a trained model

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

from __future__ import print_function
import numpy as np
from mhcnuggets.src.models import get_predictions, mhcnuggets_lstm
from mhcnuggets.src.dataset import Dataset, mask_peptides, cut_pad_peptides
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


def predict(class_, peptides_path, mhc,
            model='lstm', weights_path=None, output=None):
    '''
    Prediction protocol
    '''

    # read peptides
    peptides = [p.strip() for p in open(peptides_path)]

    # set the length
    if class_.upper() == 'I':
        mask_len = MHCI_MASK_LEN
    elif class_.upper() == 'II':
        mask_len = MHCII_MASK_LEN

    print('Predicting for %d peptides' % (len(peptides)))

    # apply cut/pad or mask to same length
    if 'lstm' in model or 'gru' in model:
        normed_peptides = mask_peptides(peptides, max_len=mask_len)
    else:
        normed_peptides = cut_pad_peptides(peptides)

    # get tensorized values for prediction
    peptides_tensor = tensorize_keras(normed_peptides, embed_type='softhot')

    # make model
    print('Building model')
    # define model
    if model == 'lstm':
        model = mhcnuggets_lstm(input_size=(mask_len, NUM_AAS))

    if weights_path:
        model.load_weights(weights_path)
    else:
        if class_.upper() == 'I':
            predictor_mhc = closest_mhcI(mhc)
        elif class_.upper() == 'II':
            predictor_mhc = closest_mhcII(mhc)

        print("Closest allele found", predictor_mhc)
        model.load_weights(os.path.join(MHCNUGGETS_HOME, "saves",
                                        "production",
                                        predictor_mhc + '.h5'))

    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    # test model
    preds_continuous, preds_binary = get_predictions(peptides_tensor, model)
    ic50s = [map_proba_to_ic50(p[0]) for p in preds_continuous]

    # write out results
    if output:
        filehandle = open(output, 'w')
    else:
        filehandle = sys.stdout

    print(','.join(('peptide', 'ic50')), file=filehandle)
    for i, peptide in enumerate(peptides):
        print(','.join((peptide, str(ic50s[i]))), file=filehandle)


def parse_args():
    '''
    Parse user arguments
    '''

    info = 'Predict IC50 for a batch of peptides using a trained model'
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-m', '--model',
                        type=str, default='lstm',
                        help=('Type of MHCnuggets model used to predict' +
                              'options are just lstm for now'))

    parser.add_argument('-c', '--class',
                        type=str, required=True,
                        help='MHC class - options are I or II')

    parser.add_argument('-w', '--weights',
                        type=str, default=None,
                        help=('Path to weights of the model useful if' +
                              'using user trained models'))

    parser.add_argument('-p', '--peptides',
                        type=str, required=True,
                        help='New line separated list of peptides')

    parser.add_argument('-a', '--allele',
                        type=str, default=None,
                        help = 'Allele used for prediction')

    parser.add_argument('-o', '--output',
                        type=str, default=None,
                        help=('Path to output file, if None, ' +
                              'output is written to stdout'))

    args = parser.parse_args()
    return vars(args)


def main():
    '''
    Main function
    '''

    opts = parse_args()
    predict(model=opts['model'], class_=opts['class'],
            weights_path=opts['weights'],
            peptides_path=opts['peptides'],
            mhc=opts['allele'], output=opts['output'])


if __name__ == '__main__':
    main()
