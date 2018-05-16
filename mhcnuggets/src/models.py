'''
Implementation of MHCnuggets models in
Keras

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
import keras.metrics
from keras.layers.core import Dropout, Flatten, Masking, Reshape, Lambda
from keras.layers.recurrent import LSTM, GRU
import math
from keras.layers import Input
from keras.layers import Conv1D, GlobalMaxPooling1D
from mhcnuggets.src.aa_embeddings import MASK_VALUE
from mhcnuggets.src.aa_embeddings import NUM_AAS, MAX_MASK_LEN
from keras.layers import dot, concatenate
import keras.backend as K

# constants
IC50_THRESHOLD = 500
MAX_IC50 = 50000


def get_predictions(test_peptides, model, binary=False, embed_peptides=None):
    '''
    Get predictions from a given model
    '''

    if embed_peptides is None:
        preds_cont = model.predict(test_peptides)
    else:
        preds_cont = model.predict([test_peptides, embed_peptides])
    preds_cont = [0 if y < 0 else y for y in preds_cont]
    if not binary:
        preds_bins = [1 if y >= 1-math.log(IC50_THRESHOLD, MAX_IC50)
                      else 0 for y in preds_cont]
    else:
        preds_bins = [1 if y >= 0.5 else 0 for y in preds_cont]
    return preds_cont, preds_bins


def mhcnuggets_lstm(input_size=(MAX_MASK_LEN, NUM_AAS),
                    hidden_size=64, output_size=1, dropout=0.2):
    '''
    MHCnuggets-LSTM model
    -----------
    input_size : Num dimensions of the encoding (11, 21) = (len, numAA)
    hidden_size : Num hidden dimensions
    output_size : Num of outputs
    dropout : dropout probability to apply
    '''

    model = Sequential()
    model.add(Masking(mask_value=MASK_VALUE, input_shape=(input_size)))
    model.add(LSTM(hidden_size, dropout=dropout, recurrent_dropout=dropout))
    model.add(Dense(hidden_size))
    model.add(Dropout(dropout))
    model.add(Activation('tanh'))
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    return model
