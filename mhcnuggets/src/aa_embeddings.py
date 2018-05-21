'''
Various embeddings for amino acids

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

from __future__ import print_function
import numpy as np

# constants
NUM_AAS = 21
MHCI_MASK_LEN = 15
MHCII_MASK_LEN = 30
MAX_MASK_LEN = 30

MASK_VALUE = -4200000
AA_LIST = list('ACDEFGHIKLMNPQRSTVWYX')
CAN_AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')

# establish indexing from amino acid to number
AA_VOCAB = {aa: i for i, aa in enumerate(AA_LIST)}
AA_INV_VOCAB = {AA_VOCAB[aa]: aa for aa in AA_VOCAB}

# generate onehot encoding
AA_ONEHOT = {}
for aa in AA_VOCAB:
    AA_ONEHOT[aa] = [0]*len(AA_LIST)
    AA_ONEHOT[aa][AA_VOCAB[aa]] = 1
AA_ONEHOT['Z'] = [MASK_VALUE] * len(AA_LIST)

# generate a more noisy onehot encoding
# given by 0.9 at the identity and (1-0.9)/20
# elsewhere
AA_SOFTHOT = {}
for aa in AA_VOCAB:
    filler = (1-0.9)/(len(AA_LIST)-1)
    AA_SOFTHOT[aa] = [filler]*len(AA_LIST)
    AA_SOFTHOT[aa][AA_VOCAB[aa]] = 0.9
AA_SOFTHOT['Z'] = [MASK_VALUE] * len(AA_LIST)

AA_VOCAB = {aa: i+1 for i, aa in enumerate(AA_LIST)}
AA_VOCAB['Z'] = 0


if __name__ == '__main__':
    print(AA_ONEHOT)
    print(AA_SOFTHOT)
