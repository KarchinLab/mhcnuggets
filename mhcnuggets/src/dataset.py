'''
Dataset class for
pMHC data

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

from __future__ import print_function
import math
import os
from sklearn.model_selection import train_test_split
from mhcnuggets.src.aa_embeddings import AA_ONEHOT
from mhcnuggets.src.aa_embeddings import AA_SOFTHOT
from mhcnuggets.src.aa_embeddings import AA_VOCAB
from mhcnuggets.src.aa_embeddings import AA_INV_VOCAB
from mhcnuggets.src.aa_embeddings import MASK_VALUE
import numpy as np

IC50_THRESHOLD = 500
MAX_IC50 = 50000
NUM_AAS = 21
MAX_CORE_LEN = 9
MAX_MASK_LEN = 15


def standardize_mhc(mhc):
    """
    Standardize the name of mhc
    """

    mhc = mhc.replace('*', '')
    mhc = mhc.replace(':', '')
    return mhc


def map_ic50_for_regression(ic50):
    """
    Map the IC50 between 0 or 1
    """

    if ic50 > MAX_IC50:
        ic50 = MAX_IC50
    return 1-math.log(ic50, MAX_IC50)


def map_proba_to_ic50(proba):
    """
    Map score outputs from the models to IC50s
    """

    return MAX_IC50 ** (1-proba)


def binarize_ic50(ic50):
    """
    Binarize ic50 based on a threshold
    """

    if ic50 <= IC50_THRESHOLD:
        return 1
    return 0


def mask_peptides(peptides, pad_aa='Z', max_len=MAX_MASK_LEN,
                  pre_pad=True):
    """
    Mask and pad for an LSTM
    """

    padded_peptides = []
    for i, peptide in enumerate(peptides):

        if len(peptide) > max_len:
            print('Skipping ' + peptide + ' > ' + str(max_len) + ' residues')
            continue
        # determine and apply padding
        num_pad = max_len - len(peptide)
        if pre_pad:
            peptide += pad_aa * num_pad
        else:
            peptide = pad_aa * num_pad + peptide
        padded_peptides.append(peptide)

    return padded_peptides


def cut_pad_peptides(peptides, max_len=MAX_CORE_LEN,
                     padding_aa='X', padding_pos=6):
    """
    Cut and pad peptides to a desired max length
    """

    padded_peptides = []
    for peptide in peptides:

        # determine and apply padding
        num_pad = max_len - len(peptide)
        peptide = (peptide[0:padding_pos] + padding_aa*num_pad +
                   peptide[padding_pos:])

        # cut until length matches max_len
        while len(peptide) > max_len:
            peptide = peptide[0:padding_pos] + peptide[padding_pos+1:]

        padded_peptides.append(peptide)

    return padded_peptides


def tensorize_keras(peptides, embed_type='softhot'):
    """
    Tensorize the data
    """

    encoded_peptides = []
    if embed_type == 'softhot':
        embedding = AA_SOFTHOT
    elif embed_type == 'onehot':
        embedding = AA_ONEHOT

    # go through each peptide and embed
    for i, peptide in enumerate(peptides):

        # embed the peptide
        encoded_peptide = []
        for residue in peptide:
            encoded_peptide.append(embedding[residue])
        encoded_peptides.append(encoded_peptide)

    # make numpy arrays for the input
    encoded_peptides = np.array(encoded_peptides, dtype='float32')

    return encoded_peptides


def get_validation_split(X, y, split=0.2):
    """
    Get train and validation split
    """

    return train_test_split(X, y, test_size=split, random_state=42)


def exclude_from_data(data1, data2):
    """
    Exclude data from data2 from data1
    """

    for allele, peptide in data2.alleles, data2.peptides:
        print(allele, peptide)


class Dataset():
    """
    Dataset class
    """

    def __init__(self, alleles, peptides, affinities,
                 m_affinities, b_affinities):
        '''
        Constructor for Dataset

        alleles : list of each tested allele
        peptides : list of corresponding tested peptides
        affinities : IC50 nM affinity for pMHC pairs
        m_affinities : mapped IC50 between 0-1
        b_affinities : binarized IC50 at fixed threshold (500 nM)
        '''

        self.alleles = alleles
        self.peptides = peptides
        self.affinities = affinities
        self.continuous_targets = m_affinities
        self.binary_targets = b_affinities


    @staticmethod
    def from_csv(filename, sep, allele_column_name,
                 peptide_column_name, affinity_column_name):
        """
        Create a Dataset object from a csv file
        """

        in_file = open(filename)
        header = in_file.readline().strip().split(sep)
        allele_ind = header.index(allele_column_name)
        peptide_ind = header.index(peptide_column_name)
        affinity_ind = header.index(affinity_column_name)

        alleles, peptides, affinities = [], [], []
        m_affinities, b_affinities = [], []  # continuous and binary targets

        for line in in_file:

            line = line.strip().split(sep)
            allele = standardize_mhc(line[allele_ind])
            affinity = float(line[affinity_ind])
            alleles.append(allele)
            peptides.append(line[peptide_ind])
            affinities.append(affinity)
            m_affinities.append(map_ic50_for_regression(affinity))
            b_affinities.append(binarize_ic50(affinity))

        in_file.close()
        return Dataset(alleles, peptides, affinities,
                       m_affinities, b_affinities)


    def get_allele(self, allele, length=None):
        """
        Return a subset dataset containing info
        for a given allele
        """

        alleles, peptides, affinities = [], [], []
        m_affinities, b_affinities = [], []

        for i in range(len(self.alleles)):
            if self.alleles[i] == allele and (length is None or length == len(self.peptides[i])):
                alleles.append(self.alleles[i])
                peptides.append(self.peptides[i])
                affinities.append(self.affinities[i])
                m_affinities.append(self.continuous_targets[i])
                b_affinities.append(self.binary_targets[i])

        return Dataset(alleles, peptides, affinities,
                       m_affinities, b_affinities)


    def cut_pad_peptides(self, max_len=MAX_CORE_LEN,
                         padding_aa='X', padding_pos=6):
        """
        Cut and pad peptides to a desired max length
        """

        padded_peptides = []
        for peptide in self.peptides:

            # determine and apply padding
            num_pad = max_len - len(peptide)
            peptide = (peptide[0:padding_pos] + padding_aa*num_pad +
                       peptide[padding_pos:])

            # cut until length matches max_len
            while len(peptide) > max_len:
                peptide = peptide[0:padding_pos] + peptide[padding_pos+1:]

            padded_peptides.append(peptide)

        self.peptides = padded_peptides


    def mask_peptides(self, pad_aa='Z', max_len=MAX_MASK_LEN,
                      pre_pad=True):
        """
        Mask and pad for an LSTM
        """

        padded_peptides = []
        m_affinities = []
        b_affinities = []
        affinities = []
        alleles = []
        num_skipped = 0
        num_total = 0
        for i, peptide in enumerate(self.peptides):

            num_total += 1
            if len(peptide) > max_len:
                num_skipped += 1
                continue

            # determine and apply padding
            num_pad = max_len - len(peptide)
            if pre_pad:
                peptide += pad_aa * num_pad
            else:
                peptide = pad_aa * num_pad + peptide

            padded_peptides.append(peptide)
            m_affinities.append(self.continuous_targets[i])
            b_affinities.append(self.binary_targets[i])
            affinities.append(self.affinities[i])
            alleles.append(self.alleles[i])

        self.peptides = padded_peptides
        self.continuous_targets = m_affinities
        self.binary_targets = b_affinities
        self.alleles = alleles
        self.affinities = affinities
        print("Number of peptides skipped/total due to length", num_skipped, '/', num_total)


    def tensorize_keras(self, embed_type='softhot'):
        """
        Tensorize the data
        """

        peptides = []
        if embed_type == 'softhot':
            embedding = AA_SOFTHOT
        elif embed_type == 'onehot':
            embedding = AA_ONEHOT

        # go through each peptide and embed
        for i, peptide in enumerate(self.peptides):

            # embed the peptide
            encoded_peptide = []
            for residue in peptide:
                encoded_peptide.append(embedding[residue])
            peptides.append(encoded_peptide)

        # make numpy arrays for the input, continuous and binary targets
        peptides = np.array(peptides, dtype='float32')
        continuous_targets = np.array(self.continuous_targets, dtype='float32')
        binary_targets = np.array(self.binary_targets, dtype='int')
        return peptides, continuous_targets, binary_targets


def main():
    """
    Test
    """

    train_data = Dataset.from_csv(filename='data/kim2014/train.csv',
                                  sep=',',
                                  allele_column_name='mhc',
                                  peptide_column_name='peptide',
                                  affinity_column_name='IC50(nM)')

    test_data = Dataset.from_csv(filename='data/kim2014/test.csv',
                                 sep=',',
                                 allele_column_name='mhc',
                                 peptide_column_name='peptide',
                                 affinity_column_name='IC50(nM)')

    production_data = Dataset.from_csv(filename='data/production/curated_training_data.csv',
                                       sep=',',
                                       allele_column_name='mhc',
                                       peptide_column_name='peptide',
                                       affinity_column_name='IC50(nM)')

    '''
    print('Num train loaded:', len(train_data.alleles))
    print('Num train alleles:', len(set(train_data.alleles)))
    print('Num test loaded:', len(test_data.alleles))
    print('Num test alleles:', len(set(test_data.alleles)))
    print('Num production loaded:', len(production_data.alleles))
    print('Num production alleles:', len(set(production_data.alleles)))
    print(sorted(set(train_data.alleles)))
    print(sorted(set(test_data.alleles)))
    print(sorted(set(production_data.alleles)))
    '''

    # write supported production alleles to file
    '''
    allele_file = open('data/production/supported_alleles.txt', 'w')
    for mhc in sorted(set(production_data.alleles)):
        allele_file.write(mhc + '\n')

    allele_file.close()
    '''

if __name__ == '__main__':
    main()
