'''
Dataset class for
pMHC data

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

from __future__ import print_function
import math
import os
import string
from sklearn.model_selection import train_test_split
from mhcnuggets.src.aa_embeddings import AA_ONEHOT
from mhcnuggets.src.aa_embeddings import AA_SOFTHOT
from mhcnuggets.src.aa_embeddings import AA_VOCAB
from mhcnuggets.src.aa_embeddings import AA_INV_VOCAB
from mhcnuggets.src.aa_embeddings import MASK_VALUE
from mhcnuggets.src.aa_embeddings import MHCI_MASK_LEN
from mhcnuggets.src.aa_embeddings import MHCII_MASK_LEN

import numpy as np

NUM_AAS = 21
MAX_CORE_LEN = 9


def standardize_mhc(mhc):
    """
    Standardize the name of mhc
    """

    mhc = mhc.replace('*', '')
    return mhc


def map_ic50_for_regression(ic50, max_ic50):
    """
    Map the IC50 between 0 or 1
    """

    if ic50 > max_ic50:
        ic50 = max_ic50
    return 1-math.log(ic50, max_ic50)


def map_proba_to_ic50(proba, max_ic50):
    """
    Map score outputs from the models to IC50s
    """

    return max_ic50 ** (1-proba)


def binarize_ic50(ic50, ic50_threshold):
    """
    Binarize ic50 based on a threshold
    """

    if ic50 <= ic50_threshold:
        return 1
    return 0

def mask_peptides(peptides, max_len, pad_aa='Z', 
                  pre_pad=True):
    """
    Mask and pad for an LSTM (can be used without calling from instance of Dataset)
    """
    num_skipped = 0
    num_total = 0

    padded_peptides = []
    original_peptides = []
    
    #for residues that we are not sure about and we don't have encoding for, we replace them with X
    aa_replace={'J':'X','B':'X','U':'X'}

    for peptide in peptides:
        if len(peptide) > max_len:
            num_skipped += 1
            print('Skipping ' + peptide + ' > ' + str(max_len) + ' residues')
            continue
        
        peptide=peptide.upper()
        # determine and apply padding
        original_peptides.append(peptide)  #keep track of the original and in register with the padded version

        # clean up the peptides if there is residues that we don't have encoding for 
        pep_list=list(peptide)
        pep_clean_list=[aa_replace[p] if p in aa_replace else p for p in pep_list]
        peptide=''.join(pep_clean_list)

        num_pad = max_len - len(peptide)
        if pre_pad:
            peptide += pad_aa * num_pad
        else:
            peptide = pad_aa * num_pad + peptide

        padded_peptides.append(peptide)

    print("Number of peptides skipped/total due to length", num_skipped, '/', num_total)
    return padded_peptides, original_peptides


def tensorize_keras(peptides, embed_type='softhot'):
    """
    Tensorize the data when not called from Dataframe instance
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

    # make numpy arrays for the encoded peptides
    encoded_peptides = np.array(encoded_peptides, dtype='float32')

    return encoded_peptides

    
    encoded_peptides = []
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

    def __init__(self,  alleles, peptides, affinities,
                 m_affinities, b_affinities, m_types, m_sources):
        '''
        Constructor for Dataset

        alleles : list of each tested allele
        peptides : list of corresponding tested peptides
        affinities : IC50 nM affinity for pMHC pairs
        m_affinities : mapped IC50 between 0-1
        b_affinities : binarized IC50 at selected threshold (default 500 nM)
        m_types  : list of measurement types
        m_sources  : list of measurement sources
        '''

        self.alleles = alleles
        self.peptides = peptides
        self.affinities = affinities
        self.continuous_targets = m_affinities
        self.binary_targets = b_affinities
        self.m_types = m_types
        self.m_sources = m_sources

    @staticmethod
    def from_csv(filename, ic50_threshold, max_ic50, sep, allele_column_name,
                 peptide_column_name, affinity_column_name, type_column_name, source_column_name):
        """
        Create a Dataset object from a csv file
        """

        in_file = open(filename)
        raw_header = in_file.readline().strip()
        header = ''.join(list(filter(lambda x: x in string.printable, raw_header))).split(sep)  #clean nasty invisible characters
        allele_ind = header.index(allele_column_name)
        peptide_ind = header.index(peptide_column_name)
        affinity_ind = header.index(affinity_column_name)
        type_ind = header.index(type_column_name)
        source_ind = header.index(source_column_name)

        alleles, peptides, affinities = [], [], []
        m_sources, m_types = [], []
        m_affinities, b_affinities = [], []  # continuous and binary targets

        for line in in_file:

            line = line.strip().split(sep)
            allele = standardize_mhc(line[allele_ind])
            affinity = float(line[affinity_ind])
            alleles.append(allele)
            peptides.append(line[peptide_ind])
            affinities.append(affinity)
            m_types.append(line[type_ind])
            m_sources.append(line[source_ind])
            m_affinities.append(map_ic50_for_regression(affinity, max_ic50))
            b_affinities.append(binarize_ic50(affinity, ic50_threshold))

        in_file.close()
        return Dataset(alleles, peptides, affinities,
                       m_affinities, b_affinities, m_types, m_sources)


    def get_allele(self, allele, mass_spec,ic50_threshold, length=None):
        """
        Return a subset dataset containing info
        for a given allele
        """

        alleles, peptides, affinities = [], [], []
        m_sources, m_types = [], []
        m_affinities, b_affinities = [], []

        num_positives = 0
        num_real_negatives = 0
        num_random_negatives = 0 

        for i in range(len(self.alleles)):
            if self.alleles[i] == allele and (length is None or length == len(self.peptides[i])) and \
               self.m_sources[i] != 'random':
                alleles.append(self.alleles[i])
                peptides.append(self.peptides[i])
                affinities.append(self.affinities[i])
                m_affinities.append(self.continuous_targets[i])
                b_affinities.append(self.binary_targets[i])
                m_types.append(self.m_types[i])
                m_sources.append(self.m_sources[i])

        #get basic composition of the examples for the allele
        np_affinities = np.array(affinities, dtype='float32')
        np_sources = np.array(m_sources, dtype='object')
        positives_idx = np.flatnonzero(np_affinities <= ic50_threshold)
        random_negative_idx = np.where(np.logical_and(np_affinities > ic50_threshold, np_sources == 'random'))[0]
        real_negative_idx = np.where(np.logical_and(np_affinities > ic50_threshold, np_sources != 'random'))[0]
        
        num_positives = len(positives_idx)
        num_random_negatives = len(random_negative_idx)
        num_real_negatives = len(real_negative_idx)


        return Dataset(alleles, peptides, affinities,
                       m_affinities, b_affinities, m_types, m_sources), num_positives, num_random_negatives, num_real_negatives


    def mask_peptides(self, max_len, pad_aa='Z', 
                      pre_pad=True):
        """
        Mask and pad for an LSTM
        """
        padded_peptides = []
        m_affinities = []
        b_affinities = []
        affinities = []
        alleles = []
        m_types = []
        m_sources = []
        num_skipped = 0
        num_total = 0
        
        aa_replace={'J':'X','B':'X','U':'X'}

        for i, peptide in enumerate(self.peptides):

            num_total += 1
            if len(peptide) > max_len:
                num_skipped += 1
                continue
            
            peptide=peptide.upper()
            pep_list=list(peptide)
            pep_clean_list=[aa_replace[p] if p in aa_replace else p for p in pep_list]
            peptide=''.join(pep_clean_list)
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
            m_types.append(self.m_types[i])
            m_sources.append(self.m_sources[i])
            
        self.peptides = padded_peptides
        self.continuous_targets = m_affinities
        self.binary_targets = b_affinities
        self.alleles = alleles
        self.affinities = affinities
        self.m_types = m_types
        self.m_sources = m_sources
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

def parse_args():
    '''
    Parse user arguments
    '''

    info = 'Methods to create and manipulate the Dataset class'
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-d', '--data',
                        type=str, default=None,    #currently hardcoded in main function
                        help='Path to data file')

    parser.add_argument('-l', '--ic50_threshold',
                        type=int, default=500,
                        help='Threshold on ic50 (nM) that separates binder/non-binder')

    parser.add_argument('-x', '--max_ic50',
                        type=int, default=50000,
                        help='Maximum ic50 value')


    args = parser.parse_args()
    return vars(args)

    
def main():

    opts = parse_args()
    datafile = opts['data']    #for future use
    ic50_threshold = opts['ic50_threshold']
    max_ic50 = opts['max_ic50']
    

    """
    Test
    """

    train_data = Dataset.from_csv(filename='data/kim2014/train.csv',
                                  sep=',', ic50_threshold=500,
                                  max_ic50=50000,
                                  allele_column_name='mhc',
                                  peptide_column_name='peptide',
                                  affinity_column_name='IC50(nM)'
                                  )

    test_data = Dataset.from_csv(filename='data/kim2014/test.csv',
                                 sep=',', ic50_threshold=500,
                                 max_ic50=50000,
                                 allele_column_name='mhc',
                                 peptide_column_name='peptide',
                                 affinity_column_name='IC50(nM)'
                                 )

    production_data = Dataset.from_csv(filename='data/production/curated_training_data.csv',
                                       sep=',', ic50_threshold=500,
                                       max_ic50=50000,
                                       allele_column_name='mhc',
                                       peptide_column_name='peptide',
                                       affinity_column_name='IC50(nM)'
                                       )

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
