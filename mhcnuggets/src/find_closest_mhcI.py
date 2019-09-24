"""
Script that determines what the
closest MHC Class I allele to another
one is based on training data

Rohit Bhattacharya
rohit.bhattachar@gmail.com
"""

from mhcnuggets.src.dataset import Dataset
from mhcnuggets.src.supertypes import supertype_mhcI_allele, supertype_mhcI_group, supertype_hla_C_allele, supertype_hla_C_allele_clade

import argparse

try:
    import cPickle as pickle
except:
    import pickle
import operator
import os
MHCNUGGETS_HOME = os.path.join(os.path.dirname(__file__), '..')

#constants
A_DEFAULT_PAN_MODEL = 'HLA-A02:01'
B_DEFAULT_PAN_MODEL = 'HLA-B07:02'
C_DEFAULT_PAN_MODEL = 'HLA-C04:01'

def exact_match(mhc, alleles):
    """
    Return an exact match
    """
    for allele in alleles:
        if mhc == allele:
            return allele

def better_allele(candidate_allele, current_closest_allele,
                  examples_per_allele):
    """
    Determine whether the candidate allele is a better pan choice
    than the current closest allele
    """
    # If we do not have a closest allele right now, then candidate_allele is
    # a better allele
    if current_closest_allele == '':
        return True

    if examples_per_allele[candidate_allele] > examples_per_allele[current_closest_allele]:
        return True
    else:
        return False

def find_weighted_optimal_allele(allele_lists, examples_per_allele):
    """
    Find optimal closest allele in the given allele_lists based on
    criteria specified in better_allele
    """
    closest_mhc = ''
    for allele_list in allele_lists:
        for allele in allele_list:
            if allele in examples_per_allele and better_allele(allele, closest_mhc,
                                                                examples_per_allele):
                closest_mhc = allele
        if closest_mhc != "":
            break
    return closest_mhc

def closest_human_allele_supertype(mhc, examples_per_allele):
    """
    Find the closest human allele to a given human MHC
    Search by biological supertype
    """

    alleles = sorted(examples_per_allele)
    closest_mhc = ""
    # try to search biological supertype
    if mhc[4].upper() == 'A' or mhc[4].upper() == 'B':
        if mhc in supertype_mhcI_allele:
            _super_type = supertype_mhcI_allele[mhc]
            allele_lists = supertype_mhcI_group[_super_type]
            closest_mhc = find_weighted_optimal_allele(allele_lists,
                                                       examples_per_allele)
    elif mhc[4].upper() == 'C':
        if mhc in supertype_hla_C_allele:
            _super_type = supertype_hla_C_allele[mhc]
            allele_lists = supertype_hla_C_allele_clade[_super_type][mhc]
            closest_mhc = find_weighted_optimal_allele(allele_lists,
                                                       examples_per_allele)
    return closest_mhc

def closest_human_allele_name(mhc, examples_per_allele):
    """
    Find the closest human allele to a given human MHC
    Search by name similarity
    """
    # get name-based super gene/supertype/subtype of allele
    try:
        _gene = mhc[4]
        _super_type = int(mhc[5:7])
        _sub_type = int(mhc[8:10])

    except ValueError as e:
        print("Invalid human allele")
        return

    _n_training = 0
    alleles = sorted(examples_per_allele)
    closest_mhc = ""

    for allele in alleles:
        try:
            gene, super_type, sub_type = (allele[4],
                                          int(allele[5:7]),
                                          int(allele[8:10]))
        except:
            continue

        if (_gene == gene and _super_type == super_type and
            better_allele(allele, closest_mhc, examples_per_allele)):
            closest_mhc = allele
    return closest_mhc

def default_closest_human_allele(mhc):
    """
    Assign default closest allele to mhc based on the gene
    """
    _gene = mhc[4]
    closest_mhc = ""
    if _gene == 'A':
        closest_mhc = A_DEFAULT_PAN_MODEL
    elif _gene == 'B':
        closest_mhc = B_DEFAULT_PAN_MODEL
    elif _gene == 'C':
        closest_mhc = C_DEFAULT_PAN_MODEL

    return closest_mhc

def mhc_allele_group_protein_naming(mhc):
    """
    Shorten alleles to allele_group - protein naming
    HLA-A01:01:01:01 --> HLA-A01:01
    If mhc name does not contain '-' and ':', return the mhc name
    """
    if '-' in mhc and ':' in mhc:
        split_mhc = mhc.split('-')
        mhc_correct_name = split_mhc[0]

        # Shorten all allele segment names
        for name_segment in split_mhc[1:]:
            mhc_correct_name = mhc_correct_name + '-' + ':'.join(name_segment.split(':')[:2])

        return mhc_correct_name
    else:
        return mhc

def closest_allele(mhc,pickle_path):
    """
    Find the closest allele to a
    given MHC
    """
    mhc = mhc_allele_group_protein_naming(mhc)
    examples_per_allele = pickle.load(open(os.path.join(MHCNUGGETS_HOME,pickle_path), 'rb'))

    alleles = sorted(examples_per_allele)

    # search for exact match
    match = exact_match(mhc, alleles)
    if match:
        return match

    # search for biological supertype allele match
    closest_mhc = closest_human_allele_supertype(mhc, examples_per_allele)
    # if no biological supertype match, search for name-based allele match
    if not closest_mhc:
        closest_mhc = closest_human_allele_name(mhc, examples_per_allele)
    # if no name-based allele match, assign default closest allele
    if not closest_mhc:
        closest_mhc = default_closest_human_allele(mhc)
    return closest_mhc


def parse_args():
    '''
    Parse user arguments
    '''

    info = 'Evaluate a MHCnuggets model on given data'
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-d', '--data',
                        type=str, default='data/production/curated_training_data.csv',
                        help='Data file to use for evaluation, should ' +
                              'be csv files w/ formatting similar to ' +
                              'data/production/curated_training_data.csv')

    parser.add_argument('-s', '--save_path',
                        type=str, required=False, default='saves/production/',
                        help='Path to which the model weights are saved')

    parser.add_argument('-k', '--pickle_path',
                        type=str, required=False, default='data/production/examples_per_allele.pkl',
                        help='Path to which the pickle file is saved')

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

    opts = parse_args()
    train_data = Dataset.from_csv(filename=opts['data'],
                                  sep=',', ic50_threshold=opts['ic50_threshold'], 
                                  max_ic50=opts['max_ic50'],
                                  allele_column_name='mhc',
                                  peptide_column_name='peptide',
                                  affinity_column_name='IC50(nM)',
                                  type_column_name='measurement_type',
                                  source_column_name='measurement_source'
                                    )

    # get the alleles for which training actually succeeded
    trained_alleles = []
    for trained_models in os.listdir(opts['save_path']):
        trained_alleles.append(trained_models.split('.')[0])

    allele_example_dict = {}
    for allele in sorted(trained_alleles):
        adata, num_pos, num_rand_neg, num_real_neg = train_data.get_allele(allele,opts['mass_spec'], opts['ic50_threshold'])
        n_training = len(adata.peptides)
        allele_example_dict[allele] = n_training

    sorted_alleles = sorted(allele_example_dict.items(),
                            key=operator.itemgetter(1))

    for a in sorted_alleles:
        print(a)
    pickle.dump(allele_example_dict, open(opts['pickle_path'], 'wb'))


if __name__ == "__main__":
    main()
