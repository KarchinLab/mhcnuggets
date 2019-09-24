"""
Script that determines what the
closest allele MHC Class II to another
one is based on training data

Rohit Bhattacharya
rohit.bhattachar@gmail.com
"""

from mhcnuggets.src.dataset import Dataset
from mhcnuggets.src.supertypes import supertype_mhcII_allele, supertype_mhcII_allele_clade
import argparse

try:
    import cPickle as pickle
except:
    import pickle
import os
import operator
MHCNUGGETS_HOME = os.path.join(os.path.dirname(__file__), '..')

#constants
DQ_DEFAULT_PAN_MODEL = 'HLA-DQA105:01-DQB102:01'
DP_DEFAULT_PAN_MODEL = 'HLA-DPA102:01-DPB101:01'
DR_DEFAULT_PAN_MODEL = 'HLA-DRB101:01'


def shorten_allele_dict_names(examples_per_allele):
    """
    Shorten allele names for a dictionary with alleles for keys to fit supertype naming scheme
    Dimer names are shortened to only include beta-chain portion of name
    Ex. HLA-DPA103:01-DPB104:02 --> HLA-DPB104:02
    """
    shortened_dict = dict()
    for allele in examples_per_allele.keys():
        # Checking if the allele has a dimer name and is a human allele
        if len(allele.split('-')) == 3 and allele[0:3] == 'HLA':
            shortened_allele = 'HLA-' + allele.split('-')[2]
        else:
            shortened_allele = allele
        if (shortened_allele not in shortened_dict or
        (shortened_allele in shortened_dict and
         examples_per_allele[allele] > shortened_dict[shortened_allele])):
            shortened_dict[shortened_allele] = examples_per_allele[allele]
    return shortened_dict

def create_short_to_full_dict(examples_per_allele):
    """
    Creates dictionary of shortened allele names to full allele names
    If one short name maps to multiple full names,
    we map to the full allele with the largest sample size
    """
    short_to_full_dict = dict()
    short_to_sample_size = dict()
    for allele in examples_per_allele.keys():
        # Checking if the allele has a dimer name and is a human allele
        if len(allele.split('-')) == 3 and allele[0:3] == 'HLA':
            shortened_allele = 'HLA-' + allele.split('-')[2]
        else:
            shortened_allele = allele
        if (shortened_allele not in short_to_full_dict or
        (shortened_allele in short_to_full_dict and
         examples_per_allele[allele] > short_to_sample_size[shortened_allele])):
            short_to_full_dict[shortened_allele] = allele
            short_to_sample_size[shortened_allele] = examples_per_allele[allele]
    return short_to_full_dict

def exact_match(mhc, alleles):
    """
    Return an exact match
    if there is one
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

def find_weighted_optimal_allele(allele_lists, examples_per_allele_shortened,
                                 short_to_full_names):
    """
    Find optimal closest allele in the given allele_lists based on
    training sample sizes and other specified criteria.
    """
    closest_mhc = ''
    for allele_list in allele_lists:
        for allele in allele_list:
            if allele in examples_per_allele_shortened and better_allele(allele, closest_mhc,
                                                                         examples_per_allele_shortened):
                closest_mhc = short_to_full_alleles[allele]
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

    # shortening allele naming to remove alpha chain name for dimer molecules
    examples_per_allele_shortened = shorten_allele_dict_names(examples_per_allele)
    if len(mhc.split('-')) == 3:
        mhc_shortened = 'HLA-' + mhc.split('-')[2]
    else:
        mhc_shortened = mhc

    # a map from shortened allele names to full allele names
    short_to_full_alleles = create_short_to_full_dict(examples_per_allele)

    # try to search biological supertype
    if mhc_shortened in supertype_mhcII_allele:
        _super_type = supertype_mhcII_allele[mhc_shortened]
        allele_lists = supertype_mhcII_allele_clade[_super_type][mhc_shortened]
        closest_mhc = find_weighted_optimal_allele(allele_lists,
                                                   examples_per_allele_shortened,
                                                   short_to_full_names)
    return closest_mhc

def closest_human_allele_name(mhc, examples_per_allele):
    """
    Find the closest human allele to a given human MHC
    Search by name similarity
    """
    try:
        _gene = mhc[4:8]
        _super_type = mhc[8:10]
        _sub_type = mhc[11:13]
    except ValueError as e:
        print("Invalid human allele")
        return
    _n_training = 0
    alleles = sorted(examples_per_allele)

    closest_mhc = ""

    for allele in alleles:
        try:
            gene = allele[4:8]
            super_type = allele[8:10]
            sub_type = allele[11:13]
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
    closest_mhc = ""
    if 'DQ' in mhc:
        closest_mhc = DQ_DEFAULT_PAN_MODEL
    elif 'DP' in mhc:
        closest_mhc = DP_DEFAULT_PAN_MODEL
    else:
        closest_mhc = DR_DEFAULT_PAN_MODEL
    return closest_mhc

def closest_mouse_allele(mhc, examples_per_allele):
    """
    Find the closest mouse allele to a given mouse MHC
    """
    closest_mhc = ""
    try:
        _gene = mhc[4:6]
    except ValueError as e:
        print("Invalid mouse allele")
        return

    for allele in examples_per_allele:
        gene = allele[4:6]
        if _gene == gene and better_allele(allele, closest_mhc,
                                           examples_per_allele):
            closest_mhc = allele

    return closest_mhc

def mhc_allele_group_protein_naming(mhc):
    """
    Shorten alleles to allele_group - protein naming
    HLA-DRB101:01:01:01 --> HLA-DRB101:01
    HLA-DPA101:01:01:01-DPB101:01:01:01 --> HLA-DPA101:01-DPB101:01
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

    closest_mhc = ""

    # human allele
    if mhc[0:3] == 'HLA':
        # search for biological supertype allele match
        closest_mhc = closest_human_allele_supertype(mhc, examples_per_allele)
        # if no biological supertype match, search for name-based allele match
        if not closest_mhc:
            closest_mhc = closest_human_allele_name(mhc, examples_per_allele)
        # if no name-based allele match, assign default closest allele
        if not closest_mhc:
            closest_mhc = default_closest_human_allele(mhc)

    # mouse allele
    elif mhc[0:3] == 'H-2':
        closest_mhc = closest_mouse_allele(mhc, examples_per_allele)
    else:
        print("Invalid human or mouse allele")
        return

    return closest_mhc

def parse_args():
    '''
    Parse user arguments
    '''

    info = 'Evaluate a MHCnuggets model on given data'
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-d', '--data',
                        type=str, default='data/production/mhcII/curated_training_data.csv',
                        help=('Data file to use for evaluation, should ' +
                              'be csv files w/ formatting similar to ' +
                              'data/production/curated_training_data.csv'))

    parser.add_argument('-s', '--save_path',
                        type=str, required=True, default='saves/production/',
                        help=('Path to which the model weights are saved'))

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
    train_data = Dataset.from_csv(filename=opts['data'], ic50_threshold=opts['ic50_threshold'],
                                  max_ic50=opts['max_ic50'],
                                  sep=',',  
                                  allele_column_name='mhc',
                                  peptide_column_name='peptide',
                                  affinity_column_name='IC50(nM)',
                                  type_column_name='measurement_type',
                                  source_column_name='measurement_source'
                                  )


    trained_alleles = []
    for trained_models in os.listdir(opts['save_path']):
        trained_alleles.append(trained_models.split('.')[0])
    allele_example_dict = {}

    for allele in sorted(trained_alleles):
        adata, num_pos, num_rand_neg, num_real_neg = train_data.get_allele(allele,opts['mass_spec'],opts['ic50_threshold'] )
        n_training = len(adata.peptides)
        allele_example_dict[allele] = n_training

    sorted_alleles = sorted(allele_example_dict.items(),
                            key=operator.itemgetter(1))
    for a in sorted_alleles:
        print(a)
    pickle.dump(allele_example_dict, open(opts['pickle_path'], 'wb'))

if __name__ == "__main__":
    main()
