"""
Script that determines what the
closest allele MHC Class II to another
one is based on training data

Rohit Bhattacharya
rohit.bhattachar@gmail.com
"""

from mhcnuggets.src.dataset import Dataset
from mhcnuggets.src.supertypes import supertype_mhcII_allele, supertype_mhcII_allele_clade
try:
    import cPickle as pickle
except:
    import pickle
import os
import operator
MHCNUGGETS_HOME = os.path.join(os.path.dirname(__file__), '..')


def shorten_allele_names(alleles):
    """
    Shorten allele names for a list of alleles to fit supertype naming scheme
    Dimer names are shortened to only include beta-chain portion of name
    Ex. HLA-DPA103:01-DPB104:02 --> HLA-DPB104:02
    """
    shortened_alleles = list()
    for allele in alleles:
        if 'HLA' in allele and len(allele.split('-')) == 3:
            allele = 'HLA-' + allele.split('-')[2]
        shortened_alleles.append(allele)
    return shortened_alleles

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

def closest_human_allele_supertype(mhc, examples_per_allele):
    """
    Find the closest human allele to a given human MHC
    Search by biological supertype
    """

    alleles = sorted(examples_per_allele)
    closest_mhc = ""

    # shortening allele naming to remove alpha chain name for dimer molecules
    alleles_shortened = shorten_allele_names(alleles)
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
        _n_training = 0
        # search for nearest neighbour within the supertype
        for allele_list in supertype_mhcII_allele_clade[_super_type][mhc_shortened]:
            for allele in allele_list:
                if allele in alleles_shortened and examples_per_allele_shortened[allele] > _n_training:
                    _n_training = examples_per_allele_shortened[allele]
                    closest_mhc = short_to_full_alleles[allele]
            if closest_mhc != "":
                break
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

        n_training = examples_per_allele[allele]
        if (_gene == gene and _super_type == super_type and
            n_training > _n_training):
            closest_mhc = allele
            _n_training = n_training

    if closest_mhc == "":
        if 'DQ' in mhc:
            closest_mhc = 'HLA-DQA105:01-DQB102:01'
        elif 'DP' in mhc:
            closest_mhc = 'HLA-DPA102:01-DPB101:01'
        else:
            closest_mhc = 'HLA-DRB101:01'
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

    _n_training = 0
    for allele in examples_per_allele:
        gene = allele[4:6]
        n_training = examples_per_allele[allele]
        if _gene == gene and n_training > _n_training:
            closest_mhc = allele
            _n_training = n_training

    return closest_mhc

def closest_allele(mhc):
    """
    Find the closest allele to a
    given MHC
    """
    examples_per_allele = pickle.load(open(os.path.join(MHCNUGGETS_HOME,
                                                        "data",
                                                        "production",
                                                        "mhcII",
                                                        "examples_per_allele.pkl"), 'rb'))

    alleles = sorted(examples_per_allele)
    # search for exact match
    match = exact_match(mhc, alleles)
    if match:
        return match

    closest_mhc = ""

    # human allele
    if mhc[0:3] == 'HLA' and len(mhc) >= 12:
        closest_mhc = closest_human_allele_supertype(mhc, examples_per_allele)
        if not closest_mhc:
            closest_mhc = closest_human_allele_name(mhc, examples_per_allele)
    # mouse allele
    elif mhc[0:3] == 'H-2' and len(mhc) >= 7:
        closest_mhc = closest_mouse_allele(mhc, examples_per_allele)
    else:
        print("Invalid human or mouse allele")
        return

    return closest_mhc

def main():
    train_data = Dataset.from_csv(filename='data/production/mhcII/curated_training_data.csv',
                                  sep=',',
                                  allele_column_name='mhc',
                                  peptide_column_name='peptide',
                                  affinity_column_name='IC50(nM)')

    trained_alleles = []
    for trained_models in os.listdir('saves/production/'):
        trained_alleles.append(trained_models.split('.')[0])
    allele_example_dict = {}

    for allele in sorted(trained_alleles):
        n_training = len(train_data.get_allele(allele).peptides)
        allele_example_dict[allele] = n_training

    sorted_alleles = sorted(allele_example_dict.items(),
                            key=operator.itemgetter(1))
    for a in sorted_alleles:
        print(a)
    pickle.dump(allele_example_dict, open("data/production/mhcII/examples_per_allele.pkl", 'wb'))

if __name__ == "__main__":
    main()
