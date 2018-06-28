"""
Script that determines
what the closest allele to another
one is based on training data

Rohit Bhattacharya
rohit.bhattachar@gmail.com
"""

from mhcnuggets.src.dataset import Dataset
try:
    import cPickle as pickle
except:
    import pickle
import operator
import os
MHCNUGGETS_HOME = os.path.join(os.path.dirname(__file__), '..')


def exact_match(mhc, alleles):
    """
    Return an exact match
    if there is one
    """
    for allele in alleles:
        if mhc == allele:
            return allele


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
    match = exact_match(mhc, alleles)

    # search for exact match
    if match:
        return match

    closest_mhc = ""
    _n_training = 0

    # get gene/supertype/subtype of allele
    try:
        split_mhc = mhc.split('-')
        mhc = split_mhc[0] + split_mhc[1]
        _gene = mhc[4:8]
        #_super_type = int(mhc[8:10])
        #_sub_type = int(mhc[10:12])
        _super_type, sub_type = mhc[8:].split(':')

    except ValueError as e:
        print("Invalid human allele")
        return

    # find if there's a supertype, and select
    # the one with the max training examples
    for allele in alleles:
        try:
            split_mhc = mhc.split('-')
            mhc = split_mhc[0] + split_mhc[1]
            gene = allele[4:8]
            super_type, sub_type = mhc[8:].split(':')

        except:
            continue

        n_training = examples_per_allele[allele]
        if (_gene == gene and _super_type == super_type and
            n_training > _n_training):
            closest_mhc = allele
            _n_training = n_training


    # otherwise choose gene level
    if closest_mhc == "":
        if 'DQA' in mhc:
            closest_mhc = 'HLA-DQA105:01-DQB102:01'
        else:
            closest_mhc = 'HLA-DRB101:01'

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
