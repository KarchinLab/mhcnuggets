"""
Script that determines what the
closest MHC Class I allele to another
one is based on training data

Rohit Bhattacharya
rohit.bhattachar@gmail.com
"""

from mhcnuggets.src.dataset import Dataset
from mhcnuggets.src.supertypes import supertype_mhcI_allele, supertype_mhcI_group
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
    # try to search biological supertype
    if mhc in supertype_mhcI_allele:
        _super_type = supertype_mhcI_allele[mhc]
        _n_training = 0
        # search for nearest neighbour within the supertype
        for allele_list in supertype_mhcI_group[_super_type]:
            for allele in allele_list:
                if allele in alleles and examples_per_allele[allele] > _n_training:
                    _n_training = examples_per_allele[allele]
                    closest_mhc = allele
            if closest_mhc != "":
                break
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

        n_training = examples_per_allele[allele]
        if (_gene == gene and _super_type == super_type and
            n_training > _n_training):
            closest_mhc = allele
            _n_training = n_training
    if closest_mhc == "":
        if _gene == 'A':
            closest_mhc = 'HLA-A02:01'
        elif _gene == 'B':
            closest_mhc = 'HLA-B07:02'
        elif _gene == 'C':
            closest_mhc = 'HLA-C04:01'
    return closest_mhc


def closest_allele(mhc):
    """
    Find the closest allele to a
    given MHC
    """

    examples_per_allele = pickle.load(open(os.path.join(MHCNUGGETS_HOME,
                                                        "data",
                                                        "production",
                                                        "mhcI",
                                                        "examples_per_allele.pkl"), 'rb'))

    alleles = sorted(examples_per_allele)

    # search for exact match
    match = exact_match(mhc, alleles)
    if match:
        return match

    # search for biological supertype match
    closest_mhc = closest_human_allele_supertype(mhc, examples_per_allele)
    if not closest_mhc:
        # search for name-based supertype match
        closest_mhc = closest_human_allele_name(mhc, examples_per_allele)
    return closest_mhc



def main():
    train_data = Dataset.from_csv(filename='data/production/curated_training_data.csv',
                                  sep=',',
                                  allele_column_name='mhc',
                                  peptide_column_name='peptide',
                                  affinity_column_name='IC50(nM)')

    # get the alleles for which training actually succeeded
    trained_alleles = []
    for trained_models in os.listdir('saves/production/'):
        trained_alleles.append(trained_models.split('.')[0])

    allele_example_dict = {}
    #for allele in sorted(set(train_data.alleles)):
    for allele in sorted(trained_alleles):
        n_training = len(train_data.get_allele(allele).peptides)
        allele_example_dict[allele] = n_training

    sorted_alleles = sorted(allele_example_dict.items(),
                            key=operator.itemgetter(1))

    for a in sorted_alleles:
        print(a)
    pickle.dump(allele_example_dict, open("data/production/examples_per_allele.pkl", 'wb'))


if __name__ == "__main__":
    main()
