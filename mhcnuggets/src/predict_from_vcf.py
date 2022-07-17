'''
Predict IC50s for a batch of peptides
from mutation call vcfs with a trained model

Xiaoshan (Melody) Shao
xshao5@jhu.edu
'''

from mhcnuggets.src.extract_pep_sequences import read_patient_vcf
from mhcnuggets.src.get_candidate_neoantigens import get_candidate_neoantigens,output_pm,output_mc
from mhcnuggets.src.predict import predict

import os
import sys
import pandas as pd
import argparse
MHCNUGGETS_HOME = os.path.join(os.path.dirname(__file__), '..')
try:
    import cPickle as pickle
except:
    import pickle


def predict_from_vcf(vcf_path,mhcs,class_,output_path,expr_path=None,pep_length_class='both',
                     pickle_path='data/production/examples_per_allele.pkl',
                     model='lstm',model_weights_path="saves/production/",
                     mass_spec=False, ic50_threshold=500, max_ic50=50000, embed_peptides=False,
                     binary_preds=False, ba_models=False, rank_output=False,ensembl='ensembl_grch37',
                     hp_ic50s_cI_pickle_path='data/production/mhcI/hp_ic50s_cI.pkl',
                     hp_ic50s_positions_cI_pickle_path='data/production/mhcI/hp_ic50s_positions_cI.pkl',
                     hp_ic50s_hp_lengths_cI_pickle_path='data/production/mhcI/hp_ic50s_hp_lengths_cI.pkl',
                     hp_ic50s_first_percentiles_cI_pickle_path='data/production/mhcI/hp_ic50s_first_percentiles_cI.pkl',
                     hp_ic50s_cII_pickle_path='data/production/mhcII/hp_ic50s_cII.pkl',
                     hp_ic50s_positions_cII_pickle_path='data/production/mhcII/hp_ic50s_positions_cII.pkl',
                     hp_ic50s_hp_lengths_cII_pickle_path='data/production/mhcII/hp_ic50s_hp_lengths_cII.pkl',
                     hp_ic50s_first_percentiles_cII_pickle_path='data/production/mhcII/hp_ic50s_first_percentiles_cII.pkl'):
    
    '''
    Prediction from vcf protocol
    '''
    
    #read mutation calls and extract context AA sequence of a mutation
    pkl,pkl_fs,meta_df=read_patient_vcf(vcf_path,expr_path,ensembl)
    samplename=vcf_path.split("/")[-1].split('.')[0]

    #output intermediate files that contain all the context sequences of the mutation calls
    #two types of files are outputed: 
    #1. machine readable pickle files for all expressed mutation context AA
    #2. csv file with all the mutation and context AA & expression information
    PKL_DIR=output_path+'/mutation_calls_context_AA_seq_pkl'
    META_DIR=output_path+'/mutation_calls_context_AA_seq_csv'

    if not os.path.exists(PKL_DIR):
        os.mkdir(PKL_DIR)
    if not os.path.exists(META_DIR):
        os.mkdir(META_DIR)
    
    pickle.dump(pkl,open(PKL_DIR+'/'+samplename+'.pkl','wb'))
    pickle.dump(pkl_fs,open(PKL_DIR+'/'+samplename+'_fs.pkl','wb'))
    meta_df.to_csv(META_DIR+'/'+samplename+'.csv',index=False)

    #get candidate neoantigens by windowing around mutation_AA
    PEPS_DIR=output_path+'/candidate_neoantigens'
    if not os.path.exists(PEPS_DIR):
        os.mkdir(PEPS_DIR)

    if pep_length_class=='both':
        MHC_CLASS_DIR=[os.path.join(PEPS_DIR,'cI'),os.path.join(PEPS_DIR,'cII')]
    elif pep_length_class=='I':
        MHC_CLASS_DIR=[os.path.join(PEPS_DIR,'cI')]
    elif pep_length_class=='II':
        MHC_CLASS_DIR=[os.path.join(PEPS_DIR,'cII')]
    
    for directory in MHC_CLASS_DIR:
        if not os.path.exists(directory):
            os.mkdir(directory)

    PM_DIRs=[os.path.join(directory,'point_AA_mutations') for directory in MHC_CLASS_DIR]
    MC_DIRs=[os.path.join(directory,'multi_AA_mutations') for directory in MHC_CLASS_DIR]
    
    for directory in PM_DIRs+MC_DIRs:
        if not os.path.exists(directory):
            os.mkdir(directory)

    TEMP_CI_PM_FILES,TEMP_CII_PM_FILES,TEMP_CI_MC_FILES,TEMP_CII_MC_FILES=[],[],[],[]
    for i in range(len(PM_DIRs)):
        out_pm_dir,out_mc_dir=PM_DIRs[i],MC_DIRs[i]
        if '/cI/' in out_pm_dir and '/cI/' in out_mc_dir:
            pm_mutfile,pm_reffile,_,mc_mutfile,mc_reffile,_,_=get_candidate_neoantigens(PKL_DIR,'I',samplename,out_pm_dir,out_mc_dir)
            TEMP_CI_PM_FILES.extend([pm_mutfile,pm_reffile])
            TEMP_CI_MC_FILES.extend([mc_mutfile,mc_reffile])
            
            
        elif '/cII/'in out_pm_dir and '/cII/' in out_mc_dir:
            pm_mutfile,pm_reffile,_,mc_mutfile,mc_reffile,_,_=get_candidate_neoantigens(PKL_DIR,'II',samplename,out_pm_dir,out_mc_dir)
            TEMP_CII_PM_FILES.extend([pm_mutfile,pm_reffile])
            TEMP_CII_MC_FILES.extend([mc_mutfile,mc_reffile])
           
 
    #predict the lists of peptides
    PRED_RESULT_DIR=output_path+'/prediction_on_candidate_peptides'
    if not os.path.exists(PRED_RESULT_DIR):
        os.mkdir(PRED_RESULT_DIR)
    if not os.path.exists(PRED_RESULT_DIR+'/c'+class_):
        os.mkdir(PRED_RESULT_DIR+'/c'+class_)

    MHC_PM_PRED_DIR=os.path.join(PRED_RESULT_DIR,'c'+class_,'point_AA_mutations')
    MHC_MC_PRED_DIR=os.path.join(PRED_RESULT_DIR,'c'+class_,'multi_AA_mutations')
    if not os.path.exists(MHC_PM_PRED_DIR):
        os.mkdir(MHC_PM_PRED_DIR)
    if not os.path.exists(MHC_MC_PRED_DIR):
        os.mkdir(MHC_MC_PRED_DIR)
    
    if not os.path.exists(MHC_PM_PRED_DIR+'/'+samplename):
        os.mkdir(MHC_PM_PRED_DIR+'/'+samplename)
    if not os.path.exists(MHC_MC_PRED_DIR+'/'+samplename):
        os.mkdir(MHC_MC_PRED_DIR+'/'+samplename)
   
    hlas=mhcs.split(',')    
    for hla in hlas:
        print(hla)
        if class_=='I':
            pm_f1=TEMP_CI_PM_FILES[0]
            pm_f2=TEMP_CI_PM_FILES[1]
            mc_f1=TEMP_CI_MC_FILES[0]
            mc_f2=TEMP_CI_MC_FILES[1]
        elif class_=='II':
            pm_f1,pm_f2=TEMP_CII_PM_FILES[0],TEMP_CII_PM_FILES[1]
            mc_f1,mc_f2=TEMP_CII_MC_FILES[0],TEMP_CII_MC_FILES[1]
        
        out_pm_f1=MHC_PM_PRED_DIR+'/'+samplename+'/'+hla+'.'+pm_f1.split('.')[-1].split('peps')[0]+'preds'
        out_pm_f2=MHC_PM_PRED_DIR+'/'+samplename+'/'+hla+'.'+pm_f2.split('.')[-1].split('peps')[0]+'preds'
        out_mc_f1=MHC_MC_PRED_DIR+'/'+samplename+'/'+hla+'.'+mc_f1.split('.')[-1].split('peps')[0]+'preds'
        out_mc_f2=MHC_MC_PRED_DIR+'/'+samplename+'/'+hla+'.'+mc_f2.split('.')[-1].split('peps')[0]+'preds'
        try:
            predict(class_=class_,peptides_path=pm_f1,mhc=hla,output=out_pm_f1,pickle_path=pickle_path,
                    model=model,model_weights_path=model_weights_path,mass_spec=mass_spec,
                    ic50_threshold=ic50_threshold,max_ic50=max_ic50,embed_peptides=embed_peptides,
                    binary_preds=binary_preds,ba_models=ba_models,rank_output=rank_output,
                    hp_ic50s_cI_pickle_path=hp_ic50s_cI_pickle_path,
                    hp_ic50s_positions_cI_pickle_path=hp_ic50s_positions_cI_pickle_path,
                    hp_ic50s_hp_lengths_cI_pickle_path=hp_ic50s_hp_lengths_cI_pickle_path,
                    hp_ic50s_first_percentiles_cI_pickle_path=hp_ic50s_first_percentiles_cI_pickle_path,
                    hp_ic50s_cII_pickle_path=hp_ic50s_cII_pickle_path,
                    hp_ic50s_positions_cII_pickle_path=hp_ic50s_positions_cII_pickle_path,
                    hp_ic50s_hp_lengths_cII_pickle_path=hp_ic50s_hp_lengths_cII_pickle_path,
                    hp_ic50s_first_percentiles_cII_pickle_path=hp_ic50s_first_percentiles_cII_pickle_path)
            predict(class_=class_,peptides_path=pm_f2,mhc=hla,output=out_pm_f2,pickle_path=pickle_path,
                    model=model,model_weights_path=model_weights_path,mass_spec=mass_spec,
                    ic50_threshold=ic50_threshold,max_ic50=max_ic50,embed_peptides=embed_peptides,
                    binary_preds=binary_preds,ba_models=ba_models,rank_output=rank_output,
                    hp_ic50s_cI_pickle_path=hp_ic50s_cI_pickle_path,
                    hp_ic50s_positions_cI_pickle_path=hp_ic50s_positions_cI_pickle_path,
                    hp_ic50s_hp_lengths_cI_pickle_path=hp_ic50s_hp_lengths_cI_pickle_path,
                    hp_ic50s_first_percentiles_cI_pickle_path=hp_ic50s_first_percentiles_cI_pickle_path,
                    hp_ic50s_cII_pickle_path=hp_ic50s_cII_pickle_path,
                    hp_ic50s_positions_cII_pickle_path=hp_ic50s_positions_cII_pickle_path,
                    hp_ic50s_hp_lengths_cII_pickle_path=hp_ic50s_hp_lengths_cII_pickle_path,
                    hp_ic50s_first_percentiles_cII_pickle_path=hp_ic50s_first_percentiles_cII_pickle_path)
            predict(class_=class_,peptides_path=mc_f1,mhc=hla,output=out_mc_f1,pickle_path=pickle_path,
                    model=model,model_weights_path=model_weights_path,mass_spec=mass_spec,
                    ic50_threshold=ic50_threshold,max_ic50=max_ic50,embed_peptides=embed_peptides,
                    binary_preds=binary_preds,ba_models=ba_models,rank_output=rank_output,
                    hp_ic50s_cI_pickle_path=hp_ic50s_cI_pickle_path,
                    hp_ic50s_positions_cI_pickle_path=hp_ic50s_positions_cI_pickle_path,
                    hp_ic50s_hp_lengths_cI_pickle_path=hp_ic50s_hp_lengths_cI_pickle_path,
                    hp_ic50s_first_percentiles_cI_pickle_path=hp_ic50s_first_percentiles_cI_pickle_path,
                    hp_ic50s_cII_pickle_path=hp_ic50s_cII_pickle_path,
                    hp_ic50s_positions_cII_pickle_path=hp_ic50s_positions_cII_pickle_path,
                    hp_ic50s_hp_lengths_cII_pickle_path=hp_ic50s_hp_lengths_cII_pickle_path,
                    hp_ic50s_first_percentiles_cII_pickle_path=hp_ic50s_first_percentiles_cII_pickle_path)
            predict(class_=class_,peptides_path=mc_f2,mhc=hla,output=out_mc_f2,pickle_path=pickle_path,
                    model=model,model_weights_path=model_weights_path,mass_spec=mass_spec,
                    ic50_threshold=ic50_threshold,max_ic50=max_ic50,embed_peptides=embed_peptides,
                    binary_preds=binary_preds,ba_models=ba_models,rank_output=rank_output,
                    hp_ic50s_cI_pickle_path=hp_ic50s_cI_pickle_path,
                    hp_ic50s_positions_cI_pickle_path=hp_ic50s_positions_cI_pickle_path,
                    hp_ic50s_hp_lengths_cI_pickle_path=hp_ic50s_hp_lengths_cI_pickle_path,
                    hp_ic50s_first_percentiles_cI_pickle_path=hp_ic50s_first_percentiles_cI_pickle_path,
                    hp_ic50s_cII_pickle_path=hp_ic50s_cII_pickle_path,
                    hp_ic50s_positions_cII_pickle_path=hp_ic50s_positions_cII_pickle_path,
                    hp_ic50s_hp_lengths_cII_pickle_path=hp_ic50s_hp_lengths_cII_pickle_path,
                    hp_ic50s_first_percentiles_cII_pickle_path=hp_ic50s_first_percentiles_cII_pickle_path)
        except:
            continue


def parse_args():
    '''
    Parse user arguments
    '''

    info = '''Going from a VCF file, this function would extract context mutant and wt 
    AA sequences of each mutation, generating files stored under /mutation_calls_context_AA_seq.
    For each mutation, candidate peptides are generated by windowing around the AA mutation. 
    Results are stored under /candidate_neoantigens. Each candidate peptide generated would then
    be predicted againsted a MHC model of choice for its IC50 value, stored under 
    /prediction_on_candidate_peptides'''

    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-v', '--vcf',
                        type=str, required=True,
                        help='VCF file of mutation calls')
    
    parser.add_argument('-c', '--class',
                        type=str, required=True,
                        help='MHC class - options are I or II')

    parser.add_argument('-o', '--output',
                        type=str, required=True,
                        help='Path to a output directory,'+
                        'where all results will be stored')

    parser.add_argument('-p', '--pep_length_class',
                        type=str, required=False,default='both',
                        help='class(s) of candidate peptides length'+ 
                        'to generate for - options are I,II,both.'+
                        'I generates 8-12mer peptides, II generate 12-20mer peptides')

    parser.add_argument('-a', '--allele',
                        type=str, required=True,
                        help = 'Allele(s) used for prediction,'+
                        'if multiple, please separate by comma')

    parser.add_argument('-X', '--expr_path',
                        type=str, required=False,default=None,
                        help='a pickle file that contains RNA expression info'+ 
                        'for genes of interest related to mutation calls.'+
                        'pickle file should be formatted as such:{RNA_trascriptID:expr_level}')

    parser.add_argument('-m', '--model',
                        type=str, default='lstm',
                        help='Type of MHCnuggets model used to predict' +
                              'options are just lstm for now')

    parser.add_argument('-s', '--model_weights_path',
                        type=str, required=False, default='saves/production/',
                        help='Path to which the model weights are saved')

    parser.add_argument('-k', '--pickle_path',
                        type=str, required=False, default='data/production/examples_per_allele.pkl',
                        help='Path to which the pickle file is saved')

    parser.add_argument('-e', '--mass_spec', default=False, type=lambda x: (str(x).lower()== 'true'),
                        help='Train on mass spec data if True, binding affinity data if False')

    parser.add_argument('-l', '--ic50_threshold',
                        type=int, default=500,
                        help='Threshold on ic50 (nM) that separates binder/non-binder')

    parser.add_argument('-x', '--max_ic50',
                        type=int, default=50000,
                        help='Maximum ic50 value')

    parser.add_argument('-q', '--embed_peptides',
                        type=bool, default=False,
                        help='Embedding of peptides used')

    parser.add_argument('-B', '--binary_predictions',
                        type=bool, default=False,
                        help='Binary prediction')

    parser.add_argument('-M', '--ba_models', default=False, type=bool,
                        help='Use binding affinity trained models only instead of mass spec trained models')

    parser.add_argument('-r', '--rank_output', type=lambda x: (str(x).lower()== 'true'),
                        default=False,
                        help='Additionally write output files of predicted peptide ic50 binding '+
                        'percentiles compared to human proteome peptides')

    parser.add_argument('-g', '--genome', type=str,default='ensembl_grch37',
                        help='Genome ensembl of choice, default is ensembl_grch37,'+
                        'could pick between "ensembl_grch37" or "ensembl_grch38"')
    args = parser.parse_args()
    return vars(args)


def main():
    '''
    Main function
    '''

    opts = parse_args()
    predict_from_vcf(vcf_path=opts['vcf'],pep_length_class=opts['pep_length_class'],
                     expr_path=opts['expr_path'],model=opts['model'], class_=opts['class'],
                     model_weights_path=opts['model_weights_path'], pickle_path=opts['pickle_path'],
                     mhcs=opts['allele'], output_path=opts['output'],mass_spec=opts['mass_spec'],
                     ic50_threshold=opts['ic50_threshold'],
                     max_ic50=opts['max_ic50'], embed_peptides= opts['embed_peptides'],
                     binary_preds=opts['binary_predictions'],ba_models=opts['ba_models'],
                     rank_output=opts['rank_output'],ensembl=opts['genome'])


if __name__ == '__main__':
    main()
