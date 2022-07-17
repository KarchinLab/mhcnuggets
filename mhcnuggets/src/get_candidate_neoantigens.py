try:
    import cPickle as pickle
except:
    import pickle
import os
import argparse
import numpy as np
import pandas as pd

MHCI_pep_length=range(8,13)
MHCII_pep_length=range(12,21)

def load_pickles(file_dir,sample):
    sample_dict=pickle.load(open(os.path.join(file_dir,sample+'.pkl'),'rb'))
    sample_fs_dict=pickle.load(open(os.path.join(file_dir,sample+'_fs.pkl'),'rb'))
    return sample_dict,sample_fs_dict

def get_pep_sequence(seq,length,mutpos):
    peplist,mutloc=[],[]
    for i in range(0,len(seq)-length+1):
        pep=seq[i:i+length]
        pep_mutloc=max(length-i,0)
        if 'U' not in pep:
            peplist.append(pep)
            mutloc.append(pep_mutloc)
    return peplist,mutloc

def window_around_mutation(mutpos,mut_seq,orig_seq,lengths):
    ref_peps,mut_peps,pep_mutpos=[],[],[]
    for l in lengths:
        mut_seq_l=mut_seq[max(mutpos-l+1,0):mutpos+l]
        ref_seq_l=orig_seq[max(mutpos-l+1,0):mutpos+l]
        new_mutpos=max(mutpos-l+1,0)
        mut_seq_peps,mutloc=get_pep_sequence(mut_seq_l,l,new_mutpos)
        ref_seq_peps,_=get_pep_sequence(ref_seq_l,l,new_mutpos)
        mut_peps.extend(mut_seq_peps)
        ref_peps.extend(ref_seq_peps)
        pep_mutpos.extend(mutloc)
    return mut_peps,ref_peps,pep_mutpos
        
def get_point_mut_seqs(sample_pickle,pep_lengths):
    peptides={}
    for mutation in sample_pickle:
        mut_seq=str(sample_pickle[mutation]['mut_seq'])
        orig_seq=str(sample_pickle[mutation]['orig_seq'])
        gene=str(sample_pickle[mutation]['Gene'])
        mutpos=int(sample_pickle[mutation]['mutpos'])
        mut_peps,ref_peps,pep_mutpos=window_around_mutation(mutpos,mut_seq,orig_seq,pep_lengths)
        peptides[mutation]={'mut':mut_peps,'ref':ref_peps,'Gene':gene,'mutloc':pep_mutpos}
    return peptides

def window_multi_change_mutations(mutpos,orig_seq,mut_seq,lengths):
    ref_peps,mut_peps_all,mut_all_loc=[],[],[]
    for l in lengths:
        mut_seq_l=mut_seq[max(mutpos-l+1,0):]
        ref_seq_l=orig_seq[max(mutpos-l+1,0):]
        new_mutpos=max(mutpos-l+1,0)
        ref_seq_peps,_=get_pep_sequence(orig_seq,l,new_mutpos)
        mut_seq_peps,mut_mutloc=get_pep_sequence(mut_seq,l,new_mutpos)
        ref_peps.extend(ref_seq_peps)
        mut_peps_all.extend(mut_seq_peps)
        mut_all_loc.extend(mut_mutloc)
    ref_peps_u=list(set(ref_peps))
    mask=[False if pep in ref_peps_u else True for pep in mut_peps_all]
    mut_peps=np.array(mut_peps_all)[mask]
    mut_peps_mutloc=np.array(mut_all_loc)[mask]
    return list(mut_peps),ref_peps,list(mut_peps_mutloc)

def get_macro_change_seqs(sample_fs_pickle,pep_lengths):
    peptides={}
    for mutation in sample_fs_pickle:
        mut_seq=str(sample_fs_pickle[mutation]['mut_seq'])
        orig_seq=str(sample_fs_pickle[mutation]['orig_seq'])
        gene=str(sample_fs_pickle[mutation]['Gene'])
        mutpos=int(sample_fs_pickle[mutation]['mutpos'])
        mut_peps,ref_peps,pep_mutpos=window_multi_change_mutations(mutpos,orig_seq,mut_seq,pep_lengths)
        peptides[mutation]={'mut':mut_peps,'ref':ref_peps,'Gene':gene,'mutloc':pep_mutpos}
    return peptides

def get_candidate_neoantigens(file_dir,class_,sample,out_pm_dir,out_mc_dir):
    sample_dict,sample_fs_dict=load_pickles(file_dir, sample)
    if class_=='I':
        peps_pm=get_point_mut_seqs(sample_dict,MHCI_pep_length)
        peps_mc=get_macro_change_seqs(sample_fs_dict,MHCI_pep_length)
    if class_=='II':
        peps_pm=get_point_mut_seqs(sample_dict,MHCII_pep_length)
        peps_mc=get_macro_change_seqs(sample_fs_dict,MHCII_pep_length)
    pm_mutfile,pm_reffile,pm_sumfile=output_pm(peps_pm,out_pm_dir+'/peptides',out_pm_dir+'/summaryfiles',sample)
    mc_mutfile,mc_reffile,mut_sumfile,ref_sumfile=output_mc(peps_mc,out_mc_dir+'/peptides',out_mc_dir+'/summaryfiles',sample)
    return pm_mutfile,pm_reffile,pm_sumfile,mc_mutfile,mc_reffile,mut_sumfile,ref_sumfile

def output_pm(sample_dict,out_pep_dir,out_sum_dir,sample_name):
    for directory in [out_sum_dir,out_pep_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)
    mutpeps,refpeps=[],[]
    summary_df=pd.DataFrame({'mutant':[],'wt':[],'mutation_position':[],'AA_mutations':[],
                             'gene_mutations':[]})
    for mutation in sample_dict:
        if len(sample_dict[mutation]['mut'])==len(sample_dict[mutation]['ref']):
            mutpeps.extend(sample_dict[mutation]['mut'])
            refpeps.extend(sample_dict[mutation]['ref'])
            mut_df=pd.DataFrame(sample_dict[mutation]).rename(columns={'mut':'mutant',
                                                                       'ref':'wt', 
                                                                       'mutloc':'mutation_position',
                                                                       'Gene':'AA_mutations'})
            mut_df['gene_mutations']=mutation
            summary_df=pd.concat([summary_df,mut_df])
    fm=open(out_pep_dir+'/'+sample_name+'.mutpeps','w')
    fr=open(out_pep_dir+'/'+sample_name+'.refpeps','w')
    fm.write('\n'.join(mutpeps)+'\n')
    fr.write('\n'.join(refpeps)+'\n')
    fm.close()
    fr.close()
    summary_df.to_csv(out_sum_dir+'/'+sample_name+'.summary',index=False)
    return out_pep_dir+'/'+sample_name+'.mutpeps',out_pep_dir+'/'+sample_name+'.refpeps',out_sum_dir+'/'+sample_name+'.summary'

def output_mc(sample_dict,out_pep_dir,out_sum_dir,sample_name):
    for directory in [out_sum_dir,out_pep_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)
    mutpeps,refpeps=[],[]
    summary_mut_df=pd.DataFrame({'peptide':[],'AA_mutations':[],'gene_mutations':[],
                                 'mutation_position':[]})
    summary_ref_df=pd.DataFrame({'peptide':[],'AA_mutations':[],'gene_mutations':[],
                                'mutation_position':[]})
    for mutation in sample_dict:
        mutpeps.extend(sample_dict[mutation]['mut'])
        refpeps.extend(sample_dict[mutation]['ref'])
        mut_dict={'peptide':sample_dict[mutation]['mut'],'AA_mutations':sample_dict[mutation]['Gene'],
                  'gene_mutations':mutation,'mutation_position':sample_dict[mutation]['mutloc']}
        ref_dict={'peptide':sample_dict[mutation]['ref'],'AA_mutations':sample_dict[mutation]['Gene'],
                  'gene_mutations':mutation}
        summary_mut_df=pd.concat([summary_mut_df,pd.DataFrame(mut_dict)])
        summary_ref_df=pd.concat([summary_ref_df,pd.DataFrame(ref_dict)])
    fm=open(out_pep_dir+'/'+sample_name+'.mutpeps','w')
    fr=open(out_pep_dir+'/'+sample_name+'.refpeps','w')
    fm.write('\n'.join(mutpeps)+'\n')
    fr.write('\n'.join(refpeps)+'\n')
    fm.close()
    fr.close()
    summary_mut_df.to_csv(out_sum_dir+'/'+sample_name+'_mutpeps.summary',index=False)
    summary_ref_df.to_csv(out_sum_dir+'/'+sample_name+'_refpeps.summary',index=False)
    return out_pep_dir+'/'+sample_name+'.mutpeps',out_pep_dir+'/'+sample_name+'.refpeps',out_sum_dir+'/'+sample_name+'_mutpeps.summary',out_sum_dir+'/'+sample_name+'_refpeps.summary'
