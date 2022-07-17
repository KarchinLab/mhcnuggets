import varcode
from varcode import Variant                                                 
from pyensembl import ensembl_grch37                                           
from pyensembl import ensembl_grch38
import unicodedata  
import os
import argparse
try:
    import cPickle as pickle
except:
    import pickle
import pandas as pd
import numpy as np
import io


def read_vcf(path):
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('##')]
    return pd.read_csv(
        io.StringIO(u''.join(lines)),
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})

def get_context_peptides(chrom,pos,ref,alt,ensembl='ensembl_grch37'):
    #print chrom,pos,ref,alt
    chrom_=chrom.split('chr')[-1]
    try:
        if ensembl=='ensembl_grch37':
            variant=Variant(contig=chrom_,start=pos,ref=ref,alt=alt,ensembl=ensembl_grch37)
        elif ensembl=='ensembl_grch38':
            variant=Variant(contig=chrom_,start=pos,ref=ref,alt=alt,ensembl=ensembl_grch38)
        effects=variant.effects()
    except ValueError as err:
        print(err)
        return '','','','','','','','','','Error:'+str(err)

    top_effect=effects.top_priority_effect()
    location = top_effect.aa_mutation_start_offset
    var_gene=str(top_effect.gene_name).upper()
    var_gene_id=str(top_effect.gene_id)
    if hasattr(top_effect,'short_description'):
        short_description=top_effect.short_description
    else:
        short_description='p.*'

    if location:
        #print top_effect
        alt_AA=top_effect.aa_alt
        if hasattr(top_effect,'aa_ref'):
            ref_AA=top_effect.aa_ref
            ref_AA=str(ref_AA)
        else:
            ref_AA=''
        mut_seq,orig_seq='',''
        if type(top_effect) is varcode.effects.effect_classes.FrameShift:
            mut_seq=top_effect.mutant_protein_sequence
            orig_seq=top_effect.original_protein_sequence
            mutpos=location
            macrochange=True
        elif type(top_effect) is varcode.effects.effect_classes.StopLoss:
            mut_seq=top_effect.mutant_protein_sequence
            orig_seq=top_effect.original_protein_sequence
            mutpos=location
            macrochange=True
        elif location >= 24 and type(top_effect) is varcode.effects.effect_classes.Substitution:
            mut_seq = top_effect.mutant_protein_sequence[location-24:location+25]
            orig_seq = top_effect.original_protein_sequence[location-24:location+25]
            mutpos=24
            macrochange=False
        elif 0<location<24 and type(top_effect) is varcode.effects.effect_classes.Substitution:
            mut_seq = top_effect.mutant_protein_sequence[:location+25]
            orig_seq = top_effect.original_protein_sequence[:location+25]
            mutpos=len(mut_seq)-25
            macrochange=False
        if len(alt_AA)>1 or len(ref_AA)>1:
            mut_seq=top_effect.mutant_protein_sequence
            orig_seq=top_effect.original_protein_sequence
            mutpos=location
            macrochange=True
        if type(top_effect) is varcode.effects.effect_classes.ComplexSubstitution:
            print(top_effect)
        if not mut_seq or not orig_seq:
            mut_seq=top_effect.mutant_protein_sequence
            orig_seq=top_effect.original_protein_sequence
            mutpos=location
            macrochange=True
        return mut_seq,orig_seq,mutpos,ref_AA,alt_AA,var_gene_id,var_gene,short_description,macrochange,str(top_effect)
    else:
        print('no mutation location:',top_effect)
        return '','','','','',var_gene_id,var_gene,short_description,'',str(top_effect)


def read_patient_vcf(file_dir,expression_dir='',ensembl='ensembl_grch37'):
    df_=read_vcf(file_dir)
    df=df_.replace(np.nan,'',regex=True)
    sample_dict={}
    sample_fs_dict={}
    out_df=pd.DataFrame()

    for index,row in df.iterrows():
        
        chrom,pos,ref,alt_=(str(row['CHROM']),(row['POS']),
                                str(row['REF']),str(row['ALT']))
        alts=alt_.split(',')
        if len(alts)>1:
            print('warning:alternative has more than one alt',chrom,pos,ref,alt)
        for alt in alts:
            mut_seq,orig_seq,mutpos,ref_AA,alt_AA,var_gene_id,var_gene,short_description,macrochange,top_effect=get_context_peptides(chrom,pos,ref,alt,ensembl)
                
            if expression_dir:
                expression_data=pickle.load(open(expression_dir,'rb'))
                expression=expression_data[var_gene_id]
            else:
                expression='NA'
        
            if  macrochange==False and mut_seq and alt_AA:
                mutation='chr'+chrom+'_'+str(int(pos))+'_'+ref+'_'+alt
                sample_dict[mutation]={'mut_seq':mut_seq,'orig_seq':orig_seq,'mutpos':mutpos,'Gene':var_gene+':'+short_description}
            elif macrochange==True and mut_seq and alt_AA:
                mutation='chr'+chrom+'_'+str(int(pos))+'_'+ref+'_'+alt
                sample_fs_dict[mutation]={'mut_seq':mut_seq,'orig_seq':orig_seq,'Gene':var_gene+':'+short_description,'mutpos':mutpos}
        
            top_e=str(top_effect).split("(")[0]
            row_df=df.iloc[[index]]
            row_df['MUT_SEQ']=mut_seq
            row_df['ORIG_SEQ']=orig_seq
            row_df['AFFECTED_GENE']=var_gene
            row_df['EXPRESSION']=expression
            row_df['AA_EFFECTS']=short_description
            row_df['AA_MUTPOS']=mutpos
            row_df['EFFECT_TYPE']=top_e
            out_df=pd.concat([out_df,row_df]).reset_index(drop=True)
    return sample_dict,sample_fs_dict,out_df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                        type=str, required=True,
                        help=('vcf_dir'))
    parser.add_argument('-e','--exp_pkl',default='',
                        type=str,required=False,
                        help=('pickle_dir_for_RNA_FPKM'))
    parser.add_argument('-o', '--out_dir',
                        type=str, required=True,
                        help='Path to store candidate peptides')
    args = parser.parse_args()
    return vars(args)


def main(args):
    file_dir=args['input']
    out_dir=args['out_dir']
    RNA_dir=args['exp_pkl']
    sample_dict,sample_fs_dict,out_df=read_patient_vcf(file_dir,RNA_dir)
    pickle.dump(sample_dict,open(out_dir+'.pkl','wb'))
    pickle.dump(sample_fs_dict,open(out_dir+'_fs.pkl','wb'))
if __name__=='__main__': main(parse_args())        
                
        
